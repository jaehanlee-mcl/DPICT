import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
import numpy as np
import scipy

from scipy.stats import norm
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf

from torchvision.transforms import ToPILImage

multiplier = -scipy.stats.norm.ppf(1e-09 / 2)
L = 20
mode = 3
opt_pnum = 6
pnum_btw_trit = 48
pnum_part = 1.0

def compress_DPICT(y, means_hat, scales_hat):
    device = y.device

    scales_hat = scales_hat.clamp_(min=0.04)
    tail = scales_hat * multiplier * 2
    l_per_ele = torch.ceil(torch.log(tail) / torch.log(torch.Tensor([mode]).squeeze())).int()
    l_per_ele = torch.clamp(l_per_ele, 1, l_per_ele.max().item())

    max_L = min(l_per_ele.max(), L).item()
    if torch.sum(l_per_ele == l_per_ele.max()) < 2:
        max_L = int(min(l_per_ele.max() - 1, L))
        l_per_ele = torch.clamp(l_per_ele, 1, max_L)

    y_strings = [[] for _ in range(max_L)]

    Nary_tensor = torch.zeros(list(y.shape) + [max_L]).int().to(device)
    symbol_tensor = torch.round(y - means_hat).int() + torch.div(mode ** l_per_ele, 2, rounding_mode="floor")
    symbol_tensor = torch.clamp(symbol_tensor, min=torch.zeros(y.shape).int().to(device), max=3 ** l_per_ele - 1)

    for i in range(1, max_L + 1):
        Nary_tensor[:, :, :, :, i - 1] = torch.div(symbol_tensor, (mode ** (max_L - i)), rounding_mode="floor")

        symbol_tensor = symbol_tensor % (mode ** (max_L - i))
    Nary_tensor = Nary_tensor.view(-1, max_L)
    del symbol_tensor, tail
    torch.cuda.empty_cache()

    pmfs_list = []
    xpmfs_list = []
    x2pmfs_list = []
    idx_ts_list = []
    for i in range(1, max_L + 1):
        pmf_length = mode ** i
        pmf_center = pmf_length // 2
        samples = torch.abs(torch.arange(pmf_length, device=device).repeat((l_per_ele == i).sum(), 1) - pmf_center)
        upper = _standardized_cumulative((0.5 - samples) / scales_hat.view(-1, 1)[l_per_ele.view(-1) == i])
        lower = _standardized_cumulative((-0.5 - samples) / scales_hat.view(-1, 1)[l_per_ele.view(-1) == i])
        pmfs_ = upper - lower
        pmfs_ = (pmfs_ + 1e-10) / (pmfs_ + 1e-10).sum(dim=-1).unsqueeze(-1)
        pmfs_list.insert(0, pmfs_.clone())
        del upper, lower, samples
        torch.cuda.empty_cache()
        idx_tmp = torch.arange(mode ** i, device=device).repeat(pmfs_.size(0), 1)
        xpmfs_ = pmfs_ * idx_tmp
        xpmfs_list.insert(0, xpmfs_.clone())
        x2pmfs_ = pmfs_ * torch.pow(idx_tmp, 2)
        x2pmfs_list.insert(0, x2pmfs_.clone())
        idx_ts_list.insert(0, torch.ones_like(pmfs_list[0], device=device))
        del idx_tmp, pmfs_, xpmfs_, x2pmfs_
        torch.cuda.empty_cache()

    # reconstruction value per trit, and predicted reconstruction value.
    for i in range(max_L):
        encoder = BufferedRansEncoder()
        if i < max_L - opt_pnum:
            cond_pmfs = list(map(lambda p, idx: (p * idx).view(p.size(0), 3, -1).sum(-1) / (p * idx).view(p.size(0), 1, -1).sum(-1), pmfs_list[:i + 1], idx_ts_list[:i + 1]))
            tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, cond_pmfs))
            pmf_length = list(map(lambda tm: torch.zeros_like(tm).int().to(device) + mode, tail_mass))
            cond_cdf = torch.cat(list(map(lambda p, tm, l: _pmf_to_cdf(p, tm, l, mode), cond_pmfs, tail_mass, pmf_length)), dim=0).tolist()

            total_symbols_list = torch.cat([Nary_tensor[l_per_ele.view(-1) == max_L - j, i] - (mode // 2) for j in range(i + 1)]).tolist()
            indexes_list = list(range(len(total_symbols_list)))
            cdf_lengths = [mode + 2 for _ in range(len(total_symbols_list))]

            offsets = [-(mode // 2) for _ in range(len(total_symbols_list))]
            encoder.encode_with_indexes(
                total_symbols_list, indexes_list, cond_cdf, cdf_lengths, offsets
            )
            del cond_pmfs, tail_mass, pmf_length, cond_cdf
            torch.cuda.empty_cache()
            y_strings[i].append(encoder.flush())

            for j in range(i + 1):
                Nary_part = Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i:i + 1]
                tmp_ = torch.arange(mode ** (max_L - i), device=device).repeat(Nary_part.size(0), 1).int()
                idx_ts_list[j] *= (torch.div((tmp_ % (mode ** (max_L - i))), (mode ** (max_L - 1 - i)), rounding_mode="floor") == Nary_part)
                nz_idx = idx_ts_list[j].nonzero(as_tuple=True)
                pmfs_list[j] = pmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                xpmfs_list[j] = xpmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                x2pmfs_list[j] = x2pmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                idx_ts_list[j] = idx_ts_list[j][nz_idx].view(pmfs_list[j].size(0), -1)

        else:
            pmfs_list_l = pmfs_list[:i + 1]
            xpmfs_list_l = xpmfs_list[:i + 1]
            x2pmfs_list_l = x2pmfs_list[:i + 1]
            m_old = list(map(lambda x, y: x.sum(dim=-1) / y.sum(dim=-1), xpmfs_list_l, pmfs_list_l))
            D_old = list(map(lambda x2p, xp, p, m: (x2p.sum(-1) - 2 * m * xp.sum(-1) + (m ** 2) * p.sum(-1)) / p.sum(-1), x2pmfs_list_l, xpmfs_list_l, pmfs_list_l, m_old))

            pmfs_cond_list_l = list(map(lambda x: x.view(x.size(0), -1, mode ** (max_L - 1 - i)).sum(-1), pmfs_list_l))
            xpmfs_cond_list_l = list(map(lambda xp: xp.view(xp.size(0), -1, mode ** (max_L - 1 - i)).sum(-1), xpmfs_list_l))
            x2pmfs_cond_list_l = list(map(lambda x: x.view(x.size(0), -1, mode ** (max_L - 1 - i)).sum(-1), x2pmfs_list_l))

            m_new = list(map(lambda xp, p: xp / p, xpmfs_cond_list_l, pmfs_cond_list_l))
            D_new = list(map(lambda x2p, xp, p, m, fullp: ((x2p - 2 * m * xp + (m ** 2) * p) / fullp.sum(-1).view(-1, 1)).sum(-1),
                             x2pmfs_cond_list_l, xpmfs_cond_list_l, pmfs_cond_list_l, m_new, pmfs_list_l))
            delta_D = list(map(lambda old, new: (new - old).clamp_(max=0), D_old, D_new))
            pmf_cond_list_l_norm = list(map(lambda p: p / p.sum(-1).view(-1, 1), pmfs_cond_list_l))
            delta_R = list(map(lambda p: (-p * torch.log2(p)).sum(-1), pmf_cond_list_l_norm))
            delta_R = list(map(lambda h: h * (h >= 0), delta_R))

            optim_tensor = torch.cat(list(map(lambda D, R: -(D / R), delta_D, delta_R))).clamp_(min=0)

            pmfs_norm = list(map(lambda p: p / p.sum(-1).view(-1, 1), pmfs_cond_list_l))

            tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
            pmf_length = list(map(lambda tm: torch.zeros_like(tm).int().to(device) + mode, tail_mass))
            cond_cdf = torch.cat(list(map(lambda p, tm, l: _pmf_to_cdf(p, tm, l, mode), pmfs_norm, tail_mass, pmf_length)), dim=0)
            cond_cdf = cond_cdf[torch.argsort(optim_tensor, descending=True)].tolist()
            total_symbols_list = torch.cat([Nary_tensor[l_per_ele.view(-1) == max_L - j, i] - (mode // 2) for j in range(i + 1)])
            total_symbols_list = total_symbols_list[torch.argsort(optim_tensor, descending=True)].tolist()
            total_symbols = len(total_symbols_list)
            cdf_lengths = [mode + 2 for _ in range(total_symbols)]
            offsets = [-(mode // 2) for _ in range(total_symbols)]

            torch.cuda.empty_cache()

            # sl = total_symbols // pnum_btw_trit
            pnum_part = _pnum_part(i, max_L)
            points_num = math.ceil(pnum_btw_trit * pnum_part)

            sl = total_symbols // points_num

            for point in range(points_num):
                if point == points_num - 1:
                    symbols_list = total_symbols_list[point * sl:]
                    indexes_list = list(range(len(symbols_list)))
                    encoder.encode_with_indexes(
                        symbols_list,
                        indexes_list,
                        cond_cdf[point * sl:],
                        cdf_lengths[point * sl:],
                        offsets[point * sl:]
                    )
                    y_strings[i].append(encoder.flush())
                    break
                symbols_list = total_symbols_list[point * sl:(point + 1) * sl]
                indexes_list = list(range(len(symbols_list)))
                encoder.encode_with_indexes(
                    symbols_list,
                    indexes_list,
                    cond_cdf[point * sl:(point + 1) * sl],
                    cdf_lengths[point * sl:(point + 1) * sl],
                    offsets[point * sl:(point + 1) * sl]
                )
                y_strings[i].append(encoder.flush())
                encoder = BufferedRansEncoder()

            for j in range(i + 1):
                Nary_part = Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i:i + 1]
                tmp_ = torch.arange(mode ** (max_L - i), device=device).repeat(Nary_part.size(0), 1).int()
                idx_ts_list[j] *= (torch.div((tmp_ % (mode ** (max_L - i))), (mode ** (max_L - 1 - i)), rounding_mode="floor") == Nary_part)
                nz_idx = idx_ts_list[j].nonzero(as_tuple=True)
                pmfs_list[j] = pmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                xpmfs_list[j] = xpmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                x2pmfs_list[j] = x2pmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                idx_ts_list[j] = idx_ts_list[j][nz_idx].view(pmfs_list[j].size(0), -1)

    return y_strings

def decompress_DPICT(y_strings, means_hat, scales_hat):
    device = means_hat.device
    y_hats = []
    scales_hat = scales_hat.clamp_(min=0.04)

    tail = scales_hat * multiplier * 2

    l_per_ele = torch.ceil(torch.log(tail) / torch.log(torch.Tensor([mode]).squeeze())).int()
    l_per_ele = torch.clamp(l_per_ele, 1, l_per_ele.max().item())
    max_L = min(l_per_ele.max(), L).item()

    if torch.sum(l_per_ele == l_per_ele.max()) < 2:
        max_L = min(l_per_ele.max() - 1, L)
        l_per_ele = torch.clamp(l_per_ele, 1, max_L)
    pmf_l_tensor = torch.div((mode ** l_per_ele.view(-1)), 2, rounding_mode="floor")

    Nary_tensor = torch.zeros([pmf_l_tensor.size(0)] + [max_L]).int().to(device)

    pmfs_list = []
    xpmfs_list = []
    x2pmfs_list = []
    idx_ts_list = []
    pmf_center_list = [(mode ** (max_L - j)) // 2 for j in range(max_L)]
    for i in range(1, max_L + 1):
        pmf_length = 3 ** i
        pmf_center = pmf_length // 2
        samples = torch.abs(torch.arange(pmf_length, device=device).repeat((l_per_ele == i).sum(), 1) - pmf_center)
        upper = _standardized_cumulative((0.5 - samples) / scales_hat.view(-1, 1)[l_per_ele.view(-1) == i])
        lower = _standardized_cumulative((-0.5 - samples) / scales_hat.view(-1, 1)[l_per_ele.view(-1) == i])
        pmfs_ = upper - lower
        pmfs_ = (pmfs_ + 1e-10) / (pmfs_ + 1e-10).sum(dim=-1).unsqueeze(-1)
        pmfs_list.insert(0, pmfs_.clone())
        del upper, lower, samples
        torch.cuda.empty_cache()
        idx_tmp = torch.arange(mode ** i, device=device).repeat(pmfs_.size(0), 1)
        xpmfs_ = pmfs_ * idx_tmp
        xpmfs_list.insert(0, xpmfs_.clone())
        x2pmfs_ = pmfs_ * torch.pow(idx_tmp, 2)
        x2pmfs_list.insert(0, x2pmfs_.clone())
        idx_ts_list.insert(0, torch.ones_like(pmfs_list[0], device=device))
        del idx_tmp, pmfs_, xpmfs_, x2pmfs_
        torch.cuda.empty_cache()

    # reconstruction value per trit, and predicted reconstruction value.
    for i in range(max_L):
        if i < max_L - opt_pnum:
            decoder = RansDecoder()
            decoder.set_stream(open(y_strings[i], "rb").read())

            cond_pmfs = list(map(lambda p, idx: (p * idx).view(p.size(0), 3, -1).sum(-1) / (p * idx).view(p.size(0), 1, -1).sum(-1), pmfs_list[:i + 1], idx_ts_list[:i + 1]))
            tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, cond_pmfs))
            pmf_length = list(map(lambda tm: torch.zeros_like(tm).int().to(device) + mode, tail_mass))
            cond_cdf = torch.cat(list(map(lambda p, tm, l: _pmf_to_cdf(p, tm, l, mode), cond_pmfs, tail_mass, pmf_length)), dim=0).tolist()

            symbols_num = (l_per_ele.view(-1) >= max_L - i).sum().item()
            indexes_list = list(range(symbols_num))
            cdf_lengths = [mode + 2 for _ in range(symbols_num)]
            offsets = [-(mode // 2) for _ in range(symbols_num)]
            rv = decoder.decode_stream(
                indexes_list, cond_cdf, cdf_lengths, offsets
            )
            rv = (torch.Tensor(rv) - torch.Tensor(offsets)).int().to(device)
            tmp_idx = 0
            for j in range(i + 1):
                if j == 0:
                    tmp_idx += len(pmfs_list[j])
                    Nary_tensor[l_per_ele.view(-1) == max_L - j, i] = rv[:tmp_idx]
                elif j == i:
                    Nary_tensor[l_per_ele.view(-1) == max_L - j, i] = rv[tmp_idx:]
                else:
                    Nary_tensor[l_per_ele.view(-1) == max_L - j, i] = rv[tmp_idx:tmp_idx + len(pmfs_list[j])]
                    tmp_idx += len(pmfs_list[j])

            for j in range(i + 1):
                Nary_part = Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i:i + 1]
                tmp_ = torch.arange(mode ** (max_L - i), device=device).repeat(Nary_part.size(0), 1).int()
                idx_ts_list[j] *= (torch.div((tmp_ % (mode ** (max_L - i))), (mode ** (max_L - 1 - i)), rounding_mode="floor") == Nary_part)
                nz_idx = idx_ts_list[j].nonzero(as_tuple=True)
                pmfs_list[j] = pmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                xpmfs_list[j] = xpmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                x2pmfs_list[j] = x2pmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                idx_ts_list[j] = idx_ts_list[j][nz_idx].view(pmfs_list[j].size(0), -1)

            recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l, xpmfs_list, pmfs_list, pmf_center_list))
            y_hat = means_hat.clone().view(-1)
            for j in range(i + 1):
                y_hat[l_per_ele.view(-1) == max_L - j] += recon[j]
            y_hats.append(y_hat.view(means_hat.shape))
            torch.cuda.empty_cache()

        else:
            pmfs_list_l = pmfs_list[:i + 1]
            xpmfs_list_l = xpmfs_list[:i + 1]
            x2pmfs_list_l = x2pmfs_list[:i + 1]
            m_old = list(map(lambda xp, p: xp.sum(dim=-1) / p.sum(dim=-1), xpmfs_list_l, pmfs_list_l))
            D_old = list(map(lambda x2p, xp, p, m: (x2p.sum(-1) - 2 * m * xp.sum(-1) + (m ** 2) * p.sum(-1)) / p.sum(-1), x2pmfs_list_l, xpmfs_list_l, pmfs_list_l, m_old))

            pmfs_cond_list_l = list(map(lambda x: x.view(x.size(0), -1, mode ** (max_L - 1 - i)).sum(-1), pmfs_list_l))
            xpmfs_cond_list_l = list(map(lambda xp: xp.view(xp.size(0), -1, mode ** (max_L - 1 - i)).sum(-1), xpmfs_list_l))
            x2pmfs_cond_list_l = list(map(lambda x: x.view(x.size(0), -1, mode ** (max_L - 1 - i)).sum(-1), x2pmfs_list_l))

            m_new = list(map(lambda xp, p: xp / p, xpmfs_cond_list_l, pmfs_cond_list_l))
            D_new = list(map(lambda x2p, xp, p, m, fullp: ((x2p - 2 * m * xp + (m ** 2) * p) / fullp.sum(-1).view(-1, 1)).sum(-1),
                             x2pmfs_cond_list_l, xpmfs_cond_list_l, pmfs_cond_list_l, m_new, pmfs_list_l))
            delta_D = list(map(lambda old, new: (new - old).clamp_(max=0), D_old, D_new))
            pmf_cond_list_l_norm = list(map(lambda p: p / p.sum(-1).view(-1, 1), pmfs_cond_list_l))
            delta_R = list(map(lambda p: (-p * torch.log2(p)).sum(-1), pmf_cond_list_l_norm))
            delta_R = list(map(lambda h: h * (h >= 0), delta_R))
            optim_tensor = torch.cat(list(map(lambda D, R: -D / R, delta_D, delta_R))).clamp_(min=0)

            pmfs_norm = list(map(lambda p: p / p.sum(-1).view(-1, 1), pmfs_cond_list_l))

            tail_mass = list(map(lambda p: torch.zeros([len(p), 1]).to(device) + 1e-09, pmfs_norm))
            pmf_length = list(map(lambda tm: torch.zeros_like(tm).int().to(device) + mode, tail_mass))
            cond_cdf = torch.cat(list(map(lambda p, tm, l: _pmf_to_cdf(p, tm, l, mode), pmfs_norm, tail_mass, pmf_length)), dim=0)
            cond_cdf = cond_cdf[torch.argsort(optim_tensor, descending=True)].tolist()

            total_symbols = (l_per_ele.view(-1) >= max_L - i).sum().item()
            cdf_lengths = [mode + 2 for _ in range(total_symbols)]
            offsets = [-(mode // 2) for _ in range(total_symbols)]
            del tail_mass, pmf_length
            torch.cuda.empty_cache()

            pnum_part = _pnum_part(i, max_L)
            points_num = math.ceil(pnum_btw_trit * pnum_part)

            sl = total_symbols // points_num

            decoded_rvs = []
            for point in range(points_num):
                decoder = RansDecoder()
                for cp in y_strings:
                    if f"q{max_L - i - 1:02d}_{point + 1:03d}" in cp:
                        codepath = cp
                        break
                code = open(codepath, "rb").read()
                decoder.set_stream(code)
                if point == points_num - 1:
                    symbols_num_part = total_symbols - point * sl
                    indexes_list = list(range(symbols_num_part))
                    rv = decoder.decode_stream(
                        indexes_list,
                        cond_cdf[point * sl:],
                        cdf_lengths[point * sl:],
                        offsets[point * sl:]
                    )
                    rv = (torch.Tensor(rv) - torch.Tensor(offsets[point * sl:])).int().to(device)
                    decoded_rvs.append(rv.clone())
                    rv = torch.cat(decoded_rvs)
                    Nary_tensor_tmp = rv[torch.argsort(torch.argsort(optim_tensor, descending=True), descending=False)].int()
                    tmp_idx = 0
                    for j in range(i + 1):
                        if j == 0:
                            tmp_idx += len(pmfs_list[j])
                            Nary_tensor[l_per_ele.view(-1) == max_L - j, i] = Nary_tensor_tmp[:tmp_idx]
                        elif j == i:
                            Nary_tensor[l_per_ele.view(-1) == max_L - j, i] = Nary_tensor_tmp[tmp_idx:]
                        else:
                            Nary_tensor[l_per_ele.view(-1) == max_L - j, i] = Nary_tensor_tmp[tmp_idx:tmp_idx + len(pmfs_list[j])]
                            tmp_idx += len(pmfs_list[j])

                    for j in range(i + 1):
                        Nary_part = Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i:i + 1]
                        tmp_ = torch.arange(mode ** (max_L - i), device=device).repeat(Nary_part.size(0), 1).int()
                        idx_ts_list[j] *= (torch.div((tmp_ % (mode ** (max_L - i))), (mode ** (max_L - 1 - i)), rounding_mode="floor") == Nary_part)
                        nz_idx = idx_ts_list[j].nonzero(as_tuple=True)
                        pmfs_list[j] = pmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                        xpmfs_list[j] = xpmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                        x2pmfs_list[j] = x2pmfs_list[j][nz_idx].view(pmfs_list[j].size(0), -1)
                        idx_ts_list[j] = idx_ts_list[j][nz_idx].view(pmfs_list[j].size(0), -1)

                    recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l,
                                     xpmfs_list, pmfs_list, pmf_center_list))
                    y_hat = means_hat.clone().view(-1)
                    for j in range(i + 1):
                        y_hat[l_per_ele.view(-1) == max_L - j] += recon[j]
                    y_hats.append(y_hat.view(means_hat.shape))
                    break

                indexes_list = list(range(sl))
                rv = decoder.decode_stream(
                    indexes_list,
                    cond_cdf[point * sl:(point + 1) * sl],
                    cdf_lengths[point * sl:(point + 1) * sl],
                    offsets[point * sl:(point + 1) * sl]
                )
                rv = (torch.Tensor(rv) - torch.Tensor(offsets[point * sl:(point + 1) * sl])).int().to(device)
                decoded_rvs.append(rv.clone())

                pre_cat = torch.cat(decoded_rvs)
                post_cat = torch.zeros([total_symbols - (point + 1) * sl]).to(device) - 1
                rv = torch.cat([pre_cat, post_cat])

                Nary_tensor_tmp = rv[torch.argsort(torch.argsort(optim_tensor, descending=True), descending=False)].int()
                tmp_idx = 0
                for j in range(i + 1):
                    if j == 0:
                        tmp_idx += len(pmfs_list[j])
                        Nary_tensor[l_per_ele.view(-1) == max_L - j, i] = Nary_tensor_tmp[:tmp_idx]
                    elif j == i:
                        Nary_tensor[l_per_ele.view(-1) == max_L - j, i] = Nary_tensor_tmp[tmp_idx:]
                    else:
                        Nary_tensor[l_per_ele.view(-1) == max_L - j, i] =\
                            Nary_tensor_tmp[tmp_idx:tmp_idx + len(pmfs_list[j])]
                        tmp_idx += len(pmfs_list[j])

                for j in range(i + 1):
                    Nary_part = Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i][
                        Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i] != -1]
                    tmp_ = torch.arange(mode ** (max_L - i), device=device).repeat(Nary_part.size(0), 1).int()
                    idx_ts_list[j][Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i] != -1] *= (torch.div((tmp_ % (mode ** (max_L - i))), (mode ** (max_L - 1 - i)), rounding_mode="floor") == Nary_part.view(-1, 1))
                    pmfs_list[j][Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i] != -1] *=\
                        idx_ts_list[j][Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i] != -1]
                    xpmfs_list[j][Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i] != -1] *=\
                        idx_ts_list[j][Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i] != -1]
                    x2pmfs_list[j][Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i] != -1] *=\
                        idx_ts_list[j][Nary_tensor[l_per_ele.view(-1) == max_L - j][:, i] != -1]
                Nary_tensor[Nary_tensor < 0] = 0

                recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l, xpmfs_list, pmfs_list, pmf_center_list))
                y_hat = means_hat.clone().view(-1)
                for j in range(i + 1):
                    y_hat[l_per_ele.view(-1) == max_L - j] += recon[j]
                y_hats.append(y_hat.view(means_hat.shape))

    return y_hats

def _pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
    cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
    for i, p in enumerate(pmf):
        prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
        _cdf = _pmf_to_quantized_cdf(prob.tolist(), 16)
        _cdf = torch.IntTensor(_cdf)
        cdf[i, : _cdf.size(0)] = _cdf
    return cdf

def _standardized_cumulative(inputs):
    half = float(0.5)
    const = float(-(2 ** -0.5))
    # Using the complementary error function maximizes numerical precision.
    return half * torch.erfc(const * inputs)

def _pnum_part(i, max_L):
    if i == max_L - 0:
        pnum_part = 24 / 24
    elif i == max_L - 1:
        pnum_part = 24 / 24
    elif i == max_L - 2:
        pnum_part = 24 / 24
    elif i == max_L - 3:
        pnum_part = 16 / 24
    elif i == max_L - 4:
        pnum_part = 8 / 24
    elif i == max_L - 5:
        pnum_part = 8 / 24
    elif i == max_L - 6:
        pnum_part = 3 / 48
    else:
        pnum_part = 1 / 48
    return pnum_part