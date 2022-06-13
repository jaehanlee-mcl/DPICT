## [CVPR 2022] DPICT: Deep Progressive Image Compression Using Trit-Planes
Accepted to CVPR 2022 as Oral presentation

Paper link: [CVPR](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_DPICT_Deep_Progressive_Image_Compression_Using_Trit-Planes_CVPR_2022_paper.pdf), [arXiv](https://arxiv.org/pdf/2112.06334.pdf)

If you use our code or results, please cite:
```
@InProceedings{Lee_2022_CVPR,
    author    = {Lee, Jae-Han and Jeon, Seungmin and Choi, Kwang Pyo and Park, Youngo and Kim, Chang-Su},
    title     = {DPICT: Deep Progressive Image Compression Using Trit-Planes},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {16113-16122}
}
```

------
### 1. Preparation
1. Download a [DPICT-main model](https://drive.google.com/file/d/1UALEBfnbURR_lGzGFCwiXK5_L5wuVO7h/view?usp=sharing) parameters and place them in 'checkpoint\DPICT-Main\'
2. Download [DPICT-post model 1](https://drive.google.com/file/d/1BSQbT32Al18EPPifvdMQQJDfGnZby8zs/view?usp=sharing) and [DPICT-post model 2](https://drive.google.com/file/d/1_Xc8lhxq96rOa1z_m0EFohxz0lt5wdEf/view?usp=sharing) parameters and place them in 'checkpoint\DPICT-Post\'

------
### 2. Training of DPICT-Main
1. By executing 'train_main.py', the main network of DPICT is trained.
2. The training progress is saved in the log directory.

------
### 3. Training of DPICT-Post
1. When you run 'make_post_data.py', data for training DPICT's post networks are created. The path to the dataset and the path to the DPICT main network parameter file should be set appropriately.
2. By executing 'train_post.py', two post networks of DPICT are trained.
3. The training progress is saved in the log directory.

------
### 4. Compression & evaluation
1. By executing 'evel.py', compression using the given DPICT-Main and DPICT-Post networks and evaluation of the results can be performed.
