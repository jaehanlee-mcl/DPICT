## [CVPR 2022] DPICT: Deep Progressive Image Compression Using Trit-Planes

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
