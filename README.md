
## Multiple Temporal Scale Network for Remote PPG and Heart Rate Estimation from Facial Video
#### Dao Q. Le, Wen-Nung Lie*, Po-Han Huang, Guan-Hao Fu, Quynh Nguyen Quang Nhu

#### Link: 

#### Abstract

  This paper presents a deep-learning-based multi-temporal-scale network for remote photoplethysmogram (rPPG) signal estimation from a facial video sequence, which is subsequently used to derive the physiological parameters, e.g., the heart rate. Our network consists of three 2D-convolutional streams, where one spatial stream is designed for the extraction of spatial attention mask and two temporal streams are Responsible for the capturing of temporal dependency modeling from the same video input with different frame rates. The attention masks from The spatial stream provides the required fusion information to correctly enhance the temporal streams’ feature maps. By leveraging rich temporal dependency information, our network can adeptly comprehend the video’s spatio-temporal structure for accurate rPPG signal estimation. Experimental results on four public datasets (PURE, MMSE-HR, UBFC-rPPG, and MAHNOB-HCI) show that the proposed method, compared We achieved competitive or even superior performances compared with state-of-the-art methods, showcasing our network’s learning capability.
  
_Index Terms_—remote PPG, facial video sequence, heart rate estimation, multiple temporal scale network, deep-learning.

## Citation 

``` bash
@article{lie2025two,
  title={ Multiple Temporal Scale Network for Remote PPG and Heart Rate Estimation from Facial Video},
  author={Dao Q. Le, Wen-Nung Lie, Po-Han Huang, Guan-Hao Fu, Quynh Nguyen Quang Nhu},
  journal={Biomedical Signal Processing and Control XX},
  year={2025},
  publisher={XX}
}
```

## Installation
1. This project is developed using >= python 3.10 on Ubuntu 22.04.3! NVIDIA GPUs are needed. We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment.

```shell
  # 1. Create a conda virtual environment.
  conda create -n MS_rppg python=3.10 -y
  conda activate MS_rppg
  
  # 2. Install PyTorch >= v1.6.0 following [official instruction](https://pytorch.org/). Please adapt the cuda version to yours.
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  
  # 3. Pull our code.
  git clone https://github.com/GuanHao-Fu/3stream_rppg.git
  cd MS_rppg
  
  # 4. Install other packages. This project doesn't have any special or difficult-to-install dependencies.
  pip install -r requirements.txt 
```

## Data preprocessing & build k-fold
2. Before training model, it is necessary to prepare the traing model which datatype it want. Executing **dataset_preprocess.py** transfer the video data to **h5py** data. After bulid h5py file, excuting **built_k_fold,py** generate k-fold trainging and testing dataset 
```shell
  # dataset_preprocess
  cd utils
  python dataset_preprocess.py --root_dir [original_DATASET_PATH] --save_dir [preprocessed_DATASET_PATH]

  # built_k_fold
  python built_k_fold.py --root_dir [preprocessed_DATASET_PATH] --save_dir [saved_ DATASET_PATH]
```
## Training 

```shell

  python train.py --dataset_dir [kfold_DATASET_PATH] --checkpoint_dir [CHECKPOINT_PATH]

```
## Inference 

```shell

  python test.py --dataset_dir [kfold_DATASET_PATH] --checkpoint_dir [CHECKPOINT_PATH]

```
The default video frame rate is **30Hz**. Please change the frame rate when you using the dataset recorded by other frame rate. 

## Note
If you want to change the input image size for training and testing, you needed to check **train.py** and **test.py** file import different package. We write two types code for image size =36 and 72.

if the input image size = **36**, uncomment following two lines code
```shell
  from models36 import is_model_support, get_model
  from dataset.dataset_loader36 import dataset_loader
```
if the input image size = **72**, uncomment following two lines code
```shell

  from models72 import is_model_support, get_model
  from dataset.dataset_loader72 import dataset_loader
```
