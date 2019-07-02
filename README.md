<img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# HLCV 2019 - Project Before2After

This is a fork of the original CycleGAN and Pix2Pix project by [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt) contributing to Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](http://people.csail.mit.edu/junyanz/projects/pix2pix/pix2pix.bib)

The code was written by [Sven Stauden](https://github.com/sstauden) and [Yogesh](https://github.com/taesung) and allows to setup the before2after dateset provided by us and train and test a suitable Pix2Pix model for generating images.

**CycleGAN: [Project](https://junyanz.github.io/CycleGAN/) |  [Paper](https://arxiv.org/pdf/1703.10593.pdf) |  [Torch](https://github.com/junyanz/CycleGAN)**
<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>


**Pix2pix:  [Project](https://phillipi.github.io/pix2pix/) |  [Paper](https://arxiv.org/pdf/1611.07004.pdf) |  [Torch](https://github.com/phillipi/pix2pix)**

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="800px"/>


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org) 0.4+ with the following dependencies:
  - torchvision
  - [visdom](https://github.com/facebookresearch/visdom) 
  - [dominate](https://github.com/Knio/dominate)

- Alternatively, you can setup the project using Docker. For this, just call:
```bash
docker build .
```

## Before2After Dataset
The Before2After dataset consists of about 1500 image pairs of scenes in the past and in the later time state. To setup the dataset, please follow these steps:
- open [load_dataset.py](scripts/before2after/load_dataset.py) and change the target directories, where all images should be downloaded to and where the img url file is located on your machine
- run the script with `python scripts/load_dataset.py`. This wil automatically rename the image files correctly and produce a train, validation and test set of images.
- After all images have been downloaded, apply preprocessing. For this, open [preprocessing.py](scripts/before2after/preprocessing.py) and adapt the path to your local settings of the base and target sources.
- Then call `python scripts/preprocessing.py` which will either create a new dataset with applied operations or replace the original data (depending how you specify it).

After all, you have the folders A and B with three subfolders (train, test, val) each. This structure can be used to create a dataset format for Pix2Pix calling

```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/target_folder
```

### training
- Training a model from console (takes very long)
```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./path/to/dataset --name before2after --display_id 0 --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
```
- Training from DBS (make sure that [train_pix2pix_before2after.sh](scripts/train_pix2pix_before2after.sh) has the correct configurations set).
```bash
qsub train_pix2pix_before2after.sh
```

- Training with Docker (after configuring the script file and creating the containiner, leave the container and start the training deamon with)
```bash
nvidia-docker exec -d container_id sh -c "cd pytorch-CycleGAN-and-pix2pix; ./scripts/train_pix2pix.sh"
```

### testing / generating
similar to training just using the test script





