FROM nvidia/cuda:9.0-base

RUN apt update && apt install -y wget unzip curl bzip2 git
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
RUN conda install pytorch torchvision -c pytorch
RUN conda install visdom dominate -c conda-forge
 
RUN mkdir /workspace/ && cd /workspace/ && mkdir pytorch-CycleGAN-and-pix2pix
VOLUME /workspace/pytorch-CycleGAN-and-pix2pix

WORKDIR /workspace
