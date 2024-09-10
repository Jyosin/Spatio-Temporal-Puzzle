FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip git vim wget unrar
RUN pip install --upgrade pip
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install tensorboard torchvision scikit-image pytorch-lightning==2.3.3 matplotlib==3.9.1 einops decord==0.6.0 numpy==1.26.4
WORKDIR /work