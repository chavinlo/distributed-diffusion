FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y && apt install -qq  -y curl wget screen atop htop psmisc git
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt update -y && apt install -qq -y python3-dev python3-pip cuda-11-7
RUN --mount=type=cache,target=~/.cache pip install ninja networkx torch torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu117
RUN wget https://download.pytorch.org/whl/nightly/torchtriton-2.0.0%2Bf16138d447-cp310-cp310-linux_x86_64.whl
RUN --mount=type=cache,target=~/.cache pip install torchtriton-2.0.0+f16138d447-cp310-cp310-linux_x86_64.whl
WORKDIR /
RUN git clone https://github.com/facebookresearch/xformers.git
WORKDIR xformers
RUN git submodule update --init --recursive
WORKDIR /
ENV CUDA_HOME /usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST 7.0 8.0
RUN --mount=type=cache,target=~/.cache pip install -e xformers
COPY requirements.txt .
RUN --mount=type=cache,target=~/.cache pip install -r requirements.txt
ADD . /distributed-training
WORKDIR /distributed-training
ENTRYPOINT ["python3", "server.py"]
EXPOSE 5080
