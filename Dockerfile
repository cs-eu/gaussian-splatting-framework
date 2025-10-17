FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# set default shell to bash
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt update && apt upgrade -y && apt install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    curl \
    git \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1

# Install dependencies for original gs implementation
RUN apt-get install ffmpeg libsm6 libxext6 pip -y --fix-missing
RUN pip install mkl==2024.0

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN mkdir -p $CONDA_DIR && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $CONDA_DIR/miniconda.sh && \
    bash $CONDA_DIR/miniconda.sh -b -u -p $CONDA_DIR && \
    rm $CONDA_DIR/miniconda.sh && \
    source $CONDA_DIR/bin/activate && \
    conda init --all