FROM nvcr.io/nvidia/pytorch:22.11-py3
# Conda 
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y && apt install -qq  -y curl # ffmpeg libsm6 libxext6
RUN curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor > conda.gpg 
RUN install -o root -g root -m 644 conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg 
RUN gpg --keyring /usr/share/keyrings/conda-archive-keyring.gpg --no-default-keyring --fingerprint 34161F5BF5EB1D4BFBBB8F0A8AEB4F8B29D82806
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list
RUN apt update && apt install conda 
ADD ./environment.yml /tmp/env.yml
ENV PATH /opt/conda/bin:$PATH
RUN --mount=type=cache,target=/opt/conda/pkgs conda env create -f /tmp/env.yml
# Make RUN commands use the new environment:
COPY requirements.txt .
SHELL ["conda", "run", "-n", "venv", "/bin/bash", "-c"]
RUN --mount=type=cache,target=~/.cache pip install -r requirements.txt
ADD . /distributed-training
WORKDIR /distributed-training
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "venv", "python", "server.py"]
EXPOSE 5080
