FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

USER root

WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 \
    cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
    libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf \
    wget unzip \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/arpitg1304/forge.git \
 && cd forge \
 && pip3 install -e ".[all]" 

RUN wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/download/v4.40.5/yq_linux_amd64 \
 && chmod +x /usr/local/bin/yq

RUN git clone https://github.com/real-stanford/diffusion_policy.git \
 && cd diffusion_policy \
 && yq '.dependencies[] | select(has("pip")) | .pip[]' conda_environment.yaml \
    > pip_requirements.txt \
 && yq 'del(.dependencies[] | select(has("pip")))' conda_environment.yaml \
    > conda_environment_nopip.yaml \
 && mamba env create -y -f conda_environment_nopip.yaml

RUN conda init

RUN conda run -n robodiff python -m pip install "mujoco==2.3.7" --only-binary=:all: \
 && conda run -n robodiff python -m pip install -r diffusion_policy/pip_requirements.txt \
 && conda run -n robodiff python -m pip install -U pip wheel \ 
 && conda run -n robodiff python -m pip install --force-reinstall --no-cache-dir "setuptools==69.5.1" \
 && conda run -n robodiff python -m pip install -U wandb

RUN pip3 install --no-cache-dir -U "lerobot[pusht]"

RUN mkdir -p /workspace
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]
