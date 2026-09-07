# GPU environment for IMF_denoising: torch 2.0.1 + CUDA 11.7 (the same base the paper's results were
# produced with) plus the pinned Python deps in requirements.txt.
#
# Build (from the repo root):
#   docker build -t imf_denoising:2.0 -f docker/Dockerfile docker/
# Behind the GFW, point the registry, apt and pip at mirrors (and, if Docker Desktop is configured
# with a local proxy that is not running, blank the proxy variables so the build connects directly):
#   docker build -t imf_denoising:2.0 -f docker/Dockerfile \
#       --build-arg REGISTRY=docker.1ms.run \
#       --build-arg APT_MIRROR=http://mirrors.aliyun.com \
#       --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#       --build-arg http_proxy= --build-arg https_proxy= docker/
# Then start it with docker/run_jupyter.sh (mounts the host drives under /host/<drive>).
#
# Every step fails loudly. The previous Dockerfile ended its pip line with `|| true`, which turned a
# requirements file that cannot resolve on Python 3.10 into a "successful" image with nothing
# installed -- never do that again.
ARG REGISTRY=docker.io
FROM ${REGISTRY}/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1

# APT_MIRROR replaces archive/security.ubuntu.com (e.g. http://mirrors.aliyun.com); empty = upstream.
ARG APT_MIRROR=
# libgl1/libglib2.0-0: opencv-python needs them at import time even without a display
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s@http://archive.ubuntu.com@${APT_MIRROR}@g; s@http://security.ubuntu.com@${APT_MIRROR}@g" /etc/apt/sources.list; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs tzdata libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

# Optional: the CT projector used by simulation/ (noise insertion), pulled from GitHub. Off by default
# because training / inference / evaluation never import it and github.com is often unreachable here.
#   --build-arg WITH_CT_PROJECTOR=1
ARG WITH_CT_PROJECTOR=0
RUN if [ "$WITH_CT_PROJECTOR" = "1" ]; then \
        pip install cupy-cuda11x==12.2.0 git+https://github.com/zhennongchen/CTProjector.git ; \
    fi

# /host is JupyterLab's root and where run_jupyter.sh mounts the drives; it has to exist even when the
# image is started without mounts, or the server refuses to start ("No such directory: '/host'").
RUN mkdir -p /host /workspace

# The repo is imported as the package `IMF_denoising`, so its PARENT directory has to be on PYTHONPATH.
# run_jupyter.sh passes the right value for the mounted checkout; this default matches the usual
# mount of C: at /host/c.
ENV PYTHONPATH=/host/c/Users/ROG/Documents/GitHub

WORKDIR /workspace
EXPOSE 8888
# JUPYTER_TOKEN is injected by run_jupyter.sh; without it JupyterLab generates a random token and
# prints it in the container logs (docker logs <name>).
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.root_dir=/host"]
