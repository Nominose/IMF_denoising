# Docker environment

The GPU environment every script in this repo expects: torch 2.0.1 + CUDA 11.7, Python 3.10, the
pinned deps in `requirements.txt`, host drives mounted at `/host/<drive>` (so data is
`/host/d/research/...` and the checkout is importable as the package `IMF_denoising`).

```bash
# once: build the image (~10 GB base pulled from Docker Hub, ~2 GB of layers on top)
bash docker/run_jupyter.sh --build
# every time: start (or create) the container and print the JupyterLab URL
bash docker/run_jupyter.sh
# a shell inside it, already in the repo with PYTHONPATH set
docker exec -it imf_denoising bash
```

Behind the GFW build with mirrors instead:

```bash
docker build -t imf_denoising:2.0 -f docker/Dockerfile \
    --build-arg REGISTRY=docker.1ms.run \
    --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple docker/
```

`run_jupyter.sh` works from WSL and from Git Bash. Knobs are environment variables:
`PORT` (default 8889), `BIND` (default `127.0.0.1`), `CONTAINER_NAME`, `DOCKER_IMAGE`, `SHM_SIZE`,
`JUPYTER_TOKEN`. `--dry-run` prints the docker commands; `--recreate` deletes and recreates the
container (the only way it ever deletes anything).

## Rules that come from getting this wrong before

- **Add packages to `requirements.txt` and rebuild; never only `pip install` inside the container.**
  A container-side install lives in that container's writable layer and disappears the moment the
  container is recreated. `lpips`, which every eval script imports, once existed only that way.
- **Keep JupyterLab on 127.0.0.1.** It runs as root with every drive mounted read-write; on
  `0.0.0.0` anyone on the campus network with the token gets all of it. The old runner used
  `0.0.0.0:8888` and the token `mypw`.
- **No `|| true` after `pip install`.** The previous Dockerfile had one; its requirements could not
  resolve on Python 3.10 (`cupy-cuda110==8.3.0`, `h5py==2.10.0`, `python-gdcm==3.0.10` have no
  wheels for it), so the build "succeeded" with nothing installed.
- **Keep these files in the repo.** The files that built the original `pytorch_container` lived in a
  Desktop folder that was later emptied; the container could not be reproduced from anything on disk.

## Relation to the older `pytorch_container`

That container (image `pytorch_container:2.0`, created 2025-10-07) is the same base image with a
different runner. It keeps working and `run_jupyter.sh` never touches it: the new container has its
own name (`imf_denoising`) and port (8889). Stop the old one whenever you like with
`docker stop pytorch_container` (it auto-starts with Docker Desktop otherwise).

## What is not in the image

The CT projector used by `simulation/` (noise insertion) needs GitHub access and cupy; build with
`--build-arg WITH_CT_PROJECTOR=1` when you need it. Training, inference and evaluation never import it.
