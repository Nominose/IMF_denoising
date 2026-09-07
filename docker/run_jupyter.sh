#!/usr/bin/env bash
# Create or start the GPU container for this repo and open JupyterLab in it.
#
#   bash docker/run_jupyter.sh                # start (or create) the container, print the URL
#   bash docker/run_jupyter.sh --build        # build the image first (docker/Dockerfile)
#   bash docker/run_jupyter.sh --recreate     # throw the container away and create it again
#   bash docker/run_jupyter.sh --dry-run      # print the docker commands instead of running them
#   PORT=8888 BIND=0.0.0.0 bash docker/run_jupyter.sh
#
# Works from WSL (drives at /mnt/<x>) and from Git Bash on Windows (drives at /<x>). Every host drive is
# mounted read-write at /host/<x> inside the container, so data paths are /host/d/research/... and the
# checkout is importable as `IMF_denoising` (PYTHONPATH is set to its parent).
#
# Safety, in order of importance:
#   * an existing container is never deleted unless you pass --recreate. Packages installed with pip
#     inside a container live only in that container; deleting it loses them (that is how lpips once
#     lived only in pytorch_container). Put them in docker/requirements.txt and rebuild instead.
#   * JupyterLab binds to 127.0.0.1 by default. Binding 0.0.0.0 publishes every mounted drive, as root,
#     to anyone on the network who knows the token.
#   * the token is random, generated once and kept in docker/.jupyter_token (gitignored, chmod 600),
#     so WSL and Git Bash share it. Override with JUPYTER_TOKEN=... if you must.
#   * the default port is 8889 so this can coexist with the older pytorch_container on 8888.
set -euo pipefail

CONTAINER="${CONTAINER_NAME:-imf_denoising}"
IMAGE="${DOCKER_IMAGE:-imf_denoising:2.0}"
PORT="${PORT:-8889}"
BIND="${BIND:-127.0.0.1}"
SHM="${SHM_SIZE:-16g}"

DRY_RUN=0; RECREATE=0; BUILD=0
for a in "$@"; do
  case "$a" in
    --dry-run)  DRY_RUN=1 ;;
    --recreate) RECREATE=1 ;;
    --build)    BUILD=1 ;;
    -h|--help)  sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

# --- where are the host drives, and what does the container call them? -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"                 # .../IMF_denoising
TOKEN_FILE="$SCRIPT_DIR/.jupyter_token"
if grep -qi microsoft /proc/version 2>/dev/null && [ -d /mnt/c ]; then
  FLAVOUR=wsl;  DRIVE_PREFIX=/mnt
elif [ "$(uname -o 2>/dev/null)" = "Msys" ]; then
  FLAVOUR=msys; DRIVE_PREFIX=""
  export MSYS_NO_PATHCONV=1                          # keep docker's arguments out of MSYS path mangling
else
  FLAVOUR=linux; DRIVE_PREFIX=""
fi

# host path -> path inside the container (only drive-rooted paths are translated)
to_container_path() {
  local p="$1"
  case "$FLAVOUR" in
    wsl)  [[ "$p" =~ ^/mnt/([a-z])(/.*)?$ ]] && { echo "/host/${BASH_REMATCH[1]}${BASH_REMATCH[2]}"; return; } ;;
    msys) [[ "$p" =~ ^/([a-z])(/.*)?$ ]]     && { echo "/host/${BASH_REMATCH[1]}${BASH_REMATCH[2]}"; return; } ;;
  esac
  echo "$p"
}

MOUNTS=()
if [ "$FLAVOUR" = linux ]; then
  MOUNTS+=(-v "$(dirname "$REPO_ROOT"):/host/repo")
  REPO_PARENT_IN_CONTAINER=/host/repo
else
  for l in c d e f g h i j k l m n o p q r s t u v w x y z; do
    d="${DRIVE_PREFIX}/${l}"
    [ -d "$d" ] || continue
    if [ "$FLAVOUR" = msys ]; then
      MOUNTS+=(-v "$(printf '%s' "$l" | tr a-z A-Z):\\:/host/${l}")   # C:\  ->  /host/c
    else
      MOUNTS+=(-v "${d}:/host/${l}")
    fi
  done
  REPO_PARENT_IN_CONTAINER="$(to_container_path "$(dirname "$REPO_ROOT")")"
fi
WORKDIR_IN_CONTAINER="${REPO_PARENT_IN_CONTAINER}/$(basename "$REPO_ROOT")"

# --- token ---------------------------------------------------------------------------------------
if [ -z "${JUPYTER_TOKEN:-}" ]; then
  if [ ! -s "$TOKEN_FILE" ]; then
    ( umask 077
      python -c 'import secrets; print(secrets.token_hex(24))' 2>/dev/null \
        || python3 -c 'import secrets; print(secrets.token_hex(24))' 2>/dev/null \
        || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' ) > "$TOKEN_FILE"
  fi
  JUPYTER_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
fi

run() { if [ "$DRY_RUN" = 1 ]; then printf '%q ' "$@"; echo; else "$@"; fi; }

# --- image ---------------------------------------------------------------------------------------
if [ "$BUILD" = 1 ]; then
  run docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[!] image $IMAGE not found -- run with --build (see docker/Dockerfile for the mirror build-args)" >&2
  [ "$DRY_RUN" = 1 ] || exit 1
fi

# --- container -----------------------------------------------------------------------------------
exists() { docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; }

if exists && [ "$RECREATE" = 1 ]; then
  echo "[i] removing container $CONTAINER (--recreate)"
  run docker rm -f "$CONTAINER"
fi

if exists; then
  cur="$(docker inspect -f '{{.Config.Image}}' "$CONTAINER")"
  if [ "$cur" != "$IMAGE" ]; then
    echo "[!] $CONTAINER exists but runs image '$cur', not '$IMAGE'. Starting it as is;" >&2
    echo "    pass --recreate to rebuild it from '$IMAGE' (pip installs made inside it will be lost)." >&2
  fi
  echo "[i] starting existing container $CONTAINER"
  run docker start "$CONTAINER"
else
  echo "[i] creating $CONTAINER from $IMAGE ($FLAVOUR; drives -> /host/<x>; workdir $WORKDIR_IN_CONTAINER)"
  run docker run -d --gpus all \
    --name "$CONTAINER" \
    --restart unless-stopped \
    -p "${BIND}:${PORT}:8888" \
    --shm-size="$SHM" \
    "${MOUNTS[@]}" \
    -e "JUPYTER_TOKEN=${JUPYTER_TOKEN}" \
    -e "PYTHONPATH=${REPO_PARENT_IN_CONTAINER}" \
    --workdir "$WORKDIR_IN_CONTAINER" \
    "$IMAGE"
fi

echo
echo "JupyterLab: http://${BIND}:${PORT}/lab?token=${JUPYTER_TOKEN}"
echo "shell:      docker exec -it ${CONTAINER} bash"
echo "logs:       docker logs -f ${CONTAINER}"
