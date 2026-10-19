#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_name="${HARNESSADAPT_TRAIN_IMAGE:-harnessadapt-training:20261019}"
target="${1:-runtime}"
args=()
case "$target" in
  runtime) ;;
  teachers)
    : "${CODEX_VERSION:?Set the explicit Codex CLI package version}"
    : "${OPENCODE_VERSION:?Set the explicit OpenCode CLI package version}"
    args+=(--build-arg "CODEX_VERSION=${CODEX_VERSION}" --build-arg "OPENCODE_VERSION=${OPENCODE_VERSION}")
    ;;
  *) echo "Usage: bash integrations/training/build.sh [runtime|teachers]" >&2; exit 2 ;;
esac
docker build --target "$target" --tag "$image_name" "${args[@]}" "$source_dir"
docker image inspect --format '{{.Id}}' "$image_name"
