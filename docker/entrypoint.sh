#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/opt"
DST_ROOT="/workspace"

copy_if_missing() {
  local name="$1"
  local src="${SRC_ROOT}/${name}"
  local dst="${DST_ROOT}/${name}"

  if [[ ! -d "${dst}" ]]; then
    echo "[entrypoint] Initializing ${name} into ${dst}"
    # Use cp -a to preserve symlinks/permissions/timestamps
    cp -a "${src}" "${dst}"
  else
    echo "[entrypoint] ${name} already exists at ${dst} (skip)"
  fi
}

copy_if_missing "forge"
copy_if_missing "diffusion_policy"

exec "$@"
