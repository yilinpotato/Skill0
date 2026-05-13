#!/usr/bin/env bash
set -euo pipefail

TARGET_CACHE="/data2/myl/home_configs/.cache/alfworld"
TARGET_CACHE_ROOT="/data2/myl/home_configs/.cache"
CACHE_ROOT="/home/myl/.cache"
LINK_CACHE="/home/myl/.cache/alfworld"

if [[ ! -d "$TARGET_CACHE" ]]; then
  echo "[setup_alfworld_cache] Target cache not found: $TARGET_CACHE" >&2
  exit 1
fi

if [[ -L "$CACHE_ROOT" && ! -e "$CACHE_ROOT" ]]; then
  stale_target="$(readlink "$CACHE_ROOT")"
  echo "[setup_alfworld_cache] Replacing broken cache root symlink: $CACHE_ROOT -> $stale_target"
  rm "$CACHE_ROOT"
fi

if [[ -L "$CACHE_ROOT" ]]; then
  current_root="$(readlink "$CACHE_ROOT")"
  if [[ "$current_root" != "$TARGET_CACHE_ROOT" ]]; then
    echo "[setup_alfworld_cache] Replacing cache root symlink: $CACHE_ROOT -> $current_root"
    rm "$CACHE_ROOT"
    ln -s "$TARGET_CACHE_ROOT" "$CACHE_ROOT"
  fi
elif [[ -e "$CACHE_ROOT" && ! -d "$CACHE_ROOT" ]]; then
  backup="${CACHE_ROOT}.bak.$(date +%Y%m%d_%H%M%S)"
  echo "[setup_alfworld_cache] Existing non-directory cache root found, moving to: $backup"
  mv "$CACHE_ROOT" "$backup"
  ln -s "$TARGET_CACHE_ROOT" "$CACHE_ROOT"
elif [[ ! -e "$CACHE_ROOT" ]]; then
  ln -s "$TARGET_CACHE_ROOT" "$CACHE_ROOT"
else
  mkdir -p "$CACHE_ROOT"
fi

if [[ -d "$LINK_CACHE" && "$(readlink -f "$LINK_CACHE")" == "$TARGET_CACHE" ]]; then
  echo "[setup_alfworld_cache] ALFWorld cache already available: $LINK_CACHE"
  exit 0
fi

if [[ -L "$LINK_CACHE" ]]; then
  current_target="$(readlink "$LINK_CACHE")"
  if [[ "$current_target" == "$TARGET_CACHE" ]]; then
    echo "[setup_alfworld_cache] Symlink already configured: $LINK_CACHE -> $TARGET_CACHE"
    exit 0
  fi
  echo "[setup_alfworld_cache] Replacing stale symlink: $LINK_CACHE -> $current_target"
  rm "$LINK_CACHE"
elif [[ -e "$LINK_CACHE" ]]; then
  backup="${LINK_CACHE}.bak.$(date +%Y%m%d_%H%M%S)"
  echo "[setup_alfworld_cache] Existing path found, moving to: $backup"
  mv "$LINK_CACHE" "$backup"
fi

ln -s "$TARGET_CACHE" "$LINK_CACHE"
echo "[setup_alfworld_cache] Created symlink: $LINK_CACHE -> $TARGET_CACHE"
