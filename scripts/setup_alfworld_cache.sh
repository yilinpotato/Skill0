#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-$REPO_ROOT/.cache}"
ALFWORLD_CACHE="${ALFWORLD_DATA:-$PROJECT_CACHE_ROOT/alfworld}"

mkdir -p "$PROJECT_CACHE_ROOT"
mkdir -p "$ALFWORLD_CACHE"

export ALFWORLD_DATA="$ALFWORLD_CACHE"

echo "[setup_alfworld_cache] ALFWORLD_DATA=$ALFWORLD_DATA"

if [[ ! -d "$ALFWORLD_CACHE/json_2.1.1" || ! -d "$ALFWORLD_CACHE/logic" ]]; then
  echo "[setup_alfworld_cache] ALFWorld data is not complete under $ALFWORLD_CACHE" >&2
  echo "[setup_alfworld_cache] Run: ALFWORLD_DATA=$ALFWORLD_CACHE alfworld-download -f" >&2
fi
