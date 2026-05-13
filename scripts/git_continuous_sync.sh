#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

SYNC_REMOTE_NAME="${SYNC_REMOTE_NAME:-github}"
SYNC_REMOTE_URL="${SYNC_REMOTE_URL:-https://github.com/yilinpotato/SkillRL.git}"
SYNC_BRANCH="${SYNC_BRANCH:-main}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"
SYNC_MESSAGE_PREFIX="${SYNC_MESSAGE_PREFIX:-autosync}"
SYNC_PUSH="${SYNC_PUSH:-1}"

echo "[autosync] repo=$REPO_ROOT"
echo "[autosync] remote=$SYNC_REMOTE_NAME url=$SYNC_REMOTE_URL branch=$SYNC_BRANCH"
echo "[autosync] interval=${SYNC_INTERVAL_SECONDS}s"

while true; do
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "[autosync] $timestamp changes detected"
    TARGET_REMOTE_NAME="$SYNC_REMOTE_NAME" \
    TARGET_REMOTE_URL="$SYNC_REMOTE_URL" \
    TARGET_BRANCH="$SYNC_BRANCH" \
    COMMIT_MESSAGE="$SYNC_MESSAGE_PREFIX: $timestamp" \
    PUSH="$SYNC_PUSH" \
    bash "$REPO_ROOT/scripts/git_sync_to_github.sh" || true
  else
    echo "[autosync] $timestamp no changes"
  fi
  sleep "$SYNC_INTERVAL_SECONDS"
done
