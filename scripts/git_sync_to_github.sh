#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET_REMOTE_NAME="${TARGET_REMOTE_NAME:-github}"
TARGET_REMOTE_URL="${TARGET_REMOTE_URL:-https://github.com/yilinpotato/SkillRL.git}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-sync: update SkillRL changes}"
PUSH="${PUSH:-1}"

echo "[git-sync] repo: $REPO_ROOT"
echo "[git-sync] target remote: $TARGET_REMOTE_NAME -> $TARGET_REMOTE_URL"
echo "[git-sync] target branch: $TARGET_BRANCH"

if ! git remote get-url "$TARGET_REMOTE_NAME" >/dev/null 2>&1; then
  git remote add "$TARGET_REMOTE_NAME" "$TARGET_REMOTE_URL"
else
  git remote set-url "$TARGET_REMOTE_NAME" "$TARGET_REMOTE_URL"
fi

git status --short --branch

git add -A

if git diff --cached --quiet; then
  echo "[git-sync] no staged changes; nothing to commit"
else
  git commit -m "$COMMIT_MESSAGE"
fi

if [[ "$PUSH" == "1" ]]; then
  git push "$TARGET_REMOTE_NAME" "HEAD:$TARGET_BRANCH"
  echo "[git-sync] pushed to $TARGET_REMOTE_NAME/$TARGET_BRANCH"
else
  echo "[git-sync] PUSH=0, skipped push"
fi
