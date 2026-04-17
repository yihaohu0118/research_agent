#!/usr/bin/env bash
# One-shot: initialise git, create initial commit, push to the research_agent repo.
#
# Usage (run in a regular Terminal, not through Cursor's sandboxed shell):
#
#   cd ~/Desktop/AgentEvolver-main
#   bash scripts/push_to_github.sh
#
# You will be prompted once for your GitHub credentials. If you have a
# Personal Access Token, paste it as the "Password". SSH users should first
# edit REMOTE_URL below to the git@ form.
set -euo pipefail

REMOTE_URL="${REMOTE_URL:-https://github.com/yihaohu0118/research_agent.git}"
BRANCH="${BRANCH:-main}"
COMMIT_MSG=${COMMIT_MSG:-"Initial commit: AgentEvolver + TOCF + PACE + GCCE (400/400 BFCL split)"}

cd "$(dirname "$0")/.."

if [[ -d .git ]]; then
  echo "[push_to_github] .git already exists; skipping git init."
else
  git init -b "$BRANCH"
fi

# Local identity (only for this repo).
git config user.email "huyihao@users.noreply.github.com"
git config user.name  "huyihao"

# Remote (idempotent).
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

git add -A

if git diff --cached --quiet; then
  echo "[push_to_github] Nothing staged; aborting."
  exit 0
fi

git commit -m "$COMMIT_MSG"

echo "[push_to_github] Pushing to $REMOTE_URL ($BRANCH)..."
git push -u origin "$BRANCH"

echo "[push_to_github] Done. Visit: ${REMOTE_URL%.git}"
