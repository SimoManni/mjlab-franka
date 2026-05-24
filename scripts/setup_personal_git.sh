#!/usr/bin/env bash
# Configure a personal git identity + SSH key for THIS repo only.
#
# Usage:
#   scripts/setup_personal_git.sh "Your Name" you@personal.example ~/.ssh/id_ed25519_personal
#
# This writes per-repo settings via `git config --local` and never touches
# your global ~/.gitconfig. Run from anywhere inside the repo.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <user.name> <user.email> [ssh-key-path]" >&2
  exit 1
fi

NAME="$1"
EMAIL="$2"
SSH_KEY="${3:-}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

git config --local user.name  "$NAME"
git config --local user.email "$EMAIL"

if [[ -n "$SSH_KEY" ]]; then
  if [[ ! -f "$SSH_KEY" ]]; then
    echo "warning: SSH key not found at $SSH_KEY" >&2
  fi
  # IdentitiesOnly avoids ssh-agent picking up the wrong key.
  git config --local core.sshCommand "ssh -i $SSH_KEY -o IdentitiesOnly=yes"
fi

echo "Repo-local git identity set:"
git config --local --get user.name
git config --local --get user.email
git config --local --get core.sshCommand || true
