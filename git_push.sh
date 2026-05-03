#!/usr/bin/env bash
# Usage: ./git_push.sh "your commit message"
# Stages all changes (including local deletions), commits, and pushes.
# Files deleted locally are automatically removed from the remote branch on push.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Error: please provide a commit message."
    echo "Usage: $0 \"your commit message\""
    exit 1
fi

COMMIT_MSG="$1"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
REMOTE="origin"

# Stage everything: new files, modifications, and deletions
git add -A

# Check if there is anything to commit
if git diff --cached --quiet; then
    echo "Nothing to commit on branch '$BRANCH'."
    exit 0
fi

# Show what will be committed
echo "=== Changes to be committed ==="
git diff --cached --stat
echo ""

# Highlight any deletions explicitly
DELETED="$(git diff --cached --name-only --diff-filter=D)"
if [ -n "$DELETED" ]; then
    echo "=== Files deleted locally (will be removed from remote) ==="
    echo "$DELETED"
    echo ""
fi

git commit -m "$COMMIT_MSG"

# Push; set upstream automatically if this branch has no remote tracking yet
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" &>/dev/null; then
    git push "$REMOTE" "$BRANCH"
else
    echo "No upstream set for '$BRANCH'; pushing and setting upstream..."
    git push -u "$REMOTE" "$BRANCH"
fi

echo ""
echo "Done. Branch '$BRANCH' pushed to $REMOTE."
