#!/usr/bin/env bash
#
# Repoint the EC2 instance's git remote after the move to Rawyal-Org.
#
# WHY THIS IS REQUIRED, not cosmetic: deploys are pull-based — a systemd
# timer on the instance polls `origin` and pulls (see the /version docstring
# in backend/main.py). The timer follows whatever `origin` is ON THE BOX. If
# that still points at the pre-migration repository, the instance keeps
# pulling from a repo nobody pushes to any more: no error, no alert, it just
# quietly serves stale code forever.
#
# Safe to re-run. Changes one git config value and fetches; it does not pull,
# reset, restart anything, or touch the working tree.
#
# Usage:
#   EC2_HOST=1.2.3.4 EC2_USER=ubuntu DEPLOY_PATH=/home/ubuntu/aurora \
#     bash scripts/repoint-instance-remote.sh
#
#   Add DRY_RUN=1 to print what it would do and change nothing.

set -euo pipefail

NEW_URL="https://github.com/Rawyal-Org/auroraPython.git"

: "${EC2_HOST:?set EC2_HOST to the instance hostname or IP}"
: "${EC2_USER:?set EC2_USER to the ssh user, e.g. ubuntu or ec2-user}"
: "${DEPLOY_PATH:?set DEPLOY_PATH to the git checkout the service runs from}"
DRY_RUN="${DRY_RUN:-}"

echo "host   : ${EC2_USER}@${EC2_HOST}"
echo "path   : ${DEPLOY_PATH}"
echo "new url: ${NEW_URL}"
echo

ssh "${EC2_USER}@${EC2_HOST}" DEPLOY_PATH="${DEPLOY_PATH}" NEW_URL="${NEW_URL}" \
    DRY_RUN="${DRY_RUN}" 'bash -seu' <<'REMOTE'
cd "$DEPLOY_PATH" || { echo "no such directory: $DEPLOY_PATH" >&2; exit 1; }
git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git checkout: $DEPLOY_PATH" >&2; exit 1; }

current="$(git remote get-url origin 2>/dev/null || echo '<none>')"
echo "current origin: $current"

if [ "$current" = "$NEW_URL" ]; then
  echo "already repointed — nothing to do"
else
  if [ -n "$DRY_RUN" ]; then
    echo "DRY_RUN: would run  git remote set-url origin $NEW_URL"
    # Still worth answering the question the dry run is really for: can this
    # box read the new repo at all? ls-remote asks without touching config,
    # so a credential problem surfaces now rather than after the switch.
    echo "DRY_RUN: checking reachability of $NEW_URL ..."
    if git ls-remote --heads "$NEW_URL" >/dev/null 2>&1; then
      echo "DRY_RUN: reachable — the real run would succeed"
    else
      echo "DRY_RUN: NOT reachable with this box's credentials." >&2
      echo "DRY_RUN: re-issue its deploy key / PAT for Rawyal-Org first." >&2
      exit 1
    fi
    exit 0
  fi
  git remote set-url origin "$NEW_URL"
  echo "set origin -> $(git remote get-url origin)"
fi

# Prove the new remote is actually reachable with the credentials this box
# holds. A private repo in a new org is exactly where a deploy key or PAT
# scoped to the old account stops working — better to fail here, loudly,
# than silently at 3am in the timer.
echo
echo "verifying fetch..."
if git fetch origin --quiet 2>/dev/null; then
  echo "fetch OK — origin/main is $(git rev-parse --short origin/main)"
  echo "checkout  is $(git rev-parse --short HEAD)"
else
  echo "FETCH FAILED — the box cannot read the new repo." >&2
  echo "Its credentials (deploy key / PAT / git credential helper) are still" >&2
  echo "scoped to the old account and need re-issuing for Rawyal-Org." >&2
  exit 1
fi
REMOTE

echo
echo "done. Next timer tick will pull from the new remote."
