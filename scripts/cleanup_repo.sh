#!/usr/bin/env bash
# One-time cleanup: ~9GB duplicate patches + optional archives. Run from repo root:
#   bash scripts/cleanup_repo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== duplicate processed (saves ~9GB); keeps aconvvnet_patches_chen ==="
if [[ -d processed/aconvnet_patches ]]; then
  echo "Removing processed/aconvnet_patches ..."
  rm -rf processed/aconvnet_patches
else
  echo "already removed"
fi

echo "=== root zip archives (already under data/) ==="
shopt -s nullglob
for z in MSTAR-*.zip; do
  echo "rm $z"
  rm -f "$z"
done

echo "=== broken .venv if present ==="
[[ -d .venv ]] && rm -rf .venv

echo "=== optional: legacy processed (uncomment if sure) ==="
# rm -rf processed/soc_88 processed/aconvnet_128

du -sh processed data artifacts runs 2>/dev/null || true
echo "Done. See docs/SHARING.md"