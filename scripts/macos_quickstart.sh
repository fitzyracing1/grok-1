#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found in PATH."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install dm_haiku==0.0.12 numpy==1.26.4 sentencepiece==0.2.0 "jax==0.4.25"

python scripts/system_check.py

cat <<'EOF'

Setup complete.

Notes:
- This machine is prepared for local development workflows.
- Full Grok-1 local inference is usually not feasible on Apple Silicon due to memory limits.
- Continue with docs/PERFECT_GROK1_PLAN.md for optimization and scaling steps.

EOF