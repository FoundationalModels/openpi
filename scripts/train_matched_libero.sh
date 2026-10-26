#!/usr/bin/env bash
# Usage: bash scripts/train_matched_libero.sh /absolute/path/to/lerobot-v21 [train.py options]
set -euo pipefail
OPENPI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || "$1" == --help || "$1" == -h ]]; then
    echo "Usage: bash scripts/train_matched_libero.sh DATASET_V21 [train.py options]"
    echo "Environment: OPENPI_PYTHON, EXP_NAME, HF_LEROBOT_HOME, EXPECTED_SCENES, SKIP_NORM_STATS=1"
    echo "See examples/libero/README_matched.md for v3 preparation, smoke tests, and resume."
    if [[ $# -lt 1 ]]; then exit 1; else exit 0; fi
fi
DATASET="$1"
shift
# The parent project's isolated environment, or a standalone OpenPi checkout's .venv.
PYTHON="${OPENPI_PYTHON:-$OPENPI_ROOT/../.venv-openpi/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="${OPENPI_PYTHON:-$OPENPI_ROOT/.venv/bin/python}"; fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Set OPENPI_PYTHON to the OpenPi environment's Python executable." >&2
    exit 1
fi
cd -- "$OPENPI_ROOT"
"$PYTHON" scripts/prepare_matched_libero.py "$DATASET" --check-only \
    --expected-scenes "${EXPECTED_SCENES:-150}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$OPENPI_ROOT/../.cache/openpi-matched-lerobot}"
"$PYTHON" - "$DATASET" "$HF_LEROBOT_HOME" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).resolve(strict=True)
link = Path(sys.argv[2]).expanduser() / "local/libero_goal_matched_150_multi_targets"
link.parent.mkdir(parents=True, exist_ok=True)
if link.is_symlink() or link.exists():
    if link.resolve() != source:
        raise SystemExit(f"Dataset cache already points elsewhere: {link}. Choose another HF_LEROBOT_HOME.")
else:
    link.symlink_to(source, target_is_directory=True)
PY
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$OPENPI_ROOT/../.cache/openpi}"
export TOKENIZERS_PARALLELISM=false
# Recompute by default so statistics always correspond to this dataset.
if [[ "${SKIP_NORM_STATS:-0}" != 1 ]]; then
    "$PYTHON" scripts/compute_norm_stats.py --config-name pi05_matched_lora
fi
exec "$PYTHON" scripts/train.py pi05_matched_lora \
    --exp-name "${EXP_NAME:-matched_vlm_lora}" "$@"
