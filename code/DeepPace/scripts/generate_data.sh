#!/usr/bin/env bash
# Generate the DeepPace-Sim dataset (Phase I). Plain Python — no uv, no torch.
#
# Usage:
#   scripts/generate_data.sh                    # full v3 dataset -> _data/v3 (~400 MB, 60 props)
#   scripts/generate_data.sh dev                # 5 props, bucketed grid only -> _data/dev
#   scripts/generate_data.sh full --validate    # extra args pass through to generate_dataset
#   SEED=1 OUT=_data/seed1 scripts/generate_data.sh
#
# Modes:
#   full   all properties, full 0..365 DTA grid + bucketed grid (default)
#   dev    --limit 5 --bucketed-only, for fast iteration
#
# Env overrides:
#   OUT     output directory (default: _data/v3 for full, _data/dev for dev)
#   SEED    RNG seed (default: SimConfig.seed = 20260909)
#   FORCE   set to 1 to regenerate even if the output already exists
#   PYTHON  interpreter to use (default: first working of python, python3, py)
#
# Requires Python >= 3.12 with numpy, pandas and pyarrow:
#   pip install "numpy>=2.0" "pandas>=2.2" "pyarrow>=17.0"
#
# Output (see src/deeppace_sim/writer.py): properties, stay_dates, on_books(_bucketed),
# ground_truth, promo_calendar, event_calendar .parquet, plus config.json with generator_hash.
# The output is gitignored (_*) and fully reproducible from seed + config.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root, so `-m scripts.generate_dataset` resolves

MODE="full"
if [[ $# -gt 0 && "$1" != -* ]]; then
    MODE="$1"
    shift
fi

case "$MODE" in
    full) DEFAULT_OUT="_data/v3";  ARGS=() ;;
    dev)  DEFAULT_OUT="_data/dev"; ARGS=(--limit 5 --bucketed-only) ;;
    *)    echo "unknown mode '$MODE' (expected: full | dev)" >&2; exit 2 ;;
esac

OUT="${OUT:-$DEFAULT_OUT}"
ARGS+=(--out "$OUT")
[[ -n "${SEED:-}" ]] && ARGS+=(--seed "$SEED")

if [[ -f "$OUT/config.json" && "${FORCE:-0}" != "1" ]]; then
    echo "$OUT already holds a dataset (config.json present); reusing it."
    echo "Set FORCE=1 to regenerate."
    exit 0
fi

# find a usable interpreter (runs each candidate, so the Windows Store python3 stub is skipped)
PY=""
for cand in ${PYTHON:-} python python3 py; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import sys" >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done
[[ -n "$PY" ]] || { echo "no working python found — install Python >= 3.10 or set PYTHON=..." >&2; exit 1; }

# single-line on purpose: multi-line -c args get truncated by Windows .bat shims (pyenv-win)
"$PY" -c 'import sys, importlib.util as u; sys.version_info >= (3, 10) or sys.exit("Python >= 3.10 required, found " + sys.version.split()[0]); m = [x for x in ("numpy", "pandas", "pyarrow") if u.find_spec(x) is None]; m and sys.exit("missing packages: " + ", ".join(m) + " -- pip install " + " ".join(m))'

# the package is not installed, so put src/ (deeppace_sim) and the repo root (scripts) on the path
# (separator comes from python itself: native Windows python wants ';', not ':')
SEP="$("$PY" -c 'import os; print(os.pathsep, end="")')"
export PYTHONPATH="src${SEP}.${PYTHONPATH:+${SEP}${PYTHONPATH}}"
"$PY" -m scripts.generate_dataset "${ARGS[@]}" "$@"
