#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKS_DIR="${PACKS_DIR:-$REPO_ROOT/packs}"

SUN_ROOT="${SUN_ROOT?Environment variable SUN_ROOT must be set}"
POLYPGEN_ROOT="${POLYPGEN_ROOT?Environment variable POLYPGEN_ROOT must be set}"
SUN_SEED="${SUN_SEED:-47}"
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"

mkdir -p "$PACKS_DIR"

SUN_FULL="$PACKS_DIR/sun_full"
python -m polypdb.cli --log-level "$LOG_LEVEL" sun build --root "$SUN_ROOT" --out "$SUN_FULL" --seed "$SUN_SEED"

SUN_MORPH="$PACKS_DIR/sun_morphology"
python -m polypdb.cli --log-level "$LOG_LEVEL" sun morph --root "$SUN_ROOT" --out "$SUN_MORPH" --seed "$SUN_SEED"

SUN_SUBSETS="$PACKS_DIR/sun_subsets"
python -m polypdb.cli --log-level "$LOG_LEVEL" sun subsets --pack "$SUN_FULL" --percents 5 10 25 50 100 --seeds 13 29 47 --out "$SUN_SUBSETS"

POLYPGEN_TEST="$PACKS_DIR/polypgen_clean_test"
python -m polypdb.cli --log-level "$LOG_LEVEL" polypgen build --root "$POLYPGEN_ROOT" --out "$POLYPGEN_TEST" --track clean

POLYPGEN_FEWSHOT="$PACKS_DIR/polypgen_fewshot"
python -m polypdb.cli --log-level "$LOG_LEVEL" polypgen fewshot --pack "$POLYPGEN_TEST" --sizes 50 100 200 500 --seeds 13 29 47 --out "$POLYPGEN_FEWSHOT"

ROBUST_SPEC="$REPO_ROOT/specs/robustness.yaml"
if [ -f "$ROBUST_SPEC" ]; then
    python -m polypdb.cli --log-level "$LOG_LEVEL" pipeline "$ROBUST_SPEC"
fi

find "$PACKS_DIR" -name manifest.yaml | sort | while read -r man; do
    dir="$(dirname "$man")"
    counts=$(python - "$man" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
print(data.get('counts', {}))
PY
)
    echo "$dir"
    echo "$counts"
done

