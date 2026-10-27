#!/usr/bin/env bash
set -euo pipefail

# OBSOLETE: this script reproduces the pre-v3 flow (seeds 13/29/47, `sun perturb`).
# For the v3 packs use scripts/build_v3_packs.py (see README, "Reproducing the v3 manifest packs").
echo "scripts/build_all.sh is obsolete; use scripts/build_v3_packs.py" >&2
[ "${ALLOW_OBSOLETE:-0}" = "1" ] || exit 1

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

SUN_PERT="$PACKS_DIR/sun_test_perturbations"
python -m polypdb.cli --log-level "$LOG_LEVEL" sun perturb --pack "$SUN_FULL" --out "$SUN_PERT" --rng-master-seed 47

SUN_MORPH="$PACKS_DIR/sun_morphology"
python -m polypdb.cli --log-level "$LOG_LEVEL" sun morph --root "$SUN_ROOT" --out "$SUN_MORPH" --seed "$SUN_SEED"

SUN_SUBSETS="$PACKS_DIR/sun_subsets"
python -m polypdb.cli --log-level "$LOG_LEVEL" sun subsets --pack "$SUN_FULL" --percents 5 10 25 50 100 --seeds 13 29 47 --out "$SUN_SUBSETS"

POLYPGEN_TEST="$PACKS_DIR/polypgen_clean_test"
python -m polypdb.cli --log-level "$LOG_LEVEL" polypgen build --root "$POLYPGEN_ROOT" --out "$POLYPGEN_TEST" --track clean

POLYPGEN_TEST_EXT="$PACKS_DIR/polypgen_clean_test_extended"
python -m polypdb.cli --log-level "$LOG_LEVEL" polypgen build --root "$POLYPGEN_ROOT" --out "$POLYPGEN_TEST_EXT" --track clean --include-negative-sequences

POLYPGEN_FEWSHOT="$PACKS_DIR/polypgen_fewshot"
python -m polypdb.cli --log-level "$LOG_LEVEL" polypgen fewshot --pack "$POLYPGEN_TEST_EXT" --sizes 50 100 200 500 --seeds 13 29 47 --out "$POLYPGEN_FEWSHOT"

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

