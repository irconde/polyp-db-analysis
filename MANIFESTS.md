# Manifest Packs

## CSV schema
Manifest CSV files share a common set of columns.  Required fields are:

- `dataset`
- `split`
- `case_id`
- `frame_id`
- `frame_path`
- `label`
- `store_id`

Optional metadata fields include `morphology`, `source_video`, `frame_index`,
`width`, `height`, `variant`, and `severity`.

## Determinism policy
- Frames are ranked using HMAC‑SHA256 with a user‑supplied seed.
- Splitting happens at the case level.
- By default 33 frames are retained per case.

## Pack layout and versioning
A manifest pack is a directory containing CSVs for each split and a
`manifest.yaml` file with:

- `name` and `version`
- `generator` info (`repo`, `git_commit`, `command`, `seed`)
- `policy` (unit of split, frames per case, balance, selection)
- `roots` mapping `store_id` values to dataset roots
- `hashes` and `counts` for each split

## Overriding roots at runtime
`manifest.yaml` records default roots, but tools call `load_roots` to merge
sources in priority order:

1. JSON file passed via `--roots`
2. Roots in `manifest.yaml`
3. A fallback supplied with `--default-root`

This allows local overrides without editing the manifest.

## Build commands
Example commands to generate packs:

- **SUN full:** `polypdb sun build --root /path/to/sun --out sun_full --seed 0`
- **SUN subsets:** `polypdb sun subsets --pack sun_full --percents 25 50 --seeds 0 1 --out subsets`
- **SUN corruption:** `polypdb sun corrupt --pack sun_full --out sun_corrupt --variant corrupt`
- **PolypGen eval:** `polypdb polypgen build --root /path/to/polypgen --out polypgen_eval`
- **PolypGen few‑shot:** `polypdb polypgen fewshot --pack polypgen_eval --sizes 1 5 --seeds 0 1 --out fewshot`

## Validation
Run one of the following checks before using a pack:

- `polypdb validate-canonical <canonical_dir>`
- `python scripts/check_paths.py --csv train.csv --roots roots.json`

Validation ensures SHA256 hashes match, frame IDs are unique per split, labels are
0 or 1, morphology is one of `flat`, `polypoid`, or empty, and SUN case IDs do
not overlap across splits.
