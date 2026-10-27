# Polyp DB Analysis

Toolkit for standardising colonoscopy polyp datasets.  The project includes
dataset converters for multiple sources, a pipeline for merging and preparing
data for object detection or image classification, and utilities for cleaning
and exporting images.

Key features include:

- Convert raw datasets into a canonical directory layout with unified
  metadata files.
- Merge datasets and generate training/validation/test splits for object
  detection or classification experiments.
- Prepare Ultralytics/YOLO datasets and export images for conventional
  classification workflows.
- Preprocess frames by masking viewports, removing overlays and stripping
  padding.
- Count class samples, subset datasets and other helper CLI commands.

## Installation

An optional Conda environment specification is provided at
`environment.yml` for a reproducible setup:

```bash
conda env create -f environment.yml
conda activate polyp-db-processing
```

Install the package and its dependencies.  The OCR utilities rely on
`PaddleOCR`, which in turn requires the `paddlepaddle` runtime.  Install
`paddlepaddle` first, then the package itself:

```bash
pip install "paddlepaddle<3"
pip install -e .
```

This exposes the `polypdb` command line interface.

## Reproducing the v3 manifest packs

The v3 partitions used by [SSL4POLYP](https://github.com/irconde/SSL4POLYP)
(`data_packs/v3`: 5 SUN morphology partitions, `sun_subsets` and PolypGen
few-shot packs, 221 CSV files in total) are produced by
[`scripts/build_v3_packs.py`](scripts/build_v3_packs.py). The script only calls
the public `polypdb` CLI (`sun morph`, `sun subsets-canonical`, `polypgen build`,
`polypgen fewshot`), so the commands recorded in every `manifest.yaml` are the
ones that were executed.

1. **Environment.** Use pandas 2.x (`pandas>=2.1,<3`). With pandas 3 the
   string representation of missing values changes and the hash of
   `polypgen_clean_test_extended` differs. The reference packs were generated
   with Python 3.14, pandas 2.3.3 and numpy 2.3.5 (also recorded in each
   manifest under `generator.environment`).
2. **Canonical datasets.** The raw SUN and PolypGen data are not redistributed.
   Obtain them from their original sources, edit the placeholders
   (`<RAW_ROOT>`, `<DATA_ROOT>`) in `config/sun_config.yaml` and
   `config/polypgen_config.yaml` and run `polypdb pipeline <config>` to
   produce the canonical `frames.parquet` / `cases.parquet`. Each manifest stores
   the SHA-256 of those parquet files (`provenance`), so you can check that your
   canonical data matches the one used for the published packs.
3. **Build.**

   ```bash
   python scripts/build_v3_packs.py \
       --sun-root /path/to/canonical/sun \
       --polypgen-root /path/to/canonical/polypgen \
       --out /path/to/v3_out
   ```

   Partition seeds are 47 (k=1) and the first four candidates in 101..120 that
   pass validation (101-104); subsets use seed 13 and few-shot packs use seed
   `1000*k + s` with `s` in 13, 29, 47.
4. **Verify.** Compare `v3_out/index.yaml` with
   `SSL4POLYP/data_packs/v3/index.yaml`; the `csv_sha256` entries must be
   identical for all 221 CSV files (the `manifest.yaml` files differ in paths
   and in the recorded tool commit, the CSV hashes do not).

`scripts/build_all.sh` is obsolete (pre-v3 flow) and is kept only for reference.

## CLI usage

Each command expects a YAML configuration file. Example configs are provided in
`merge_config.example.yaml` and `ultralytics_config.example.yaml`.

The CLI also accepts a global ``--log-level`` option to control verbosity. Place
it before the subcommand:

```bash
polypdb --log-level DEBUG pipeline pipeline_config.yaml
```

By default the log level is ``INFO``.

For standalone scripts or notebooks, enable debug logging with:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Merge datasets

Merge multiple dataset folders into a unified dataset described by a metadata
spreadsheet. The configuration requires:

```yaml
base_path: /path/to/datasets
num_elements: null      # optional limit per dataset
plot_images: false      # draw bounding boxes for inspection
```

Run the merge step with:

```bash
polypdb merge merge_config.yaml
```

During the merge step each lesion receives a global identifier in the
``<DATASET>_<LESION_ID>`` format (e.g. ``SUN_0034`` or ``PICCOLO_00017``).
All associated frames are then renamed deterministically using
``<global_lesion_id>.<frame_idx>.<ext>`` where ``frame_idx`` starts at
``000001`` for every lesion and ``ext`` keeps the original file extension.
This ensures collision‑free filenames and preserves the link to the source
dataset.

Negative frames are supported alongside positives. Configure their locations
via the ``negative_sources`` mapping in ``merge_config.yaml``:

```yaml
negative_sources:
  sun: sun/sundatabase_negative
```

Each directory may contain subfolders named after a deterministic group such as
the source video or acquisition date, **or** images can live directly inside the
directory. In the latter case the directory name itself acts as the group
identifier. During merging every image is copied to ``canonical/images`` and
renamed using the synthetic lesion key ``NEG_<DATASET>_<GROUP>`` resulting in
filenames like ``NEG_SUN_clipA.000123.jpg`` or ``NEG_POLAR_negatives.000045.png``.
Negative handling and has_polyp

- has_polyp: Frames from spreadsheets are marked positive if they contain at
  least one bounding box; otherwise they are marked negative in
  ``frames.parquet`` (``has_polyp = False``).
- External negatives: Images supplied via ``negative_sources`` are de‑duplicated
  and ingested as negatives as well (``has_polyp = False``) under synthetic
  case keys ``NEG_<DATASET>_<GROUP>``.
- SUN cache: when a SUN negatives cache file ``sun_neg_hash_cache.csv`` exists
  under the negative source, it is reused to skip re‑hashing on subsequent runs.
- Cases: Datasets without case/lesion ids may result in frames with
  ``global_case_id = NULL``. Validation and statistics skip case foreign‑key
  checks for such frames, and ``cases.parquet`` may be omitted entirely when no
  per‑case data is available.

PolypGen specifics

- Multiple boxes per frame are preserved (semicolon‑separated in ``BOUNDING BOX``).
- Bounding box files are looked up as ``<image>.txt`` and, if missing, as
  ``<image>_mask.txt``.

### Prepare Ultralytics YOLO dataset

Convert a merged dataset into a YOLO‐style directory structure. The configuration
includes dataset location, class type and split options:

```yaml
src_path: /path/to/merged-dataset
classification_type: BINARY
splitting_options:
  train_ratio: 0.7
  validation_ratio: 0.2
  test_ratio: 0.1
  split_type: STRATIFIED
  split_level: POLYP ID

  # or specify absolute counts (frame-level only)
  # train_count: 500
  # validation_count: 0
  # split_level: FRAME
negative_sampling:
  ratio: 0.5
  distribution: SPLIT-RATIO
```

Execute the preparation:

```bash
polypdb prepare ultralytics_config.yaml
```

### Prepare classification dataset

Create a HyperKvasir-style dataset for classification using the same Ultralytics
configuration file:

```bash
polypdb prepare-classification ultralytics_config.yaml
```

### Count dataset samples

Count the number of samples per class in a YOLO dataset directory:

```bash
polypdb count path/to/dataset
```

### Dataset statistics

Compute summary statistics for a fused dataset using the canonical directory or
explicit manifest parquet files.

 - From a canonical directory (expects `manifests/frames.parquet` and
   optionally `manifests/cases.parquet` inside):

```bash
polypdb stats /path/to/canonical
```

- Directly from parquet files:

```bash
polypdb stats-manifests --frames /path/to/frames.parquet \
                        --cases /path/to/cases.parquet \  # optional
                        --json-out stats.json   # optional
```

The command logs a concise summary and can optionally write the full statistics
dictionary to a JSON file with `--json-out`.

### Export manifests as CSV

Export one or more manifest parquet files from a canonical dataset to CSV:

```bash
polypdb export-csv /path/to/canonical /path/to/output_csv_dir
```

Export a subset of tables only:

```bash
polypdb export-csv /path/to/canonical /path/to/output_csv_dir --tables frames cases annotations
```

### Export flat classification dataset

Create a simple ``class_0``/``class_1`` folder structure from a canonical
dataset. Frames where ``has_polyp`` is ``True`` are placed into ``class_1`` and
all others into ``class_0``:

```bash
polypdb export-classification-flat /path/to/canonical /path/to/output_dir
```

### Full preprocessing pipeline

Run conversion, merging and optional dataset preparation in a single step. The
pipeline reads a configuration describing raw dataset locations and output
paths. If the ``prepare`` step is enabled, include a ``prepare_config`` entry
pointing to the Ultralytics configuration file (absolute or relative to
``base_path``). When negatives exist, define their sources under
``negative_sources``:

```yaml
raw_sources:
  piba:
    path: /path/to/raw/piba
    # Prefer fetching from the database when the Excel dump is unavailable
    db_config:
      host: localhost
      user: myuser
      password: <DB_PASSWORD>
      database: pibadb
  polar:
    path: /path/to/raw/polar
    metadata_file: polar_annotations.xlsx
base_path: /path/to/standardized
datasets: [piba, polar]
num_elements: null
prepare: true
prepare_config: ultralytics_config.yaml
negative_sources:
  piba: piba/negatives
  polar: polar/negatives
```

Execute the pipeline:

```bash
polypdb pipeline pipeline_config.yaml
```

To preprocess only one dataset from the configuration and generate its
standardised output without merging others, provide the dataset name:

```bash
polypdb pipeline --dataset polypgen pipeline_config.yaml
```

The PolypGen converter crops black padding and masks the dataset's small
bottom-left viewport by default. To also export the untouched frames, add
``save_raw: true`` under ``raw_sources.polypgen`` in the configuration. The
raw images will be written to a ``raw_img`` directory alongside the cleaned
versions, and their original dimensions and boxes are included in the
annotations file. To build a dataset without any cleaning, set
``skip_cleaning: true`` instead, which bypasses both preprocessing steps and
stores the raw frames in ``img``.

## Image preprocessing utilities

`mask_corner_artifacts` removes static overlays such as timestamps or patient
information from image corners. With ``auto_detect=True`` (default) the
function scans outward from each corner until the pixel values differ from the
corner colour, then masks the discovered triangular region. Small sequences of
different coloured pixels (e.g., logo indicators touching an edge) are ignored
up to ``max_colour_run`` pixels so these artefacts are also covered.

To process a directory of images, masking corner artefacts before cropping, run:

```bash
polypdb mask-corners INPUT_DIR OUTPUT_DIR
```

This command writes cleaned and cropped images to ``OUTPUT_DIR`` using the
auto-detection behaviour. The padding remover assumes the border colour is
``(15,15,15)`` with a tolerance of ``20``. Adjust ``--padding-color`` and
``--padding-threshold`` if your images use different padding:

```bash
polypdb mask-corners INPUT_DIR OUTPUT_DIR \
  --padding-color 0,0,0 \
  --padding-threshold 10
```

`mask_viewport_auto_color_safe` helps remove small auxiliary viewports that
occasionally appear in the bottom-left corner of endoscopic frames. The
utility estimates the viewport's background colour and only applies the mask
when a similar colour is detected, avoiding accidental blackouts.

Process a directory of images with:

```bash
python -m polypdb.mask_viewport INPUT_DIR OUTPUT_DIR
```

The script reports which files were masked and saves the results to
``OUTPUT_DIR`` using the original filenames. Restrict processing to a specific
extension with ``--ext`` (default: ``.jpg``).

### Clean Hyper-Kvasir-unlabeled frames

Frames from the Hyper-Kvasir-unlabeled dataset often contain viewports,
text overlays and padding. Use the provided helper to clean and crop them
before further processing:

```bash
python -m polypdb.clean_hyperkvasir_unlabeled INPUT_DIR OUTPUT_DIR
```

This command masks auxiliary viewports, removes detected text regions with
PaddleOCR, crops borders and strips uniform padding, writing the cleaned
images to ``OUTPUT_DIR``.

## Development

Run tests with `pytest` to verify changes. The suite includes checks for the
viewport masking helper in ``tests/test_utils_viewport_mask.py``:

```bash
pytest
```

### Convert datasets

Convert one or more raw datasets to the standardized layout without merging or
preparing YOLO datasets:

```bash
polypdb convert pipeline_config.yaml            # convert all datasets in config
polypdb convert pipeline_config.yaml --dataset sun  # only SUN
```

This uses the same YAML structure as the pipeline. Keys under
``raw_sources.<dataset>`` are forwarded to the dataset converter. For PIBA you
can fetch metadata directly from MySQL by providing ``db_config``:

```yaml
raw_sources:
  piba:
    path: /path/to/raw/piba
    db_config:
      host: localhost
      user: myuser
      password: <DB_PASSWORD>
      database: pibadb
base_path: /path/to/standardized
datasets: [piba]
```
## History rewrite

The git history of this repository was rewritten once before publication to
remove (a) the image outputs of the root `sun-database-analysis.ipynb` notebook
in the commits where they were embedded and (b) tool co-authorship trailers
from two commit messages. File contents of every other commit are unchanged:
the tree of every commit after `5c9dddc` is identical to the original one.
Because commit hashes depend on their ancestors, all hashes changed (also
because GitHub merge signatures are not preserved). The full mapping, with
author and commit dates and the tree hash, is in
[`docs/history-rewrite.csv`](docs/history-rewrite.csv).

Commits cited by the SSL4POLYP manifests (`generator.git_commit` fields in the
published packs carry the **old** hashes):

| Pack generation | Old hash | New hash | Tag |
|---|---|---|---|
| v2 packs (SUN full/morphology, PolypGen clean) | `7de86f7` | `6dbcb2b` | `v2-packs` |
| v2 subsets and few-shot | `dafb5a5` | `4486226` | `v2-subsets-fewshot` |
| v3 packs (221 CSV) | `d78ed76` | `73729bf` | `v3-packs` |

The final state submitted with the paper is tagged `v3-cmpb-submission`.

## Contributing

Contributions are welcome! Fork the repository and open a pull request with your
proposed changes. Please ensure the test suite passes locally by running
`pytest` before submitting.
