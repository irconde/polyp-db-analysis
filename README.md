# Polyp DB Analysis

Utilities for merging colonoscopy polyp datasets and preparing them for object detection or classification experiments.

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

## CLI usage

Each command expects a YAML configuration file. Example configs are provided in
`merge_config.example.yaml` and `ultralytics_config.example.yaml`.

The CLI also accepts a global ``--log-level`` option to control verbosity. Place
it before the subcommand:

```bash
polypdb --log-level DEBUG pipeline pipeline_config.yaml
```

By default the log level is ``INFO``.

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
Negative entries appear in ``frames.parquet`` with ``has_polyp`` set to ``False``
and ``global_lesion_id`` left ``NULL``. If the dataset lacks per-case identifiers,
the merge step skips generating a ``cases.parquet`` file.

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
polypdb export-csv /path/to/canonical /path/to/output_csv_dir --tables frames lesions
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
    metadata_file: piba_db_dump.xlsx
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

## Contributing

Contributions are welcome! Fork the repository and open a pull request with your
proposed changes. Please ensure the test suite passes locally by running
`pytest` before submitting.
