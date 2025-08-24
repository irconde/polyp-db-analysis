# Polyp DB Analysis

Utilities for merging colonoscopy polyp datasets and preparing them for object detection or classification experiments.

## Installation

Install the package and its dependencies:

```bash
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
and ``global_lesion_id`` left ``NULL``.

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

## Image preprocessing utilities

`mask_corner_artifacts` removes static overlays such as timestamps or patient
information from image corners. With ``auto_detect=True`` (default) the
function scans outward from each corner until the pixel values differ from the
corner colour, then masks the discovered triangular region.

## Development

Run tests with `pytest` to verify changes.
