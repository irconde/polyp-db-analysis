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
paths:

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
```

Execute the pipeline:

```bash
polypdb pipeline pipeline_config.yaml
```

## Development

Run tests with `pytest` to verify changes.
