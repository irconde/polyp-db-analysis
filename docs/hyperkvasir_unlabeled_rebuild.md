# HyperKvasir-unlabeled (cleaned, "-r") rebuild

The preprocessed HyperKvasir-unlabeled set used for MAE pre-training (SSL-colon-r)
was not kept, so it was rebuilt from the official download on 2026-10-29.
This document records everything needed to audit or repeat the rebuild. Images are
not stored in the repository, only their hashes.

## 1. Source data

| Item | Value |
|---|---|
| Official page | https://datasets.simula.no/hyper-kvasir/ (links to OSF: https://osf.io/mh9sj/) |
| Downloaded file | https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-unlabeled-images.zip |
| Stored as (unmodified) | `/workspace/data/raw/hyperkvasir/hyper-kvasir-unlabeled-images.zip` |
| Download started / zip verified complete | 2026-10-29 06:56 UTC / 08:06 UTC |
| Size | 29,402,908,886 bytes (equal to the server `Content-Length`) |
| SHA-256 | `684d0bbdfd41eefeece7e72c94a2ada2cff83105248dd3da93e8f91218b3ad21` |
| Server headers | `Last-Modified: Mon, 10 Aug 2020 15:01:58 GMT`, `ETag: "6d88cc8d6-5ac873c7c72c1"` |
| Integrity | `unzip -t`: no errors |
| Contents | 99,417 `.jpg` files in `unlabeled-images/images/` (no duplicate basenames) plus `unlabeled-images/license.txt` |

Extracted (not modified) to `/workspace/data/hyperkvasir-unlabeled-extracted/`; the
cleaning input is `unlabeled-images/images/` (flat directory, 99,417 files).

**TLS note.** `datasets.simula.no` serves an incomplete certificate chain (the
Sectigo intermediate "Sectigo Public Server Authentication CA DV R36" is missing), so
default `curl` fails verification. Verification was **not** disabled: the missing
intermediate was fetched from the certificate's own AIA URL
(`http://crt.sectigo.com/SectigoPublicServerAuthenticationCADVR36.crt`), appended to
the system CA bundle and passed with `--cacert`.

**License / terms.** The zip ships `license.txt` (copy in
[`hyperkvasir_unlabeled/license.txt`](hyperkvasir_unlabeled/license.txt)): Creative
Commons Attribution 4.0 International (CC BY 4.0). The dataset page adds under
"Terms of Use": any document or paper that uses the dataset or reports results based
on it must cite Borgli et al., *Scientific Data* 7, 283 (2020),
https://www.nature.com/articles/s41597-020-00622-y.

## 2. Environment

Created from this repository's `environment.yml` (conda-forge) in
`/workspace/envs/hk_ocr`, then `pip install -e ".[ocr]"` (repository at commit
`8f8cfcc72a542c39d9ef218626593dd9c296993d`, the tip of `main` at the time).

| Component | Version |
|---|---|
| Python | 3.12.16 |
| paddlepaddle | 2.6.2 (CPU wheel; no GPU used) |
| paddleocr | 2.10.0 |
| opencv (`cv2.__version__`) | 5.0.0 |
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| libjpeg-turbo (conda) | 3.2.1 |

Full lists: [`conda_packages.txt`](hyperkvasir_unlabeled/conda_packages.txt) and
[`pip_packages.txt`](hyperkvasir_unlabeled/pip_packages.txt).

Caveat: the environment contains several OpenCV distributions that all provide the
same `cv2` package and all have version 5.0.0 (conda `py-opencv`, pip
`opencv-python`, `opencv-python-headless`, `opencv-contrib-python 5.0.0.93`); the pip
`[ocr]` dependencies pulled the pip ones in. The `cv2` actually imported is
`/workspace/envs/hk_ocr/lib/python3.12/site-packages/cv2`. This was not cleaned up,
because changing it after the run would no longer describe what was executed.

**OCR models.** PaddleOCR 2.10.0 downloaded these on first use into
`~/.paddleocr/whl/` (file hashes in
[`ocr_models_sha256.txt`](hyperkvasir_unlabeled/ocr_models_sha256.txt)):

| Role | Model | Source |
|---|---|---|
| Text detection | `en_PP-OCRv3_det_infer` (PP-OCRv3, English) | https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar |
| Recognition | `en_PP-OCRv4_rec_infer` (PP-OCRv4, English) | https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar |
| Angle classifier | `ch_ppocr_mobile_v2.0_cls_infer` (downloaded but unused: `use_angle_cls=False`) | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar |

The cleaning code calls detection only (`PaddleOCR(use_angle_cls=False, lang="en")`).

## 3. Cleaning

Module `polypdb.clean_hyperkvasir_unlabeled`, default parameters (no `--inpaint`, so
masked text is blacked out, not inpainted). Per image: mask the bottom-left viewport
when detected, mask text boxes found by PaddleOCR, crop a 5 px border, crop uniform
padding, fill bottom-right padding. Output files keep the input file name.

Command (one invocation per shard, same code and defaults), run from the repository
at commit `8f8cfcc`:

```bash
/workspace/envs/hk_ocr/bin/python -m polypdb.clean_hyperkvasir_unlabeled \
    <INPUT_DIR> /workspace/data/hyperkvasir-unlabeled-r
```

where `<INPUT_DIR>` is a directory of symlinks to a subset of the raw images (see
below). A single process is not practical (about 1.4 s/image, roughly 35 h), and
cleaning is independent per image, so the result does not depend on the sharding.

**Execution history (all runs wrote into the same output directory):**

1. 24 parallel processes, one per shard. At 10:13 CET the operating system killed
   them for memory use (PaddleOCR processes grow to several GB) with 21,418 images
   written. All 21,418 were decoded successfully, and the 100 most recently written
   were deleted and regenerated in case a write had been cut short.
2. A second attempt with 16 processes and batches of 1,000 images was stopped after a
   few minutes because memory was growing too fast (100 newest files again deleted
   and regenerated).
3. Final run: 16 workers, each running the unmodified CLI once per batch of at most
   150 images (fresh process per batch, which bounds memory to ~60 GB in total),
   over exactly the images not yet present in the output. No batch failed
   (`failures.txt` empty), and no log contains "couldn't read" or a traceback.

Note: `batch_clean_hyperkvasir_unlabeled(resume=True)` builds a list of pending images
but then iterates over all of them, so `resume` has no effect; this is why resuming
was done by restricting the input directory. The code was not changed.

## 4. Verification

Script: [`scripts/verify_hyperkvasir_rebuild.py`](../scripts/verify_hyperkvasir_rebuild.py).

- **Output count:** 99,417 files, all `.jpg`, one per raw image, same file names; no
  missing and no extra files. Total size 17 GB.
- **Hash list:** `hyperkvasir_unlabeled_r_sha256.csv` (columns `file, sha256, bytes,
  width, height`; sorted by file name), stored gzipped as
  [`hyperkvasir_unlabeled/hyperkvasir_unlabeled_r_sha256.csv.gz`](hyperkvasir_unlabeled/hyperkvasir_unlabeled_r_sha256.csv.gz)
  (`gzip -9n`).
  SHA-256 of the uncompressed list:
  `d9ad754e868aabcc41663b41b27c64778c7fa4662a93c3dc65b59cc8a5782b25`
  (also in [`hyperkvasir_unlabeled_r_sha256.csv.sha256`](hyperkvasir_unlabeled/hyperkvasir_unlabeled_r_sha256.csv.sha256)).
  Check with `gunzip -c hyperkvasir_unlabeled_r_sha256.csv.gz | sha256sum`.
- **Visual review:** 40 random (raw, cleaned) pairs, `random.Random(0)`, on a contact
  sheet at `/workspace/data/hk_rebuild_meta/verification/contact_sheet_pairs.png`
  (file names in `contact_sheet_pairs.txt` next to it; not committed). In the sample
  the endoscopic circular view is preserved, on-screen text and the corner viewport
  are blacked out and image padding is cropped.

**Reproducibility caveat.** Pixel-exact reproduction of the hash list is not
guaranteed on another machine: the output depends on PaddleOCR detection results and
on the JPEG encoder in OpenCV/libjpeg-turbo. Re-running in the environment recorded
above is the closest possible; any difference should be checked by comparing against
the list rather than assumed to be a defect.
