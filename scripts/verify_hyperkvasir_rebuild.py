#!/usr/bin/env python3
"""Verify a rebuilt HyperKvasir-unlabeled set and produce review artefacts.

Writes, under ``--out``:

* ``hyperkvasir_unlabeled_r_sha256.csv``  (file, sha256, bytes, width, height) of every cleaned image
* ``hyperkvasir_unlabeled_r_sha256.csv.sha256``  SHA-256 of that list
* ``contact_sheet_pairs.png``  random (original, cleaned) pairs for visual review

Exits non-zero when the cleaned set does not contain exactly one image per raw image.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path) -> tuple[str, str, int, int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return path.name, sha256(path), path.stat().st_size, w, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="directory with the raw images")
    ap.add_argument("--clean", type=Path, required=True, help="directory with the cleaned images")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected", type=int, default=99417)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    raw = {p.name for p in args.raw.iterdir() if p.suffix.lower() == ".jpg"}
    clean = sorted(p for p in args.clean.iterdir() if p.is_file())
    names = {p.name for p in clean}
    print(f"raw={len(raw)} clean={len(clean)} expected={args.expected}")
    missing = sorted(raw - names)
    extra = sorted(names - raw)
    if missing or extra or len(clean) != args.expected:
        print(f"MISSING ({len(missing)}): {missing[:50]}")
        print(f"EXTRA ({len(extra)}): {extra[:50]}")
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(describe, clean))
    csv_path = args.out / "hyperkvasir_unlabeled_r_sha256.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["file", "sha256", "bytes", "width", "height"])
        w.writerows(rows)
    digest = sha256(csv_path)
    (args.out / (csv_path.name + ".sha256")).write_text(f"{digest}  {csv_path.name}\n")
    print(f"list sha256 {digest}")

    rng = random.Random(args.seed)
    picks = rng.sample(sorted(raw), args.pairs)
    t, cols, lab = 240, 4, 18
    rows_n = (len(picks) + cols - 1) // cols
    cell_w, cell_h = 2 * t + 8, t + lab + 6
    sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h), "white")
    d = ImageDraw.Draw(sheet)
    for i, name in enumerate(picks):
        x, y = (i % cols) * cell_w + 2, (i // cols) * cell_h + 2
        for j, base in enumerate((args.raw, args.clean)):
            im = Image.open(base / name).convert("RGB")
            im.thumbnail((t, t))
            sheet.paste(im, (x + j * (t + 4), y))
        d.text((x, y + t + 2), f"{name[:36]}  (raw | clean)", fill="black")
    sheet.save(args.out / "contact_sheet_pairs.png")
    (args.out / "contact_sheet_pairs.txt").write_text("\n".join(picks) + "\n")
    print("contact sheet written:", sheet.size)


if __name__ == "__main__":
    main()
