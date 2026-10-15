#!/usr/bin/env python3
"""Build the v3 partitions (SUN morphology + subsets + PolypGen few-shot).

Every pack is produced through the public ``polypdb`` CLI so the commands
recorded in the manifests are exactly the ones executed here.

Layout written under ``--out``::

    _parent/polypgen_clean_test_extended
    k{n}/base                              # sun morph pack
    k{n}/sun_subsets/sun_pXX
    k{n}/polypgen_fewshot/seed{s}/s{size}
    index.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cli(*args: object) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "polypdb.cli", *map(str, args)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def must(proc: subprocess.CompletedProcess, what: str) -> None:
    if proc.returncode != 0:
        raise SystemExit(f"{what} failed:\n{proc.stderr[-2000:]}")


def failure_reason(proc: subprocess.CompletedProcess) -> str:
    lines = [l for l in proc.stderr.strip().splitlines() if l.strip()]
    return lines[-1] if lines else f"exit code {proc.returncode}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sun-root", type=Path, required=True)
    ap.add_argument("--polypgen-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k1-seed", type=int, default=47)
    ap.add_argument("--candidate-seeds", type=int, nargs="+", default=list(range(101, 121)))
    ap.add_argument("--partitions", type=int, default=5)
    ap.add_argument("--subset-seed", type=int, default=13)
    ap.add_argument("--subset-percents", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[13, 29, 47])
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500])
    ap.add_argument("--reference-packs", type=Path, help="existing data_packs/ used to check k=1 hashes")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    # PolypGen parent (deterministic, no seed)
    parent = out / "_parent" / "polypgen_clean_test_extended"
    must(
        cli("polypgen", "build", "--root", args.polypgen_root, "--out", parent,
            "--track", "clean", "--include-negative-sequences"),
        "polypgen parent build",
    )

    # Partitions
    partitions: dict[int, int] = {}
    failed: list[dict] = []
    k1 = out / "k1" / "base"
    must(cli("sun", "morph", "--root", args.sun_root, "--out", k1, "--seed", args.k1_seed),
         "k=1 sun morph")
    partitions[1] = args.k1_seed
    pending = iter(args.candidate_seeds)
    for k in range(2, args.partitions + 1):
        while True:
            try:
                seed = next(pending)
            except StopIteration:
                raise SystemExit(f"ran out of candidate seeds before reaching k={k}")
            base = out / f"k{k}" / "base"
            proc = cli("sun", "morph", "--root", args.sun_root, "--out", base, "--seed", seed)
            if proc.returncode == 0:
                partitions[k] = seed
                break
            failed.append({"seed": seed, "reason": failure_reason(proc)})
            shutil.rmtree(out / f"k{k}", ignore_errors=True)

    # Subsets and few-shot per partition
    for k in partitions:
        must(
            cli("sun", "subsets-canonical", "--pack", out / f"k{k}" / "base",
                "--percents", *args.subset_percents, "--seed", args.subset_seed,
                "--out", out / f"k{k}" / "sun_subsets"),
            f"k={k} subsets",
        )
        for s in args.train_seeds:
            for size in args.sizes:
                must(
                    cli("polypgen", "fewshot", "--pack", parent, "--sizes", size,
                        "--seeds", 1000 * k + s,
                        "--out", out / f"k{k}" / "polypgen_fewshot" / f"seed{s}",
                        "--dir-template", "s{size}"),
                    f"k={k} fewshot seed={s} size={size}",
                )

    # Index
    index: dict = {"partitions": {}, "failed_partition_seeds": failed}
    for k, seed in partitions.items():
        kdir = out / f"k{k}"
        index["partitions"][k] = {
            "partition_seed": seed,
            "subset_seed": args.subset_seed,
            "fewshot_pack_seeds": {s: 1000 * k + s for s in args.train_seeds},
            "csv_sha256": {
                p.relative_to(out).as_posix(): sha256(p) for p in sorted(kdir.rglob("*.csv"))
            },
        }
    index["parent"] = {
        p.relative_to(out).as_posix(): sha256(p) for p in sorted((out / "_parent").rglob("*.csv"))
    }
    k1_check = {}
    if args.reference_packs:
        ref = yaml.safe_load((args.reference_packs / "sun_morphology" / "manifest.yaml").read_text())
        for split, expected in ref["hashes"].items():
            k1_check[split] = sha256(k1 / f"{split}.csv") == expected
        refp = yaml.safe_load(
            (args.reference_packs / "polypgen_clean_test_extended" / "manifest.yaml").read_text()
        )
        k1_check["polypgen_clean_test_extended"] = (
            sha256(parent / "test.csv") == refp["hashes"]["test"]
        )
    index["k1_matches_current_packs"] = k1_check
    (out / "index.yaml").write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")
    if k1_check and not all(k1_check.values()):
        raise SystemExit(f"k=1 hash check FAILED: {k1_check}")
    print(f"done: partitions {partitions}; failed seeds {failed}")


if __name__ == "__main__":
    main()
