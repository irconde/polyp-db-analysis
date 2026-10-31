#!/usr/bin/env python3
"""Build the "200 frames, sequence negatives" PolypGen few-shot variant (E4 sensitivity).

For every registered v3 pack (partition ``k`` in 1..5, training seed ``s`` in
13/29/47, pack seed ``1000*k + s``) this writes a new pack whose 200-frame support
keeps the 100 positives of the registered ``s200`` pack but replaces its 100
isolated negatives by 100 negatives taken from negative *sequences*.

The registered packs are only read, never modified (their CSV/manifest hashes are
checked before and after). See ``docs/polypgen_fewshot_s200seq.md`` for the
specification; the rules implemented here are:

* **Positives**: the positive rows of the registered ``s200`` pack, unchanged,
  with the same train/val assignment.
* **Candidate sequences**: the negative sequences that the registered ``s500`` pack of
  the same ``(k, s)`` assigned to its support (train or val). Those sequences are
  excluded from that pack's evaluation split, so none of their frames is evaluated
  there. Their train/val assignment is kept, so validation negatives come from
  sequences different from the training ones.
* **Quota**: as many negatives per split as the registered ``s200`` support has
  (80 train / 20 val, i.e. the same 80/20 proportion).
* **Selection** (deterministic): ``seed_seq = 200000 + pack_seed``. Sequences are
  ordered by ``hmac_sha256(seed_seq, sequence_id)``; the frames of each sequence by
  ``hmac_sha256(seed_seq, frame_id)``; frames are then taken round-robin over the
  ordered sequences (first-ranked frame of every sequence, then the second, ...)
  until the quota is met, so every sequence contributes the same number of frames
  (+-1). The registered "at most 5 frames per sequence" cap is not applied
  because 100 frames over 14 or fewer sequences cannot satisfy it.
* **Evaluation split** (``test.csv``): identical to the registered ``s500`` test
  split (same file, same hash), which is disjoint from every frame of the new
  support.

Usage::

    python scripts/build_polypgen_fewshot_s200seq.py \
        --packs-root /workspace/data_packs_v3_staging \
        --out /workspace/data_packs_v3_staging/polypgen_fewshot_s200seq
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from polypdb.fewshot.polypgen_fewshot import _load_frame_metadata  # noqa: E402
from polypdb.manifests.core import ManifestRow, hmac_rank  # noqa: E402
from polypdb.manifests.io import read_csv, write_pack  # noqa: E402
from polypdb.manifests.provenance import parent_pack_info, tool_provenance  # noqa: E402

SEED_OFFSET = 200_000
SUPPORT_SIZE = 200
PACK_NAME = "polypgen_fewshot_s200seq"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_files(pack: Path) -> Dict[str, str]:
    return {p.name: sha256(p) for p in sorted(pack.iterdir()) if p.is_file()}


def round_robin(
    sequences: List[str],
    frames_by_seq: Dict[str, List[ManifestRow]],
    seed: int,
    quota: int,
) -> List[ManifestRow]:
    """Deterministic, sequence-balanced selection of ``quota`` frames."""

    order = sorted(sequences, key=lambda s: (hmac_rank(seed, s), s))
    ranked = {
        s: sorted(frames_by_seq[s], key=lambda r: (hmac_rank(seed, r.frame_id), r.frame_id))
        for s in order
    }
    if sum(len(v) for v in ranked.values()) < quota:
        raise SystemExit(f"candidate pool smaller than quota {quota}")
    picked: List[ManifestRow] = []
    depth = 0
    while len(picked) < quota:
        for s in order:
            if depth < len(ranked[s]):
                picked.append(ranked[s][depth])
                if len(picked) == quota:
                    break
        depth += 1
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packs-root", type=Path, required=True,
                    help="v3 packs directory (contains _parent/ and k{k}/polypgen_fewshot/seed{s}/s{size})")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--partitions", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[13, 29, 47])
    args = ap.parse_args()

    root: Path = args.packs_root
    parent = root / "_parent" / "polypgen_clean_test_extended"
    parent_manifest = yaml.safe_load((parent / "manifest.yaml").read_text(encoding="utf-8"))
    roots = parent_manifest.get("roots", {}) or {}
    parent_rows = {r.frame_id: r for r in read_csv(parent / "test.csv")}
    meta = _load_frame_metadata(roots, {r.store_id for r in parent_rows.values()}, set(parent_rows))
    seq_of = {fid: m["sequence_id"] for fid, m in meta.items() if m.get("sequence_id")}

    frames_by_seq: Dict[str, List[ManifestRow]] = {}
    for fid, row in parent_rows.items():
        if row.label == 0 and fid in seq_of:
            frames_by_seq.setdefault(seq_of[fid], []).append(row)

    registered_before: Dict[str, Dict[str, str]] = {}
    composition: List[dict] = []
    index: dict = {"seed_offset": SEED_OFFSET, "packs": {}}
    command = "python scripts/build_polypgen_fewshot_s200seq.py " + " ".join(
        shlex.quote(a) for a in sys.argv[1:]
    )

    for k in args.partitions:
        for s in args.train_seeds:
            pack_seed = 1000 * k + s
            seed_seq = SEED_OFFSET + pack_seed
            base = root / f"k{k}" / "polypgen_fewshot" / f"seed{s}"
            reg200, reg500 = base / "s200", base / "s500"
            for p in (reg200, reg500):
                registered_before[str(p)] = pack_files(p)
                gen_seed = yaml.safe_load((p / "manifest.yaml").read_text(encoding="utf-8"))["generator"]["seed"]
                assert gen_seed == pack_seed, f"{p}: generator seed {gen_seed} != {pack_seed}"

            r200 = {sp: read_csv(reg200 / f"{sp}.csv") for sp in ("train", "val")}
            r500 = {sp: read_csv(reg500 / f"{sp}.csv") for sp in ("train", "val", "test")}

            # Positives: exactly those of the registered s200 support.
            pos = {sp: [r for r in r200[sp] if r.label == 1] for sp in ("train", "val")}
            quota = {sp: sum(r.label == 0 for r in r200[sp]) for sp in ("train", "val")}

            # Sequences the registered s500 pack assigned to its support, per split.
            seqs = {
                sp: sorted({seq_of[r.frame_id] for r in r500[sp] if r.label == 0 and r.frame_id in seq_of})
                for sp in ("train", "val")
            }
            assert set(seqs["train"]).isdisjoint(seqs["val"]), "s500 train/val share a sequence"
            assert seqs["train"] and seqs["val"], "s500 has no negative sequences in train or val"

            neg = {
                sp: round_robin(seqs[sp], frames_by_seq, seed_seq, quota[sp]) for sp in ("train", "val")
            }
            rows = {
                sp: [replace(r, split=sp) for r in pos[sp]] + [replace(r, split=sp) for r in neg[sp]]
                for sp in ("train", "val")
            }
            test_rows = r500["test"]

            # ---- checks -------------------------------------------------------
            support_ids = {r.frame_id for sp in rows for r in rows[sp]}
            test_ids = {r.frame_id for r in test_rows}
            assert len(support_ids) == len(rows["train"]) + len(rows["val"]), "duplicate frames in support"
            assert support_ids.isdisjoint(test_ids), "support frame inside the s500 evaluation split"
            for sp in ("train", "val"):
                reg_pos = sorted(r.frame_id for r in r200[sp] if r.label == 1)
                assert sorted(r.frame_id for r in pos[sp]) == reg_pos
                assert all(seq_of.get(r.frame_id) in set(seqs[sp]) for r in neg[sp])
                assert all(r.label == 0 for r in neg[sp])
            train_seq = {seq_of[r.frame_id] for r in neg["train"]}
            val_seq = {seq_of[r.frame_id] for r in neg["val"]}
            assert train_seq.isdisjoint(val_seq), "train/val negatives share a sequence"
            assert {r.frame_id for r in rows["train"]}.isdisjoint({r.frame_id for r in rows["val"]})

            def counts(rs: List[ManifestRow]) -> dict:
                negs = [r for r in rs if r.label == 0]
                nseq = [r for r in negs if r.frame_id in seq_of]
                return {
                    "frames": len(rs),
                    "pos": len(rs) - len(negs),
                    "neg": len(negs),
                    "neg_frame": len(negs) - len(nseq),
                    "neg_seq_frames": len(nseq),
                    "neg_seq_sequences": len({seq_of[r.frame_id] for r in nseq}),
                }

            c = {"train": counts(rows["train"]), "val": counts(rows["val"]), "test": counts(test_rows)}
            summary = [
                {"split": sp, "frames": c[sp]["frames"], "pos_frames": c[sp]["pos"],
                 "neg_frames": c[sp]["neg"], "neg_frame": c[sp]["neg_frame"],
                 "neg_seq_frames": c[sp]["neg_seq_frames"],
                 "neg_seq_sequences": c[sp]["neg_seq_sequences"]}
                for sp in ("train", "val", "test")
            ]
            out_pack = args.out / f"k{k}" / f"seed{s}"
            manifest_meta = {
                "name": PACK_NAME,
                "version": "1.0.0",
                "policy": {
                    "unit_of_split": "frame",
                    "fewshot_budget_S": SUPPORT_SIZE,
                    "variant": "sequence_negatives",
                    "parent_pack": "polypgen_clean_test_extended",
                    "positives_source": "registered polypgen_fewshot_s200 (same frames, same train/val split)",
                    "negatives_source": {
                        "origin": "negative sequences assigned to the support of the registered s500 pack",
                        "train": {"sequence_frames": c["train"]["neg_seq_frames"], "sequences": sorted(train_seq)},
                        "val": {"sequence_frames": c["val"]["neg_seq_frames"], "sequences": sorted(val_seq)},
                    },
                    "class_ratio_per_split": {
                        sp: {"pos": c[sp]["pos"], "neg": c[sp]["neg"]} for sp in ("train", "val", "test")
                    },
                    "neg_sampling": {
                        "strategy": "round_robin_over_hmac_ranked_sequences",
                        "selection_seed": seed_seq,
                        "selection_seed_rule": f"{SEED_OFFSET} + pack_seed",
                        "max_frames_per_sequence": None,
                        "sequence_level_disjointness": True,
                    },
                    "test_split": "identical to the registered polypgen_fewshot_s500 test split",
                    "sampling": (
                        "pos:registered s200; neg:round_robin_over_sequences("
                        f"hmac_sha256(sequence_id|frame_id,{seed_seq})); pack_seed={pack_seed}"
                    ),
                },
                "row_schema": {
                    "fields": [
                        {"name": "dataset", "type": "string", "desc": "constant 'polypgen_clean'"},
                        {"name": "split", "type": "string", "desc": "train|val|test (frame-level)"},
                        {"name": "case_id", "type": "string"},
                        {"name": "frame_id", "type": "string"},
                        {"name": "frame_path", "type": "path"},
                        {"name": "label", "type": "int"},
                    ]
                },
                "selection_keys": {
                    "train": f"hmac_sha256(frame_id,{seed_seq}) within sequence, round-robin across sequences",
                    "val": f"hmac_sha256(frame_id,{seed_seq}) within sequence, round-robin across sequences",
                    "test": "registered s500 test split",
                },
                "counts": c,
                "summary_by_split": summary,
                "notes": {
                    "s500_train_sequences": seqs["train"],
                    "s500_val_sequences": seqs["val"],
                    "candidate_frames": {
                        sp: sum(len(frames_by_seq[x]) for x in seqs[sp]) for sp in ("train", "val")
                    },
                },
                "roots": roots,
                "parent_pack": parent_pack_info(parent),
                "derived_from": {
                    "polypgen_fewshot_s200": {"path": str(reg200), "files_sha256": pack_files(reg200)},
                    "polypgen_fewshot_s500": {"path": str(reg500), "files_sha256": pack_files(reg500)},
                },
                "generator": {
                    "repo": "polyp-db-analysis",
                    "git_commit": tool_provenance()["git_commit"],
                    "command": command,
                    "seed": pack_seed,
                },
            }
            write_pack(out_pack, {"train": rows["train"], "val": rows["val"], "test": test_rows}, manifest_meta)

            # test.csv must be byte-identical to the registered s500 test split.
            assert sha256(out_pack / "test.csv") == sha256(reg500 / "test.csv"), "test split not identical to s500"

            hashes = yaml.safe_load((out_pack / "manifest.yaml").read_text(encoding="utf-8"))["hashes"]
            index["packs"][f"k{k}/seed{s}"] = {"pack_seed": pack_seed, "selection_seed": seed_seq, "csv_sha256": hashes}
            composition.append({
                "k": k, "seed": s, "pack_seed": pack_seed, "selection_seed": seed_seq,
                "train_pos": c["train"]["pos"], "train_neg_seq": c["train"]["neg"],
                "val_pos": c["val"]["pos"], "val_neg_seq": c["val"]["neg"],
                "train_sequences": len(train_seq), "val_sequences": len(val_seq),
                "train_candidate_frames": sum(len(frames_by_seq[x]) for x in seqs["train"]),
                "val_candidate_frames": sum(len(frames_by_seq[x]) for x in seqs["val"]),
                "test_pos": c["test"]["pos"], "test_neg": c["test"]["neg"],
                "support_in_test": len(support_ids & test_ids),
                "train_sequence_ids": " ".join(sorted(train_seq)),
                "val_sequence_ids": " ".join(sorted(val_seq)),
            })

    # Registered packs must be untouched.
    for p, before in registered_before.items():
        assert pack_files(Path(p)) == before, f"registered pack modified: {p}"

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "composition.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(composition[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(composition)
    (args.out / "index.yaml").write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")
    print(f"wrote {len(composition)} packs to {args.out}; registered packs unchanged")


if __name__ == "__main__":
    main()
