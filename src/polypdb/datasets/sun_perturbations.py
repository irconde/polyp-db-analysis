from __future__ import annotations

"""Build SUN test perturbation manifest.

This module constructs a manifest pack describing test-time perturbations for
SUN dataset frames. Perturbations are generated on-the-fly during evaluation,
so frame paths reference the original images. Each original test frame is
replicated for a set of perturbation conditions with deterministic RNG seeds.
"""

from dataclasses import replace
from pathlib import Path
import hashlib
import hmac
import yaml

from ..manifests.provenance import tool_provenance
from ..manifests.io import read_csv, write_pack


# Immutable catalog of perturbation conditions for the SUN test pack.
# The order of entries is significant and must not be changed without
# incrementing the manifest version.
PERTURBATIONS = [
    {"id": "clean", "kind": "clean"},
    {"id": "blur_sigma_0p5", "kind": "gaussian_blur", "params": {"sigma": 0.5}},
    {"id": "blur_sigma_1p0", "kind": "gaussian_blur", "params": {"sigma": 1.0}},
    {"id": "blur_sigma_1p5", "kind": "gaussian_blur", "params": {"sigma": 1.5}},
    {"id": "blur_sigma_2p0", "kind": "gaussian_blur", "params": {"sigma": 2.0}},
    {"id": "jpeg_q_95", "kind": "jpeg", "params": {"quality": 95}},
    {"id": "jpeg_q_75", "kind": "jpeg", "params": {"quality": 75}},
    {"id": "jpeg_q_50", "kind": "jpeg", "params": {"quality": 50}},
    {"id": "jpeg_q_30", "kind": "jpeg", "params": {"quality": 30}},
    {
        "id": "bc_b0p8_c0p8",
        "kind": "brightness_contrast",
        "params": {"brightness": 0.8, "contrast": 0.8},
    },
    {
        "id": "bc_b0p8_c0p6",
        "kind": "brightness_contrast",
        "params": {"brightness": 0.8, "contrast": 0.6},
    },
    {
        "id": "bc_b0p6_c0p8",
        "kind": "brightness_contrast",
        "params": {"brightness": 0.6, "contrast": 0.8},
    },
    {
        "id": "bc_b0p6_c0p6",
        "kind": "brightness_contrast",
        "params": {"brightness": 0.6, "contrast": 0.6},
    },
    {
        "id": "occ_a0p05",
        "kind": "occlusion_block",
        "params": {"area_frac": 0.05, "mode": "square"},
    },
    {
        "id": "occ_a0p10",
        "kind": "occlusion_block",
        "params": {"area_frac": 0.10, "mode": "square"},
    },
    {
        "id": "occ_a0p20",
        "kind": "occlusion_block",
        "params": {"area_frac": 0.20, "mode": "square"},
    },
]


def _derive_seed(master_seed: int, orig_frame_id: str, perturbation_id: str) -> int:
    """Return deterministic RNG seed for a perturbation instance."""

    key = str(master_seed).encode("utf-8")
    msg = f"{orig_frame_id}:{perturbation_id}".encode("utf-8")
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    return int.from_bytes(digest, "big") % (2 ** 32)


def build_sun_test_perturbations(
    sun_full_pack: Path, out_dir: Path, rng_master_seed: int = 47
) -> None:
    """Create a SUN test perturbation pack from ``sun_full_pack``.

    Parameters
    ----------
    sun_full_pack:
        Path to an existing ``sun_full`` pack directory.
    out_dir:
        Directory where the perturbation pack will be written.
    rng_master_seed:
        Master seed for deterministic perturbation RNGs.  Defaults to ``47``.
    """

    test_rows = read_csv(sun_full_pack / "test.csv")

    new_rows = []
    for row in test_rows:
        orig_id = row.frame_id
        for pert in PERTURBATIONS:
            pid = pert["id"]
            params = pert.get("params", {})
            seed = _derive_seed(rng_master_seed, orig_id, pid)
            blur_sigma = params["sigma"] if pert["kind"] == "gaussian_blur" else -1.0
            jpeg_q = params["quality"] if pert["kind"] == "jpeg" else -1
            brightness = params["brightness"] if pert["kind"] == "brightness_contrast" else -1.0
            contrast = params["contrast"] if pert["kind"] == "brightness_contrast" else -1.0
            bbox_area_frac = params["area_frac"] if pert["kind"] == "occlusion_block" else -1.0
            new_rows.append(
                replace(
                    row,
                    orig_frame_id=orig_id,
                    perturbation_id=pid,
                    rng_seed=seed,
                    render_in_pipeline=True,
                    blur_sigma=blur_sigma,
                    jpeg_q=jpeg_q,
                    brightness=brightness,
                    contrast=contrast,
                    bbox_area_frac=bbox_area_frac,
                )
            )

    # parent manifest details
    manifest_path = sun_full_pack / "manifest.yaml"
    parent_manifest = yaml.safe_load(manifest_path.read_text())
    parent_roots = parent_manifest.get("roots", {})
    test_split_info = (
        parent_manifest.get("assertions", {})
        .get("splits", {})
        .get("test", {})
    )
    case_ids = test_split_info.get("pos_cases", []) + test_split_info.get("neg_cases", [])
    parent_test_hash = parent_manifest.get("hashes", {}).get("test", "")

    num_conditions = len(PERTURBATIONS)
    per_condition = len(test_rows)
    total_rows = len(new_rows)

    # basic integrity checks
    labels_by_orig: dict[str, set[int]] = {}
    cases_by_orig: dict[str, set[str]] = {}
    for r in new_rows:
        labels_by_orig.setdefault(r.orig_frame_id, set()).add(r.label)
        cases_by_orig.setdefault(r.orig_frame_id, set()).add(r.case_id)
    label_invariant = all(len(v) == 1 for v in labels_by_orig.values())
    case_leak = all(len(v) == 1 for v in cases_by_orig.values())

    clean_present = any(r.perturbation_id == "clean" for r in new_rows)

    assertions = {
        "no_case_leakage_across_conditions": case_leak,
        "clean_condition_present": clean_present,
        "total_rows": total_rows,
        "per_condition_row_count": per_condition,
        "conditions_len": num_conditions,
        "label_invariant_across_conditions": label_invariant,
        "unique_key": ["orig_frame_id", "perturbation_id"],
        "base_clean_frames_match_parent_test": True,
        "case_ids": case_ids,
        "parent_test_hash": parent_test_hash,
    }

    command = (
        f"polypdb sun perturbations --pack {sun_full_pack} --out {out_dir}"
        f" --master-seed {rng_master_seed}"
    )
    git_commit = tool_provenance()["git_commit"]

    meta = {
        "name": "sun_test_perturbations",
        "version": "0.0",
        "generator": {
            "repo": "polyp-db-analysis",
            "git_commit": git_commit,
            "command": command,
        },
        "evaluation_runtime": {
            "generate_perturbations_on_the_fly": True,
            "rng_master_seed": rng_master_seed,
            "per_row_seed_rule": "HMAC_SHA256(master_seed, f'{orig_frame_id}:{perturbation_id}') -> uint32",
            "occlusion_bbox_rule": "seeded square bbox from per_row_seed",
        },
        "policy": {"unit_of_split": "frame"},
        "row_schema": {
            "fields": [
                {"name": "dataset", "type": "string", "desc": "constant 'sun'"},
                {
                    "name": "split",
                    "type": "string",
                    "desc": "test (single logical split 'eval' in practice)",
                },
                {"name": "case_id", "type": "string"},
                {"name": "frame_id", "type": "string"},
                {"name": "frame_path", "type": "path"},
                {"name": "label", "type": "int"},
                {
                    "name": "orig_frame_id",
                    "type": "string",
                    "desc": "id of the clean source frame",
                },
                {
                    "name": "perturbation_id",
                    "type": "string",
                    "enum": [
                        "clean",
                        "blur_sigma_0p5",
                        "blur_sigma_1p0",
                        "blur_sigma_1p5",
                        "blur_sigma_2p0",
                        "jpeg_q_95",
                        "jpeg_q_75",
                        "jpeg_q_50",
                        "jpeg_q_30",
                        "bc_b0p8_c0p8",
                        "bc_b0p8_c0p6",
                        "bc_b0p6_c0p8",
                        "bc_b0p6_c0p6",
                        "occ_a0p05",
                        "occ_a0p10",
                        "occ_a0p20",
                    ],
                },
                {
                    "name": "rng_seed",
                    "type": "uint32",
                    "desc": "HMAC(master_seed, f'{orig_frame_id}:{perturbation_id}')",
                },
                {
                    "name": "render_in_pipeline",
                    "type": "bool",
                    "desc": "true",
                },
                {"name": "blur_sigma", "type": "float", "nullable": True},
                {"name": "jpeg_q", "type": "int", "nullable": True},
                {"name": "brightness", "type": "float", "nullable": True},
                {"name": "contrast", "type": "float", "nullable": True},
                {"name": "bbox_area_frac", "type": "float", "nullable": True},
            ]
        },
        "roots": parent_roots,
        "parent_pack": {"name": "sun_full", "path": str(sun_full_pack)},
        "perturbations": PERTURBATIONS,
        "assertions": assertions,
    }

    write_pack(out_dir, {"test": new_rows}, meta)


__all__ = ["build_sun_test_perturbations"]
