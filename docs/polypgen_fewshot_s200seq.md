# PolypGen few-shot, 200 frames with sequence negatives (`polypgen_fewshot_s200seq`)

Variant of the registered PolypGen few-shot support of 200 frames for the E4
sensitivity analysis. For each of the 15 registered packs (partition `k` = 1..5,
training seed `s` in {13, 29, 47}, pack seed `1000*k + s`) it keeps the 100
positives of the registered `s200` support and replaces its 100 isolated
negatives by 100 frames from negative sequences. The registered packs (CSV and
manifests) are not modified: the script checks their file hashes before and
after, and reads them only.

Script: [`scripts/build_polypgen_fewshot_s200seq.py`](../scripts/build_polypgen_fewshot_s200seq.py).

```bash
python scripts/build_polypgen_fewshot_s200seq.py \
    --packs-root /workspace/data_packs_v3_staging \
    --out /workspace/data_packs_v3_staging/polypgen_fewshot_s200seq
```

`--packs-root` is the v3 directory (the same CSV as `SSL4POLYP/data_packs/v3`; the
manifests there must keep their real `roots`, because frame to sequence ids are read
from `<root>/manifests/frames.parquet` of the PolypGen canonical dataset).
Use pandas 2.x (see the README).

## Rules

1. **Positives.** The positive rows of the registered `s200` pack, same frames,
   same train/val assignment (80 train, 20 val).
2. **Candidate negatives.** Frames of the negative sequences that the registered
   `s500` pack of the same `(k, s)` assigned to its support. The `s500` pack removes
   those sequences from its evaluation split, so no candidate is in it. A sequence
   keeps its `s500` assignment: sequences of the `s500` training support supply
   training negatives, sequences of its validation support supply validation
   negatives. Hence validation negatives always come from sequences different from the
   training ones (the `s500` packs have 9-12 train and 2-5 val sequences).
3. **Quota.** The negatives per split of the registered `s200` (80 train / 20 val:
   the same 80/20 proportion).
4. **Selection (deterministic).** `selection_seed = 200000 + pack_seed`
   (e.g. 201013 for `k=1, s=13`; the offset only separates this derivation from the
   registered seeds). Sequences are ordered by `hmac_sha256(selection_seed,
   sequence_id)` and, within a sequence, frames by `hmac_sha256(selection_seed,
   frame_id)` (ties by id; the HMAC is `polypdb.manifests.core.hmac_rank`). Frames are
   taken round-robin: first-ranked frame of every sequence in order, then the
   second, and so on until the quota is reached. Each sequence therefore contributes
   the same number of frames (+-1). The cap of 5 frames per sequence of the registered
   packs is not applied: 100 frames over at most 14 sequences cannot satisfy it.
5. **Evaluation split.** `test.csv` is byte-identical to the registered `s500`
   `test.csv` (same SHA-256). Using the `s200` test split would put about 3,000 sequence
   frames, including the new negatives, into evaluation.

## Checks (assertions in the script; any failure aborts)

- No frame of the new support (train or val) is in the registered `s500` test split.
- Positives equal the registered `s200` positives, split by split.
- Negatives come only from sequences of the registered `s500` support, train
  negatives from its train sequences and validation negatives from its validation
  sequences; train and validation sequences are disjoint; train and validation frames
  are disjoint; no duplicates.
- Counts per class and split equal the registered `s200` (80/80 train, 20/20 val).
- `test.csv` hash equals the registered `s500` one.
- Registered `s200`/`s500` files unchanged (hash before and after).
- `write_pack` runs `validate_pack` (hashes, counts, frames within the parent pack).

## Output

`polypgen_fewshot_s200seq/k{k}/seed{s}/{train,val,test}.csv` + `manifest.yaml` (same
format as the v3 packs: name, version, policy, row_schema, counts, `hashes` with
SHA-256, `parent_pack`, `roots`, `inputs`, `generator` with tool commit, dirty
flag and environment). Extra keys: `derived_from` (file hashes of the registered `s200` and
`s500` packs) and `notes` (sequences and candidate pool sizes). The directory also has
`index.yaml` (SHA-256 of every CSV) and `composition.csv`.

## Composition

Support = train + val = 200 frames (100 positives, 100 sequence negatives) in every
pack; `candidate frames` is the number of parent frames of the sequences that were
eligible for that split.

| k | s | pack seed | selection seed | train pos / neg-seq | val pos / neg-seq | train seqs (candidate frames) | val seqs (candidate frames) | test pos / neg |
|---|---|---|---|---|---|---|---|---|
| 1 | 13 | 1013 | 201013 | 80 / 80 | 20 / 20 | 10 (1582) | 4 (1102) | 1004 / 506 |
| 1 | 29 | 1029 | 201029 | 80 / 80 | 20 / 20 | 10 (1782) | 4 (999) | 1004 / 409 |
| 1 | 47 | 1047 | 201047 | 80 / 80 | 20 / 20 | 11 (1795) | 3 (663) | 1004 / 732 |
| 2 | 13 | 2013 | 202013 | 80 / 80 | 20 / 20 | 10 (1988) | 4 (684) | 1004 / 518 |
| 2 | 29 | 2029 | 202029 | 80 / 80 | 20 / 20 | 12 (2118) | 2 (222) | 1004 / 850 |
| 2 | 47 | 2047 | 202047 | 80 / 80 | 20 / 20 | 11 (1897) | 3 (431) | 1004 / 862 |
| 3 | 13 | 3013 | 203013 | 80 / 80 | 20 / 20 | 12 (2582) | 2 (306) | 1004 / 302 |
| 3 | 29 | 3029 | 203029 | 80 / 80 | 20 / 20 | 12 (2074) | 2 (430) | 1004 / 686 |
| 3 | 47 | 3047 | 203047 | 80 / 80 | 20 / 20 | 12 (1903) | 2 (256) | 1004 / 1031 |
| 4 | 13 | 4013 | 204013 | 80 / 80 | 20 / 20 | 9 (1986) | 5 (500) | 1004 / 704 |
| 4 | 29 | 4029 | 204029 | 80 / 80 | 20 / 20 | 10 (1503) | 4 (880) | 1004 / 807 |
| 4 | 47 | 4047 | 204047 | 80 / 80 | 20 / 20 | 10 (2043) | 4 (685) | 1004 / 462 |
| 5 | 13 | 5013 | 205013 | 80 / 80 | 20 / 20 | 10 (1456) | 4 (861) | 1004 / 873 |
| 5 | 29 | 5029 | 205029 | 80 / 80 | 20 / 20 | 12 (1974) | 2 (506) | 1004 / 710 |
| 5 | 47 | 5047 | 205047 | 80 / 80 | 20 / 20 | 9 (1454) | 5 (897) | 1004 / 839 |

Sequence ids per pack are in `composition.csv` (`train_sequence_ids`, `val_sequence_ids`).
