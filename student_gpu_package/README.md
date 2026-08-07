# ConceptGraphs head-to-head — end-to-end package (GPU)

Goal: score **ConceptGraphs and our VSA memory on the SAME Replica scenes,
under CONCEPTGRAPHS' OWN evaluation** (closed-set semantic segmentation,
mAcc / F-mIoU, their scorer, their ground truth). No invented metrics: our
system is made to produce the same output type their eval consumes (a class
label per GT point), so one scorer judges both.

Time budget: ~1 evening for room0, ~1 day for all 8 scenes.
Everything below is scripted; run the stages in order. Each stage prints
`STAGE OK` on success — do not continue past a failure, send Paul the log.

## What you produce (send back to Paul)

```
handoff/
  <scene>/cg_labels.npz          their system's per-point labels (stage 3)
  <scene>/cg_objects.json        their object map [{class, x, y, z}]
  <scene>/cg_observations.json   their per-frame observation stream
  <scene>/vsa_labels.npz         our system's per-point labels (stage 4)
  <scene>/scores.json            both systems, same scorer (stage 5)
  environment.txt                exact package versions (stage 0 writes it)
```

## Stage 0 — environment (once)

```bash
bash 00_setup.sh
```
Creates the `cgraphs` conda env, clones the official repo
(https://github.com/concept-graphs/concept-graphs), installs their
dependencies + SAM/CLIP checkpoints, and writes `environment.txt`.
Their README's install section is authoritative if anything conflicts;
record any deviation in `environment.txt`.

## Stage 1 — data + verification (ALWAYS run this before stage 2)

You need the same rendered sequences Paul ran (NICE-SLAM/vMAP Replica pack).
Fastest: copy `data/replica/<scene>/` from Paul's machine (rgb + depth +
traj.txt + poses.csv, ~5 GB/scene), or run his fetcher on yours:

```bash
python tools/prepare_replica.py --scene room0     # from the main repo
```

Also copy `outputs/replica_<scene>/object_points.json` and
`outputs/replica_<scene>/detections_crops.csv` (tiny) — our frontend's
observations, so stage 4 uses the identical perception Paul measured.

Then:

```bash
python 01_check_data.py --scene room0
```

This verifies every file the later stages need (with the exact fix command
per missing item) AND builds `cg_dataset/<scene>/results/` — the directory
layout ConceptGraphs' Replica dataloader expects (frame%06d naming, traj.txt
beside it, camera params json). Our extraction is flat, theirs is nested;
without this shim stage 2 fails on data loading, guaranteed.

## Stage 2 — run ConceptGraphs (their code, unmodified)

```bash
bash 02_run_conceptgraphs.sh room0
```
Wraps their documented Replica pipeline (room0 is their demo scene). GPU
needed (SAM + CLIP). Output: their result pkl/gz under `cg_out/<scene>/`.
If their entry-point names have drifted from the script's assumptions, the
script prints the expected vs found layout — fix the two PATH variables at
the top, nothing else.

## Stage 3 — export their outputs to the handoff schema

```bash
python 03_export_cg.py --scene room0
```
Reads their result file, writes `cg_objects.json`, `cg_observations.json`,
and `cg_labels.npz` (their per-point class prediction on the eval point
cloud — uses their own label-assignment code path).

## Stage 4 — our system, same output type, same vocabulary

```bash
python 04_vsa_labels.py --scene room0
```
End-to-end on our side (runs on GPU or CPU):
1. loads our observation stream (`object_points.json` — the SAME
   detections Paul's numbers use);
2. re-labels each detection crop with CLIP against the **Replica class
   list** (their vocabulary, their prompt template — removes the COCO
   vocabulary gap; cached to `crop_clip_replica.pt`);
3. builds the 32 KB trace (class ⊗ position, bounded insertion);
4. for every point in the eval point cloud: queries every class field at
   that point's floor coordinates, assigns argmax class →
   `vsa_labels.npz`. One trace, dense labels — the memory is now speaking
   their eval's language.

## Stage 5 — one scorer, both systems

```bash
python 05_score.py --scene room0
```
Computes mAcc and F-mIoU for `cg_labels.npz` and `vsa_labels.npz` against
the GT point labels with the standard formulas (identical to their eval's
definitions; if their repo's eval function imports cleanly, it is used
directly and the script says so). Writes `scores.json`:

```json
{"scene": "room0",
 "conceptgraphs": {"mAcc": ..., "fmiou": ...},
 "vsa":           {"mAcc": ..., "fmiou": ...},
 "scorer": "conceptgraphs-repo | reimplementation (formulas identical)",
 "n_points": ..., "n_classes_gt": ...}
```

Sanity anchor: their room0 numbers should land near the paper's Replica
average (mAcc ≈ 40, F-mIoU ≈ 36). If they are wildly off, stage 2's config
drifted — stop and send logs.

## Known result going in (so nobody is surprised)

Paul's CPU smoke run of stages 4-5 on room0 (their side absent) gives our
system **mAcc ≈ 0.08** under the all-classes dense-labelling metric. This is
expected and honest: 29 GT classes include wall/floor/ceiling/rug — the
majority of all points — which a detector-fed OBJECT memory cannot label,
while their SAM pipeline segments everything. The decisive detail is which
classes THEIR eval code actually scores (several open-vocab papers exclude
structure/"stuff" classes): when you run stage 5 with their repo importable,
record which class list their function uses. If they score all classes, our
honest headline under their metric is low and we say so; the object-classes
slice is then reported ONLY as their own eval reports it, never as our own
invented subset.

## Troubleshooting — the known fail points, in the order you'll hit them

| symptom | cause | fix |
|---|---|---|
| `conda: command not found` | conda not on PATH | `source ~/miniconda3/etc/profile.d/conda.sh` then re-run 00 |
| torch reports no CUDA / wrong CUDA | wheel/driver mismatch | 00 auto-picks cu121/cu118 from `nvidia-smi`; if drivers are older, `pip install torch --index-url .../cu118` |
| `pip install -e .` fails in their repo | their deps drifted | follow THEIR README install section verbatim, then re-run 00 (idempotent) |
| SAM/CLIP download hangs | no internet on compute node | run 00 on the login node, or pre-download the two checkpoints and drop them in `concept-graphs/checkpoints/` |
| stage 2: "none of the candidate entry points import" | their code moved again | their README 'Usage' has the current Replica command — add its module name to `CANDIDATES` in 02 (top of file) |
| stage 2: CUDA out of memory | SAM vit_h needs ~10 GB | switch their config to `sam_vit_b` (checkpoint: `sam_vit_b_01ec64.pth`, same URL pattern) — record the change in environment.txt |
| stage 2: dataloader can't find frames | ran without stage 1 | `python 01_check_data.py --scene <s>` builds `cg_dataset/<scene>/results/` |
| stage 2: hydra/config name errors | their CLI grammar changed | copy the exact command from their README; only `dataset_root`/`scene_id`/`save_dir` matter to later stages |
| stage 3: "cannot find point data" | result pkl schema drifted | the script prints the first object's keys — send that log to Paul, do not guess |
| stage 5: MemoryError | scipy missing (brute-force path) | `pip install scipy` (00 installs it; check you're in `cgraphs`) |
| their room0 mAcc lands far from ~0.40 | config drift in stage 2 | stop; send `cg_out/<scene>/run.log` — do not tune toward the paper number |

Rule for anything not in this table: stop at the first failed stage, send the
stage log. A partial handoff with a clear log is worth more than a guessed fix.

## Notes that will save you time

- Record GPU + wall-clock per stage 2 run (goes in the systems table).
- Do NOT tune anything on either side. The point is the comparison, not
  the score. If something looks unfair, note it, don't fix it.
- The observation-stream export (stage 3) also enables the 2×2 on Paul's
  side (their frontend × our backend) — that's why it matters.
- Any file you can't produce: send the stage log instead; partial handoffs
  are fine.
