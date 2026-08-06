# SNN–HDC experiments (`snnhdc*.py` at repo root)

These scripts explore HRR / FHRR-style vector representations **alongside** the main `sspslam` package. They are **not** imported by core SLAM code and live at the **repository root** for active development (e.g. `snnhdc.py`, `snnhdc_fhrr.py`, `snnhdc_hrr.py`, plus phased FHRR helpers `snnhdc_fhrr_*.py`).

Runs may write model checkpoints under `./snnhdc-models/` (see each file).

## Phased FHRR scripts

- `snnhdc.py`: original Cedric-style SNN-HDC baseline. Keep this unchanged for comparison.
- `snnhdc_fhrr_eval.py`: frozen/no-backprop FHRR phase. Builds raw bundled class prototypes from frozen SNN features.
- `snnhdc_fhrr_train.py`: inline-trained FHRR phase. Trains the SNN through a fixed random FHRR projection and real-part complex-inner-product logits.
- `snnhdc_fhrr_capacity.py`: synthetic dimension/sigma sweep for capacity demonstrations.
- `snnhdc_fhrr_smoke.py`: quick algebra and gradient checks for the FHRR implementation.
- `snnhdc_fhrr_common.py`: shared utilities for the phased FHRR scripts.

`snnhdc_hrr.py` and `snnhdc_fhrr.py` are the main HRR / bundled-FHRR training scripts; the `snnhdc_fhrr_*.py` helpers support the phased eval/train/capacity/smoke workflow.
