"""Route B killer (B1) + cone measurement (E0), one pass, no map, no trace.

Pre-registered gate (scoping report B.5, frozen before this script existed):

    B1: after phasor projection with a shared W, text->crop class assignment
        retains >= 0.80 of raw CLIP cosine argmax at HD=4096.
        Kill condition: retention < 0.80 at 4096 AND not recovered at 16384.

The reference answer for (a) is already on disk: crop_clip_replica.pt is the
raw-cosine argmax over the Replica class list for every room0 crop, produced
by 04_vsa_labels.clip_relabel with the identical model and prompt template.
We re-encode a 500-crop sample and first CHECK we reproduce those labels;
that separates "my encode differs" from "the projection loses the channel".

E0, measured on the same features (the adversarial review's demand -- its
collapse table was synthetic; this replaces it with the real cone):
  - same-class vs cross-class image-image cosine (the cone)
  - image-text cosine for the correct class vs wrong classes (modality gap)
  - both again after mean-centring (non-oracle: each modality centred by its
    OWN sample mean, computable at build time with no labels)

Run (needs torch+transformers -- the lejepa venv):

    .venv_lejepa/Scripts/python.exe collab_tasks/scripts/clip_phasor_retention.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

SCENE = "room0"
N_CROPS = 500
DIMS = (1024, 4096, 16384)
W_SEEDS = (100, 101, 102)          # 3 draws of the shared projection
GATE = dict(metric="clip_retention", scope="phi4096", test="ge", value=0.80)
SAMPLE_SEED = 0


def encode(rows):
    import importlib

    import torch  # noqa: F401
    from transformers import CLIPModel, CLIPProcessor

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", "..", "student_gpu_package"))
    v04 = importlib.import_module("04_vsa_labels")
    from vsa_cognitive_mapping.clip_compat import image_features, text_features
    from vsa_cognitive_mapping.sequences import load_sequence

    classes, _src = v04.replica_class_list(SCENE)
    seq = load_sequence(f"vsa_cognitive_mapping/configs/replica_{SCENE}.json")
    pos = {int(f): i for i, f in enumerate(seq.frame_ids())}

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    with torch.no_grad():
        t = proc(text=[f"an image of a {c}" for c in classes],
                 return_tensors="pt", padding=True)
        T = text_features(model, t)
        T = (T / T.norm(dim=-1, keepdim=True)).numpy().astype(np.float64)
    V = np.empty((len(rows), T.shape[1]))
    with torch.no_grad():
        for s in range(0, len(rows), 32):
            chunk = rows[s:s + 32]
            imgs = []
            for r in chunk:
                img = seq.image(pos[int(r["frame_idx"])])
                imgs.append(img.crop((max(0, float(r["x1"])),
                                      max(0, float(r["y1"])),
                                      min(img.size[0], float(r["x2"])),
                                      min(img.size[1], float(r["y2"])))))
            px = proc(images=imgs, return_tensors="pt")
            E = image_features(model, px)
            V[s:s + 32] = (E / E.norm(dim=-1, keepdim=True)).numpy()
            if (s // 32) % 5 == 0:
                print(f"  encoded {s + len(chunk)}/{len(rows)}", flush=True)
    return V, T, classes


def phasor(X, d, seed):
    """Real (n, din) -> unit-modulus phasors (n, d), shared W per seed."""
    rng = np.random.RandomState(seed)
    W = rng.randn(X.shape[1], 2 * d) / np.sqrt(X.shape[1])
    iq = X @ W
    i, q = iq[:, :d], iq[:, d:]
    return np.exp(1j * np.arctan2(q, i))


def phasor_argmax(Zv, Zt):
    """Score class c for crop k by Re mean_j Zv[k,j] * conj(Zt[c,j])."""
    S = (Zv @ np.conj(Zt.T)).real / Zv.shape[1]
    return S.argmax(1)


def cone_stats(V, T, y):
    """y: reference class index per crop (from raw argmax)."""
    out = {}
    G = V @ V.T
    same = [G[i, j] for i in range(len(y)) for j in range(i + 1, len(y))
            if y[i] == y[j]]
    cross = [G[i, j] for i in range(len(y)) for j in range(i + 1, len(y))
             if y[i] != y[j]]
    C = V @ T.T
    correct = C[np.arange(len(y)), y]
    wrong = (C.sum(1) - correct) / (C.shape[1] - 1)
    out["img_img_same"] = float(np.mean(same))
    out["img_img_cross"] = float(np.mean(cross))
    out["img_txt_correct"] = float(np.mean(correct))
    out["img_txt_wrong"] = float(np.mean(wrong))
    return out


def main():
    rows = list(csv.DictReader(open(
        f"outputs/replica_{SCENE}/detections_crops.csv")))
    rng = np.random.RandomState(SAMPLE_SEED)
    rows = [rows[i] for i in sorted(rng.choice(len(rows), N_CROPS,
                                               replace=False))]
    print(f"encoding {N_CROPS} of {SCENE}'s crops with ViT-B/32 (CPU)...")
    V, T, classes = encode(rows)

    # ---- sanity: do we reproduce the stored raw-argmax labels? ----
    import torch
    ref = torch.load(f"outputs/replica_{SCENE}/crop_clip_replica.pt",
                     map_location="cpu", weights_only=False)
    raw = (V @ T.T).argmax(1)
    ref_idx = np.array([classes.index(ref[int(r["det_id"])])
                        if ref.get(int(r["det_id"])) in classes else -1
                        for r in rows])
    agree_ref = float((raw == ref_idx).mean())
    print(f"\nraw-argmax agreement with stored crop_clip_replica.pt: "
          f"{agree_ref:.4f}"
          + ("  OK" if agree_ref > 0.95 else "  *** encode mismatch"))

    # ---- E0: the cone, raw and centred ----
    print("\nE0 -- the measured CLIP cone (ViT-B/32, real crops)")
    s_raw = cone_stats(V, T, raw)
    Vc = V - V.mean(0, keepdims=True)
    Tc = T - T.mean(0, keepdims=True)
    Vc /= np.linalg.norm(Vc, axis=1, keepdims=True)
    Tc /= np.linalg.norm(Tc, axis=1, keepdims=True)
    s_cen = cone_stats(Vc, Tc, raw)
    print(f"{'':>24}{'raw':>10}{'centred':>10}")
    for k in s_raw:
        print(f"{k:>24}{s_raw[k]:>10.3f}{s_cen[k]:>10.3f}")

    # ---- B1: retention through the phasor projection ----
    print(f"\nB1 -- retention of text->crop argmax through phi_W "
          f"(vs raw argmax, {N_CROPS} crops, {len(classes)} classes)")
    print(f"{'d':>7}{'retention raw':>15}{'+-sd':>7}{'centred':>10}{'+-sd':>7}")
    res = {}
    for d in DIMS:
        r_raw, r_cen = [], []
        for ws in W_SEEDS:
            Z = phasor(np.vstack([V, T]), d, ws)
            r_raw.append(float(
                (phasor_argmax(Z[:len(V)], Z[len(V):]) == raw).mean()))
            Z2 = phasor(np.vstack([Vc, Tc]), d, ws)
            r_cen.append(float(
                (phasor_argmax(Z2[:len(V)], Z2[len(V):]) == raw).mean()))
        res[d] = dict(raw=(float(np.mean(r_raw)), float(np.std(r_raw))),
                      centred=(float(np.mean(r_cen)), float(np.std(r_cen))))
        print(f"{d:>7}{res[d]['raw'][0]:>15.3f}{res[d]['raw'][1]:>7.3f}"
              f"{res[d]['centred'][0]:>10.3f}{res[d]['centred'][1]:>7.3f}")

    ret = res[4096]["raw"][0]
    verdict = "PASS" if ret >= GATE["value"] else (
        "FAIL-SOFT (recovered at 16384)"
        if res[16384]["raw"][0] >= GATE["value"] else "FAIL")
    print(f"\nGATE B1: retention@4096 = {ret:.3f} vs {GATE['value']}"
          f"  ->  {verdict}")

    os.makedirs("outputs/routeB", exist_ok=True)
    json.dump(dict(scene=SCENE, n_crops=N_CROPS, n_classes=len(classes),
                   sample_seed=SAMPLE_SEED, w_seeds=list(W_SEEDS),
                   agree_with_stored=agree_ref,
                   cone=dict(raw=s_raw, centred=s_cen),
                   retention={str(k): v for k, v in res.items()},
                   gate=dict(**GATE, measured=ret, verdict=verdict)),
              open("outputs/routeB/clip_phasor_retention.json", "w"), indent=1)
    print("wrote outputs/routeB/clip_phasor_retention.json")


if __name__ == "__main__":
    main()
