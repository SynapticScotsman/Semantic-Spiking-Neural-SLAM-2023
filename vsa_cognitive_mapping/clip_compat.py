"""Version-agnostic access to CLIP's projected features.

Older transformers return a plain tensor from CLIPModel.get_text_features /
get_image_features; newer versions (Colab) return a ModelOutput wrapper.
These helpers always return the projected embedding tensor, so the
similarity math is byte-identical across versions — and therefore between
the local machine and Colab runs.
"""
from __future__ import annotations

import torch


def text_features(model, tok):
    out = model.get_text_features(**tok)
    if torch.is_tensor(out):
        return out
    v = getattr(out, "text_embeds", None)
    if v is not None and torch.is_tensor(v):
        return v
    # raw text_model output: apply the projection ourselves
    return model.text_projection(out.pooler_output)


def image_features(model, pix):
    out = model.get_image_features(**pix)
    if torch.is_tensor(out):
        return out
    v = getattr(out, "image_embeds", None)
    if v is not None and torch.is_tensor(v):
        return v
    return model.visual_projection(out.pooler_output)
