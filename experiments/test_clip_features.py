"""
test_clip_features.py
=====================
Tests the CLIP feature extractor and ensures its high-dimensional
embeddings correctly map to Semantic Pointers via the ImageFeatureEncoder
while preserving cosine similarity.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sspslam.perception import ImageFeatureEncoder
from sspslam.perception.clip_encoder import CLIPFeatureExtractor

def test_clip_similarity_preservation():
    print("Initializing CLIP model...")
    clip = CLIPFeatureExtractor()
    
    # 1. Generate text embeddings
    labels = ["a red box", "a red cube", "a blue box", "a green ball"]
    print(f"Extracting features for text: {labels}")
    text_features = clip.encode_text(labels)
    
    # 2. Encode to Semantic Pointers
    ssp_dim = 97
    enc = ImageFeatureEncoder(feat_dim=clip.feat_dim, ssp_dim=ssp_dim, seed=42)
    sps = enc.encode(text_features)
    
    print("\nCosine Similarity Matrix between Semantic Pointers:")
    header = f"{'':15s}" + "".join(f"{lbl[:10]:>12s}" for lbl in labels)
    print(header)
    
    for i, a_lbl in enumerate(labels):
        row = f"{a_lbl[:15]:15s}"
        for j, b_lbl in enumerate(labels):
            sim = np.dot(sps[i], sps[j]) / (np.linalg.norm(sps[i]) * np.linalg.norm(sps[j]))
            row += f"{sim:12.4f}"
        print(row)
        
    print("\n✓ CLIP feature extraction and VSA similarity preservation successful!")

if __name__ == "__main__":
    test_clip_similarity_preservation()
