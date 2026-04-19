"""
clip_encoder.py
===============
Wraps the HuggingFace CLIP model to extract 512-D visual and text
features for SSP-SLAM semantic encoding.
"""

import numpy as np

class CLIPFeatureExtractor:
    """Extracts aligned image and text features using OpenAI's CLIP.
    
    The returned vectors are 512-dimensional floats, and can be fed directly 
    into an ImageFeatureEncoder or bound to an SPSpace.
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model '{model_name}' on {self.device}...")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # We output a fixed 512 dimension for 'clip-vit-base-patch32'
        self.feat_dim = self.model.config.projection_dim
        
    def encode_image(self, images) -> np.ndarray:
        """Encode one or more images into continuous feature vectors.
        
        Parameters
        ----------
        images : PIL.Image or list of PIL.Image, or np.ndarray
            The image(s) to encode. If np.ndarray, it assumes RGB HWC format.
            
        Returns
        -------
        np.ndarray
            Shape (n, feat_dim)
        """
        import torch
        
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize the features over the feature dimension, as is standard in CLIP
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy()
        
    def encode_text(self, texts) -> np.ndarray:
        """Encode one or more text queries into continuous feature vectors.
        
        Parameters
        ----------
        texts : str or list of str
            The text snippet(s) to encode (e.g. "a red box").
            
        Returns
        -------
        np.ndarray
            Shape (n, feat_dim)
        """
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy()

