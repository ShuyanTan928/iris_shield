"""
CLIP white-box attack.

Pushes image embedding AWAY from original content
and TOWARD a decoy text description.
"""

import torch
import open_clip
from loguru import logger
import config


class CLIPAttacker:
    """White-box adversarial attack against CLIP vision encoder."""

    def __init__(self):
        logger.info(f"Loading CLIP model: {config.CLIP_MODEL_NAME}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME,
            pretrained=config.CLIP_PRETRAINED,
            device=config.DEVICE,
        )
        self.tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
        self.model.eval()

        with torch.no_grad():
            tokens = self.tokenizer([config.DECOY_TEXT]).to(config.DEVICE)
            self.decoy_emb = self.model.encode_text(tokens)
            self.decoy_emb = self.decoy_emb / self.decoy_emb.norm(dim=-1, keepdim=True)

        logger.info("CLIP model ready.")

    def compute_loss(self, image_tensor: torch.Tensor, orig_emb: torch.Tensor) -> torch.Tensor:
        """Attack loss: push away from original, pull toward decoy."""
        adv_emb = self.model.encode_image(image_tensor)
        adv_emb = adv_emb / adv_emb.norm(dim=-1, keepdim=True)
        sim_orig = torch.cosine_similarity(adv_emb, orig_emb, dim=-1)
        sim_decoy = torch.cosine_similarity(adv_emb, self.decoy_emb, dim=-1)
        return sim_orig - sim_decoy

    def get_image_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract normalized CLIP image embedding."""
        with torch.no_grad():
            emb = self.model.encode_image(image_tensor)
            return emb / emb.norm(dim=-1, keepdim=True)

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Extract normalized CLIP text embedding."""
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(config.DEVICE)
            emb = self.model.encode_text(tokens)
            return emb / emb.norm(dim=-1, keepdim=True)

    def identify(self, image_tensor: torch.Tensor, candidate_names: list[str]) -> list[tuple[str, float]]:
        """Check how well image matches each candidate name."""
        img_emb = self.get_image_embedding(image_tensor)
        results = []
        for name in candidate_names:
            text_emb = self.get_text_embedding(f"a photo of {name}")
            sim = torch.cosine_similarity(img_emb, text_emb).item()
            results.append((name, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)
