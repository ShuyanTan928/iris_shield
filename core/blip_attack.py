"""
BLIP targeted caption attack.

Forces BLIP to generate a specific target caption using teacher forcing.
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from loguru import logger
import config


class BLIPAttacker:
    """Targeted caption attack against BLIP vision encoder."""

    def __init__(self):
        logger.info(f"Loading BLIP model: {config.BLIP_MODEL_NAME}...")
        self.processor = BlipProcessor.from_pretrained(config.BLIP_MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(
            config.BLIP_MODEL_NAME,
            torch_dtype=torch.float16,
        ).to(config.DEVICE)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.target_text = config.TARGET_CAPTION
        self.target_ids = self.processor.tokenizer(
            self.target_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids.to(config.DEVICE)

        logger.info(f"BLIP ready. Target caption: \"{self.target_text}\"")

    def compute_loss(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Targeted caption loss via teacher forcing."""
        batch_size = pixel_values.shape[0]
        target_ids = self.target_ids.expand(batch_size, -1)
        outputs = self.model(
            pixel_values=pixel_values.half(),
            input_ids=target_ids,
            labels=target_ids,
        )
        return outputs.loss.float()

    def generate_caption(self, image) -> str:
        """Generate caption for verification."""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(config.DEVICE)
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            output_ids = self.model.generate(**inputs, max_new_tokens=50)
            return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
