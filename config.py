"""Iris-Shield configuration."""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP model
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# BLIP model (captioning)
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# Target caption for BLIP attack
TARGET_CAPTION = "data redacted"

# Decoy description for CLIP attack
DECOY_TEXT = "a solid gray background with no people or identifiable features"

# Auto-tuning: starts with min_eps, increases until attack succeeds or max_eps reached
AUTO_MIN_EPS = 2.0 / 255.0
AUTO_MAX_EPS = 6.0 / 255.0
AUTO_EPS_STEP = 2.0 / 255.0
AUTO_STEPS_PER_ROUND = 300
AUTO_LR = 0.008
AUTO_CLIP_WEIGHT = 0.5
AUTO_BLIP_WEIGHT = 0.5

# Max image dimension (higher = better quality but slower)
MAX_IMAGE_SIZE = 768
