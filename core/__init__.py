"""Iris-Shield core — Anti-VLM attack pipeline."""
from .clip_attack import CLIPAttacker
from .blip_attack import BLIPAttacker
from .ensemble_cloaker import EnsembleCloaker, CloakResult
from .background_protector import BackgroundProtector
