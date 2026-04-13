"""
Ensemble cloaker with auto-tuning epsilon.

Starts with minimal perturbation and increases only if needed.
This ensures the highest possible image quality while still
achieving effective protection.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from loguru import logger
from dataclasses import dataclass, field
import config
from .clip_attack import CLIPAttacker
from .blip_attack import BLIPAttacker
from .background_protector import BackgroundProtector


@dataclass
class CloakResult:
    """Output of the cloaking process."""
    cloaked_image: np.ndarray       # Protected image (RGB uint8 HWC)
    caption_before: str             # BLIP caption of original
    caption_after: str              # BLIP caption of protected
    plates_replaced: int            # Number of plates replaced
    signs_blurred: int              # Number of signs blurred
    identity_before: list           # CLIP identity matches before
    identity_after: list            # CLIP identity matches after
    eps_used: float                 # Final epsilon that was used
    steps_total: int                # Total optimization steps run
    rounds_used: int                # Number of auto-tune rounds


# Common names for identity recognition testing
IDENTITY_CANDIDATES = [
    "Elon Musk", "Jeff Bezos", "Mark Zuckerberg", "Bill Gates",
    "Barack Obama", "Donald Trump", "Taylor Swift", "Beyonce",
    "Leonardo DiCaprio", "Tom Cruise", "an unknown person",
]


class EnsembleCloaker:
    """Joint CLIP + BLIP attack with auto-tuning epsilon."""

    def __init__(self):
        logger.info("Initializing ensemble cloaker (CLIP + BLIP)...")
        self.clip = CLIPAttacker()
        self.blip = BLIPAttacker()
        self.bg = BackgroundProtector()
        self.device = config.DEVICE
        logger.info("Ensemble cloaker ready.")

    def _run_attack_round(
        self,
        img_tensor: torch.Tensor,
        delta: torch.Tensor,
        clip_emb_orig: torch.Tensor,
        eps: float,
        steps: int,
        lr: float,
        clip_weight: float,
        blip_weight: float,
    ) -> torch.Tensor:
        """Run one round of Adam optimization at a given epsilon."""

        optimizer = torch.optim.Adam([delta], lr=lr)

        # CLIP normalization constants
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1).to(self.device)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1).to(self.device)

        # BLIP normalization constants (same as CLIP for BLIP-base)
        blip_mean = clip_mean
        blip_std = clip_std

        # Get CLIP input size from preprocessing
        clip_size = 224  # ViT-B/32 default
        blip_size = 384  # BLIP-base default

        for step in range(steps):
            optimizer.zero_grad()
            img_adv = torch.clamp(img_tensor + delta, 0, 1)

            # CLIP loss
            clip_adv = F.interpolate(img_adv, size=(clip_size, clip_size), mode="bilinear", align_corners=False)
            clip_adv_norm = (clip_adv - clip_mean) / clip_std
            clip_loss = self.clip.compute_loss(clip_adv_norm, clip_emb_orig)

            # BLIP loss
            blip_adv = F.interpolate(img_adv, size=(blip_size, blip_size), mode="bilinear", align_corners=False)
            blip_adv_norm = (blip_adv - blip_mean) / blip_std
            blip_loss = self.blip.compute_loss(blip_adv_norm)

            total_loss = clip_weight * clip_loss + blip_weight * blip_loss
            total_loss.backward()
            optimizer.step()

            # Project onto L-inf epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -eps, eps)
                delta.data = torch.clamp(img_tensor + delta.data, 0, 1) - img_tensor

            if (step + 1) % 50 == 0:
                logger.debug(
                    f"    Step {step+1}/{steps} | "
                    f"clip={clip_loss.item():.4f} blip={blip_loss.item():.4f}"
                )

        return delta

    def cloak(self, image: Image.Image, progress_callback=None) -> CloakResult:
        """
        Auto-tuning adversarial attack.

        Starts with minimum epsilon for best image quality.
        Checks if BLIP caption changed to target.
        If not, increases epsilon and runs another round.
        Repeats until success or max epsilon reached.
        """

        # Step 1: Background protection (plates + signs) BEFORE VLM attack
        if progress_callback:
            progress_callback(0.05, "Detecting license plates and signs...")
        import cv2
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_bgr, bg_stats = self.bg.protect(img_bgr)
        if progress_callback:
            progress_callback(0.10, f"Background: {bg_stats['plates_replaced']} plates replaced, {bg_stats['signs_blurred']} signs blurred")
        image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # Step 2: Preprocess cleaned image to tensor [0, 1]
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Get original CLIP embedding
        clip_size = 224
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1).to(self.device)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1).to(self.device)

        with torch.no_grad():
            clip_input = F.interpolate(img_tensor, size=(clip_size, clip_size), mode="bilinear", align_corners=False)
            clip_input_norm = (clip_input - clip_mean) / clip_std
            clip_emb_orig = self.clip.get_image_embedding(clip_input_norm)

        # Get original captions and identity
        caption_before = self.blip.generate_caption(image)
        identity_before = self.clip.identify(clip_input_norm, IDENTITY_CANDIDATES)

        # Initialize perturbation
        delta = torch.zeros_like(img_tensor, requires_grad=True)

        # Auto-tuning loop
        eps = config.AUTO_MIN_EPS
        total_steps = 0
        rounds = 0

        while eps <= config.AUTO_MAX_EPS:
            rounds += 1
            logger.info(f"  Round {rounds}: eps={eps*255:.1f}/255, steps={config.AUTO_STEPS_PER_ROUND}")

            if progress_callback:
                progress_pct = min(0.1 + 0.7 * (eps - config.AUTO_MIN_EPS) / max(config.AUTO_MAX_EPS - config.AUTO_MIN_EPS, 1e-6), 0.85)
                progress_callback(progress_pct, f"Optimizing (round {rounds}, budget={eps*255:.0f}/255)...")

            delta = self._run_attack_round(
                img_tensor, delta, clip_emb_orig,
                eps=eps,
                steps=config.AUTO_STEPS_PER_ROUND,
                lr=config.AUTO_LR,
                clip_weight=config.AUTO_CLIP_WEIGHT,
                blip_weight=config.AUTO_BLIP_WEIGHT,
            )
            total_steps += config.AUTO_STEPS_PER_ROUND

            # Check if BLIP caption changed
            with torch.no_grad():
                img_adv = torch.clamp(img_tensor + delta, 0, 1)
                adv_np = (img_adv.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                adv_pil = Image.fromarray(adv_np)

            caption_after = self.blip.generate_caption(adv_pil)

            # Success check: did the caption change to our target?
            # Check if caption contains any protection keywords
            protection_keywords = ['redacted', 'protected', 'blocked', 'privacy', 'no visible', 'no content', 'data_redacted']
            caption_lower = caption_after.lower()
            overlap = 1.0 if any(kw in caption_lower for kw in protection_keywords) else 0.0

            logger.info(f"    Caption: \"{caption_after}\" (target overlap: {overlap:.0%})")

            if overlap >= 0.5:
                logger.info(f"  Attack successful at eps={eps*255:.1f}/255 after {total_steps} steps")
                break

            # Increase epsilon for next round
            eps += config.AUTO_EPS_STEP

        # Final identity check
        with torch.no_grad():
            clip_adv = F.interpolate(img_adv, size=(clip_size, clip_size), mode="bilinear", align_corners=False)
            clip_adv_norm = (clip_adv - clip_mean) / clip_std
            identity_after = self.clip.identify(clip_adv_norm, IDENTITY_CANDIDATES)

        logger.info(
            f"Attack complete. eps={eps*255:.1f}/255, {total_steps} steps, {rounds} rounds | "
            f"Caption: \"{caption_before}\" -> \"{caption_after}\" | "
            f"Identity: {identity_before[0][0]}({identity_before[0][1]:.2f}) -> {identity_after[0][0]}({identity_after[0][1]:.2f})"
        )

        return CloakResult(
            cloaked_image=adv_np,
            caption_before=caption_before,
            caption_after=caption_after,
            plates_replaced=bg_stats.get("plates_replaced", 0),
            signs_blurred=bg_stats.get("signs_blurred", 0),
            identity_before=identity_before[:3],
            identity_after=identity_after[:3],
            eps_used=eps,
            steps_total=total_steps,
            rounds_used=rounds,
        )
