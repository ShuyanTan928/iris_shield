"""
Background privacy protection: license plates and street signs.

Plates: EasyOCR locates each character individually, then each character
        is erased (filled with local background color) and redrawn with
        a random replacement character at the exact same position and size.
        Only the main plate number is replaced — decorative text like
        state names and small labels are left alone for realism.
Signs: gaussian blur.
"""

import random
import string
import numpy as np
import cv2
from pathlib import Path
from loguru import logger

YOLO_GENERAL = Path(__file__).resolve().parent.parent / "models" / "yolov11n" / "yolov11n.pt"
YOLO_PLATE = Path(__file__).resolve().parent.parent / "models" / "plate_detect" / "license-plate-finetune-v1n.pt"
SIGN_CLASSES = {11}

_ocr_reader = None

def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=True)
        logger.info("EasyOCR reader initialized.")
    return _ocr_reader


class BackgroundProtector:
    """Detect plates, OCR each character, replace individually."""

    def __init__(self):
        from ultralytics import YOLO
        logger.info(f"Loading general YOLO from {YOLO_GENERAL}...")
        self._general = YOLO(str(YOLO_GENERAL))
        logger.info(f"Loading plate detector from {YOLO_PLATE}...")
        self._plate = YOLO(str(YOLO_PLATE))
        logger.info("Background protector ready.")

    def protect(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Process BGR image. Returns (processed, stats)."""
        img = image.copy()
        stats = {"plates_replaced": 0, "signs_blurred": 0}

        plate_results = self._plate.predict(img, conf=0.25, iou=0.45, verbose=False)
        for r in plate_results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                pad = 2
                y1, y2 = max(0, y1 - pad), min(img.shape[0], y2 + pad)
                x1, x2 = max(0, x1 - pad), min(img.shape[1], x2 + pad)
                h, w = y2 - y1, x2 - x1
                if h > 10 and w > 20:
                    roi = img[y1:y2, x1:x2]
                    img[y1:y2, x1:x2] = self._replace_characters(roi)
                    stats["plates_replaced"] += 1

        general_results = self._general.predict(img, conf=0.35, iou=0.45, verbose=False)
        for r in general_results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in SIGN_CLASSES:
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox
                    roi = img[y1:y2, x1:x2]
                    if roi.size > 0:
                        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)
                        stats["signs_blurred"] += 1

        logger.info(f"Background protection: {stats}")
        return img, stats

    def _replace_characters(self, roi: np.ndarray) -> np.ndarray:
        """
        Use EasyOCR to find each text region on the plate,
        then for each detected text block:
          1. Sample the background color around the text
          2. Sample the text color from the text pixels
          3. Erase the original text (paint over with background)
          4. Draw new random characters in the same position with text color
        Only replaces text blocks that look like plate numbers
        (at least 3 chars, contains digits).
        """
        h, w = roi.shape[:2]
        result = roi.copy()

        try:
            reader = _get_ocr()
            # detail=1 gives per-character bounding boxes in some cases
            ocr_results = reader.readtext(roi, paragraph=False)

            if not ocr_results:
                return result

            for (bbox_points, text, conf) in ocr_results:
                text = text.strip()
                if conf < 0.15:
                    continue

                # Only replace text that looks like a plate number
                # (contains digits and is at least 3 chars)
                has_digit = any(c.isdigit() for c in text)
                if not has_digit or len(text) < 3:
                    continue

                # Get tight bounding box
                pts = np.array(bbox_points).astype(int)
                tx1 = max(0, pts[:, 0].min() - 2)
                ty1 = max(0, pts[:, 1].min() - 2)
                tx2 = min(w, pts[:, 0].max() + 2)
                ty2 = min(h, pts[:, 1].max() + 2)
                tw = tx2 - tx1
                th = ty2 - ty1

                if tw < 8 or th < 8:
                    continue

                # Analyze colors in this text region
                text_roi = roi[ty1:ty2, tx1:tx2]
                gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
                median_val = np.median(gray)

                bg_pixels = text_roi[gray > median_val]
                text_pixels = text_roi[gray <= median_val]

                if len(bg_pixels) > 0:
                    bg_color = bg_pixels.mean(axis=0).astype(int).tolist()
                else:
                    bg_color = [200, 200, 200]

                if len(text_pixels) > 0:
                    text_color = text_pixels.mean(axis=0).astype(int).tolist()
                else:
                    text_color = [0, 0, 0]

                # Generate replacement text
                fake = self._randomize_text(text)

                # Step 1: Erase original text by painting background color
                # Use a slightly larger area and blend edges
                erase_mask = np.zeros((th, tw), dtype=np.uint8)
                cv2.rectangle(erase_mask, (2, 2), (tw - 3, th - 3), 255, -1)
                erase_mask = cv2.GaussianBlur(erase_mask, (5, 5), 2)

                for c in range(3):
                    region = result[ty1:ty2, tx1:tx2, c].astype(float)
                    blend = erase_mask.astype(float) / 255.0
                    region = region * (1 - blend) + bg_color[c] * blend
                    result[ty1:ty2, tx1:tx2, c] = region.astype(np.uint8)

                # Step 2: Draw replacement text
                # Use DUPLEX font (thicker, more like plate font)
                font = cv2.FONT_HERSHEY_DUPLEX
                thickness = max(1, th // 10)

                # Find the right font scale
                scale = 0.3
                for s_int in range(3, 50):
                    s = s_int * 0.1
                    ts_size = cv2.getTextSize(fake, font, s, thickness)[0]
                    if ts_size[0] > tw * 0.9 or ts_size[1] > th * 0.85:
                        break
                    scale = s

                ts_size = cv2.getTextSize(fake, font, scale, thickness)[0]
                text_x = tx1 + (tw - ts_size[0]) // 2
                text_y = ty1 + (th + ts_size[1]) // 2

                cv2.putText(result, fake, (text_x, text_y), font, scale,
                            tuple(text_color), thickness, cv2.LINE_AA)

                logger.debug(f"  Plate char: '{text}' -> '{fake}' at [{tx1},{ty1},{tx2},{ty2}]")

        except Exception as e:
            logger.warning(f"  OCR replacement failed ({e}), using blur")
            return cv2.GaussianBlur(roi, (25, 25), 10)

        return result

    @staticmethod
    def _randomize_text(text: str) -> str:
        """Replace each char with a random one of the same type."""
        result = []
        for ch in text:
            if ch.isdigit():
                result.append(random.choice([d for d in string.digits if d != ch]))
            elif ch.isalpha():
                pool = string.ascii_uppercase if ch.isupper() else string.ascii_lowercase
                result.append(random.choice([c for c in pool if c != ch]))
            else:
                result.append(ch)
        return "".join(result)
