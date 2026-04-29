# Iris-Shield

**Anti-VLM Privacy Protection** — Make your photos unreadable to AI vision models while looking normal to humans.

## What it does

| Feature | How it works |
|---------|-------------|
| **AI Caption Attack** | When crawlers scrape your photos, they use AI captioning models to auto-label your images for training data. This attack forces those models to output "data redacted" instead of the real description, poisoning any dataset that includes your photos |
| **CLIP Embedding Attack** | AI systems use CLIP embeddings to search, classify, and tag images. This attack corrupts the embedding so your photo cannot be found via reverse image search or correctly categorized by automated pipelines |
| **License Plate Protection** | Detects plates with a dedicated YOLO model, reads text with OCR, replaces with random characters |
| **Sign Blurring** | Detects street signs and applies gaussian blur |

All processing runs **locally on your machine**. No data is uploaded anywhere.

## Requirements

- Python 3.10-3.12
- NVIDIA GPU with CUDA (optional, speeds up processing 10-20x)
- CPU-only mode fully supported
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

### Windows (CMD)

```cmd
:: Install uv if you don't have it
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

:: Clone and setup
git clone https://github.com/ShuyanTan928/iris_shield.git
cd iris-shield
uv sync

:: Download YOLO plate detection model (one time)
uv run python -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('models/plate_detect', exist_ok=True); hf_hub_download(repo_id='morsetechlab/yolov11-license-plate-detection', filename='license-plate-finetune-v1n.pt', local_dir='models/plate_detect')"
uv run python -c "from ultralytics import YOLO; import shutil, os; YOLO('yolo11n.pt'); os.makedirs('models/yolov11n', exist_ok=True); shutil.move('yolo11n.pt', 'models/yolov11n/yolov11n.pt')"

:: Launch web UI
uv run streamlit run main.py --server.port 8501
```

### macOS / Linux

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/ShuyanTan928/iris_shield.git
cd iris-shield
uv sync

# Download YOLO plate detection model (one time)
uv run python -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('models/plate_detect', exist_ok=True); hf_hub_download(repo_id='morsetechlab/yolov11-license-plate-detection', filename='license-plate-finetune-v1n.pt', local_dir='models/plate_detect')"
uv run python -c "from ultralytics import YOLO; import shutil, os; YOLO('yolo11n.pt'); os.makedirs('models/yolov11n', exist_ok=True); shutil.move('yolo11n.pt', 'models/yolov11n/yolov11n.pt')"

# Launch web UI
uv run streamlit run main.py --server.port 8501
```

First launch downloads CLIP (~400MB) and BLIP (~400MB) automatically. Subsequent launches are instant.

Open **http://localhost:8501** in your browser.

## CUDA GPU Setup (Optional but Recommended)

For NVIDIA GPU acceleration, install the CUDA version of PyTorch:

```cmd
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```cmd
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Local Testing (without web UI)

### Test VLM attack

```cmd
uv run python -c "from PIL import Image; from core.ensemble_cloaker import EnsembleCloaker; c = EnsembleCloaker(); img = Image.open('examples/face_test.jpeg'); r = c.cloak(img); Image.fromarray(r.cloaked_image).save('output.png'); print('Before:', r.caption_before); print('After:', r.caption_after)"
```

### Test plate detection

```cmd
uv run python -c "import cv2; from core.background_protector import BackgroundProtector; bp = BackgroundProtector(); img = cv2.imread('examples/car_test.jpeg'); r, s = bp.protect(img); print(s); cv2.imwrite('output_plate.jpeg', r)"
```

## Project Structure

```
iris-shield/
├── main.py                     # Streamlit entry point
├── config.py                   # All configuration in one place
├── pyproject.toml              # Dependencies (managed by uv)
├── core/
│   ├── clip_attack.py          # CLIP white-box adversarial attack
│   ├── blip_attack.py          # BLIP targeted caption attack
│   ├── ensemble_cloaker.py     # Joint CLIP+BLIP attack with auto-tuning
│   └── background_protector.py # License plate + sign detection/replacement
├── ui/
│   ├── dashboard.py            # Streamlit web interface
│   └── styles.py               # Custom CSS
├── utils/
│   └── image_utils.py          # Image I/O helpers
├── examples/                   # Test images
└── models/                     # Downloaded model weights (gitignored)
    ├── yolov11n/               # General object detection
    └── plate_detect/           # License plate detection
```

## How it works

### VLM Attack (CLIP + BLIP)

The system applies adversarial perturbations to the image using Adam optimizer:

1. **CLIP attack**: Minimizes similarity between the protected image and its original CLIP embedding, while maximizing similarity to a decoy description
2. **BLIP attack**: Uses teacher forcing to minimize cross-entropy loss against a target caption ("data redacted")
3. **Auto-tuning**: Starts with minimal perturbation (eps=2/255), checks if BLIP caption changed, increases only if needed

### Background Protection

1. **Plate detection**: Dedicated YOLOv11 model fine-tuned on 10K+ plate images
2. **Text replacement**: EasyOCR reads plate text, each character is replaced with a random one of the same type
3. **Sign detection**: General YOLOv11 detects signs, applies gaussian blur



## Open-Source Components Used

| Component | Source | What it does in our project |
|-----------|--------|---------------------------|
| [OpenAI CLIP (ViT-B/32)](https://github.com/mlfoundations/open_clip) | `open-clip-torch` pip package | Pre-trained vision-language encoder. We use it as a white-box attack target — no modifications to the model itself |
| [BLIP (blip-image-captioning-base)](https://huggingface.co/Salesforce/blip-image-captioning-base) | Salesforce via HuggingFace `transformers` | Pre-trained image captioning model. Used as a white-box attack target — no modifications to the model itself |
| [YOLOv11n](https://github.com/ultralytics/ultralytics) | Ultralytics `ultralytics` pip package | General object detection (street signs). Used as-is for inference only |
| [YOLOv11 License Plate](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) | morsetechlab via HuggingFace | Fine-tuned plate detection model. Used as-is for inference only |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | `easyocr` pip package | Reads license plate text for character replacement. Used as-is |
| [Streamlit](https://streamlit.io/) | `streamlit` pip package | Web UI framework. Used as-is |
| [Plotly](https://plotly.com/) | `plotly` pip package | Chart rendering in the dashboard. Used as-is |

## What We Built (Original Code)

All files in `core/`, `ui/`, and `utils/` are **original implementations**. Specifically:

### Novel attack pipeline (`core/`)
- **`ensemble_cloaker.py`** — Joint CLIP + BLIP adversarial attack with auto-tuning epsilon. This is the core contribution: an Adam-optimized adversarial perturbation that simultaneously (1) misdirects CLIP embeddings away from the original image content toward a decoy description, and (2) hijacks BLIP caption generation via teacher-forced cross-entropy optimization to output a controlled target string. The auto-tuning loop starts with minimal perturbation and increases only when needed, preserving image quality.
- **`clip_attack.py`** — White-box adversarial loss computation against CLIP: pushes image embeddings away from original content and toward a decoy text embedding. Includes zero-shot identity recognition testing.
- **`blip_attack.py`** — Targeted caption attack using teacher forcing: minimizes cross-entropy loss between BLIP's output distribution and a target token sequence, forcing the model to generate a specific caption.
- **`background_protector.py`** — Multi-model detection pipeline: uses a dedicated plate-detection YOLO model for license plates + general YOLO for signs, then applies OCR-based character replacement (reads each character, replaces with random same-type character) for plates and gaussian blur for signs.

### User interface (`ui/`)
- **`dashboard.py`** — Complete Streamlit web application with auto-tuning progress display, before/after comparison, AI caption test, CLIP identity recognition test, and background protection statistics.
- **`styles.py`** — Custom CSS theming.

### Key technical decisions
- **Adam optimizer instead of PGD**: Standard PGD (sign gradient) converges poorly against L2-normalized embeddings. We use Adam with L-inf projection, which provides adaptive learning rates and momentum for much better convergence.
- **Auto-tuning epsilon**: Instead of a fixed perturbation budget, the system starts at eps=2/255 and increases in steps of 2/255 only if the BLIP caption hasn't changed, capping at 6/255. This minimizes visual distortion.
- **Dual-model ensemble attack**: Attacking CLIP and BLIP jointly is more robust than attacking either alone, because they use different architectures and the perturbation must generalize across both.
- **Teacher forcing for caption control**: Rather than just disrupting BLIP features, we directly optimize the cross-entropy loss against target tokens, giving precise control over the generated caption.

## License

MIT
