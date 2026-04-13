# Iris-Shield

**Anti-VLM Privacy Protection** — Make your photos unreadable to AI vision models while looking normal to humans.

## What it does

| Feature | How it works |
|---------|-------------|
| **AI Caption Attack** | Forces AI captioning models (BLIP) to output "data redacted" instead of describing the actual image content |
| **CLIP Embedding Attack** | Pushes the image's visual embedding away from its real content, breaking AI search and auto-tagging |
| **License Plate Protection** | Detects plates with a dedicated YOLO model, reads text with OCR, replaces with random characters |
| **Sign Blurring** | Detects street signs and applies gaussian blur |

All processing runs **locally on your machine**. No data is uploaded anywhere.

## Requirements

- Python 3.10-3.12
- NVIDIA GPU with CUDA (recommended, 6GB+ VRAM)
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

### Windows (CMD)

```cmd
:: Install uv if you don't have it
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

:: Clone and setup
git clone https://github.com/YOUR_USERNAME/iris-shield.git
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
git clone https://github.com/YOUR_USERNAME/iris-shield.git
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

## License

MIT
