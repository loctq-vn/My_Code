import json, os, re
PATH = r"D:\Truong_pipeline\My_Code\src\making_ocr.ipynb"

def C(s):
    return {"cell_type": "code", "execution_count": None, "id": "c" + str(abs(hash(s)) % 100000),
            "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}

def M(s):
    return {"cell_type": "markdown", "id": "m" + str(abs(hash(s)) % 100000),
            "metadata": {}, "source": s.splitlines(keepends=True)}

cells = []

cells.append(M(r"""# Making OCR for L25 (GPU-optimized batch)

EasyOCR voi batch processing va GPU toi da. Xuat CSV (frame_id + text) ra `L25_data/ocr`."""))

cells.append(C(r"""import gc
import csv
import json
import os
import re
from pathlib import Path

import cv2
import torch
from easyocr import Reader
from tqdm.auto import tqdm

BASE_DIR = Path.cwd().parent
KEYFRAME_DIR = BASE_DIR / "L25_data" / "keyframes"
OCR_DIR = BASE_DIR / "L25_data" / "ocr"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OCR_LANGUAGES = ["vi", "en"]
MIN_CONFIDENCE = 0.30
SKIP_EXISTING_OUTPUT = True

# GPU / batch config - toi da hoa
BATCH_SIZE = 32
NUM_READER_WORKERS = min(8, os.cpu_count() or 4)""", None))

cells.append(C(r"""if not KEYFRAME_DIR.exists():
    raise FileNotFoundError(f"Khong tim thay thu muc keyframe: {KEYFRAME_DIR}")


def natural_sort_key(path: Path):
    # Sort video/frame dirs numerically, e.g. V2 before V10
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


video_dirs = sorted(
    (p for p in KEYFRAME_DIR.iterdir() if p.is_dir()), key=natural_sort_key
)
video_ids = [d.name for d in video_dirs]

OCR_DIR.mkdir(parents=True, exist_ok=True)
print(f"Tong so video: {len(video_dirs)}")
print(f"Output OCR: {OCR_DIR}")
print(f"CUDA: {torch.cuda.is_available()} | Batch: {BATCH_SIZE} | Workers: {NUM_READER_WORKERS}")""", None))

cells.append(C(r"""device = "cuda" if torch.cuda.is_available() else "cpu"

# GPU optimization - toi da hoa hieu nang
if device == "cuda":
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    try:
        torch.cuda.set_per_process_memory_fraction(1.0)
    except RuntimeError:
        pass  # already set

reader = Reader(OCR_LANGUAGES, gpu=(device == "cuda"))
print(f"Using device: {device}")
print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")""", None))

with open(r"D:\Truong_pipeline\My_Code\src\_cells_part1.json", "w") as f:
    json.dump(cells, f, ensure_ascii=False)
print(f"Part 1 done: {len(cells)} cells")