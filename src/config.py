from pathlib import Path

BASE_DIR = Path(r"C:\Users\Melike\Desktop\myFolders\DeepThink\yolo_seg_project")

DATA_YAML = BASE_DIR / "data" / "current_dataset" / "dataset" / "data.yaml"

MODELS_DIR = BASE_DIR / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
TRAINED_DIR = MODELS_DIR / "trained"

RUNS_DIR = BASE_DIR / "runs"
OUTPUTS_DIR = BASE_DIR / "outputs"

PRETRAINED_MODEL = PRETRAINED_DIR / "yolov8n-seg.pt"
BEST_MODEL = TRAINED_DIR / "best.pt"
LAST_MODEL = TRAINED_DIR / "last.pt"

EPOCHS = 100
IMGSZ = 640
BATCH = 8
DEVICE = 0

CONF = 0.80
IOU = 0.45