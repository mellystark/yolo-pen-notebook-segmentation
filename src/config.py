from pathlib import Path

BASE_DIR = Path(r"C:\Users\Melike\Desktop\myFolders\DeepThink\yolo_seg_project")

DATA_YAML = BASE_DIR / "data" / "current_dataset" / "dataset" / "data.yaml"
BEST_MODEL = BASE_DIR / "models" / "trained" / "best.pt"

MODEL_NAME = "yolov8n-seg.pt"
DEVICE = 0
EPOCHS = 100
IMGSZ = 640
BATCH = 8

CONF = 0.80
IOU = 0.45