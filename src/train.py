import shutil
from ultralytics import YOLO
from config import DATA_YAML, PRETRAINED_MODEL, RUNS_DIR, TRAINED_DIR, EPOCHS, IMGSZ, BATCH, DEVICE

def train_model():
    model_path = str(PRETRAINED_MODEL) if PRETRAINED_MODEL.exists() else "yolov8n-seg.pt"
    model = YOLO(model_path)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=0,
        project=str(RUNS_DIR),
        name="seg_train",
        pretrained=True,
        verbose=True
    )

    best_src = RUNS_DIR / "seg_train" / "weights" / "best.pt"
    last_src = RUNS_DIR / "seg_train" / "weights" / "last.pt"

    TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    if best_src.exists():
        shutil.copy2(best_src, TRAINED_DIR / "best.pt")
    if last_src.exists():
        shutil.copy2(last_src, TRAINED_DIR / "last.pt")

    print("\nEğitim tamamlandı.")
    print("Best model:", TRAINED_DIR / "best.pt")
    print("Last model:", TRAINED_DIR / "last.pt")

if __name__ == "__main__":
    train_model()