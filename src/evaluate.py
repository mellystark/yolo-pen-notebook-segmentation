from ultralytics import YOLO
from config import DATA_YAML, BEST_MODEL, IMGSZ, BATCH, DEVICE

def evaluate(split_name="val"):
    model = YOLO(str(BEST_MODEL))

    metrics = model.val(
        data=str(DATA_YAML),
        split=split_name,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        verbose=True,
        plots=True
    )

    print(f"\n--- {split_name.upper()} SONUÇLARI ---")
    print("BOX Precision :", round(metrics.box.mp, 4))
    print("BOX Recall    :", round(metrics.box.mr, 4))
    print("BOX mAP50     :", round(metrics.box.map50, 4))
    print("BOX mAP50-95  :", round(metrics.box.map, 4))

    print("MASK Precision:", round(metrics.seg.mp, 4))
    print("MASK Recall   :", round(metrics.seg.mr, 4))
    print("MASK mAP50    :", round(metrics.seg.map50, 4))
    print("MASK mAP50-95 :", round(metrics.seg.map, 4))

if __name__ == "__main__":
    evaluate("val")
    evaluate("test")