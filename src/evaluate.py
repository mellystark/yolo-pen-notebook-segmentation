from ultralytics import YOLO
from pathlib import Path

def get_next_version(base=r"C:\Users\Melike\Desktop\myFolders\DeepThink\yolo_seg_project\outputs\outputs"):
    i = 1.0
    while Path(f"{base}{i}").exists():
        i = round(i + 0.1, 1)
    return Path(f"{base}{i}")

def save_metrics(path, title, r):
    with open(path, "a") as f:
        f.write(f"\n--- {title} ---\n")
        f.write(f"BOX Precision : {r.box.mp}\n")
        f.write(f"BOX Recall    : {r.box.mr}\n")
        f.write(f"BOX mAP50     : {r.box.map50}\n")
        f.write(f"BOX mAP50-95  : {r.box.map}\n")
        f.write(f"MASK Precision: {r.seg.mp}\n")
        f.write(f"MASK Recall   : {r.seg.mr}\n")
        f.write(f"MASK mAP50    : {r.seg.map50}\n")
        f.write(f"MASK mAP50-95 : {r.seg.map}\n")

if __name__ == "__main__":
    model = YOLO(r"C:\Users\Melike\Desktop\myFolders\DeepThink\yolo_seg_project\models\trained\best.pt")

    save_dir = get_next_version()
    save_dir.mkdir(parents=True)

    val = model.val(data=r"C:\Users\Melike\Desktop\myFolders\DeepThink\yolo_seg_project\data\current_dataset\dataset\data.yaml", split="val", device=0, workers=0)
    test = model.val(data=r"C:\Users\Melike\Desktop\myFolders\DeepThink\yolo_seg_project\data\current_dataset\dataset\data.yaml", split="test", device=0, workers=0)

    save_metrics(save_dir / "metrics.txt", "VAL", val)
    save_metrics(save_dir / "metrics.txt", "TEST", test)