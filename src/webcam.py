from ultralytics import YOLO

if __name__ == "__main__":
    YOLO(r"C:\Users\Melike\Desktop\myFolders\DeepThink\yolo_seg_project\models\trained\best.pt").predict(
        source=0,
        show=True,
        conf=0.80,
        device=0
    )