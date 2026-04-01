from ultralytics import YOLO

if __name__ == "__main__":
    YOLO("yolov8n-seg.pt").train(
        data=r"C:\Users\Melike\Desktop\myFolders\DeepThink\yolo_seg_project\data\current_dataset\dataset\data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=0
    )