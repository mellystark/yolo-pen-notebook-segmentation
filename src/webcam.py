import cv2
from ultralytics import YOLO
from config import BEST_MODEL, DEVICE, CONF, IOU

def run_webcam():
    model = YOLO(str(BEST_MODEL))
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("HATA: Kamera açılamadı.")
        return

    print("Kamera açıldı. Çıkmak için q tuşuna bas.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("HATA: Kameradan görüntü alınamadı.")
            break

        results = model.predict(
            source=frame,
            conf=CONF,
            iou=IOU,
            imgsz=640,
            device=DEVICE,
            verbose=False
        )

        annotated = results[0].plot()
        cv2.imshow("YOLO Segmentation Webcam", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()