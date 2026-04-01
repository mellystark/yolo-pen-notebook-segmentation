````md
# 🎯 YOLOv8 Instance Segmentation: Pen & Notebook Demo

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-red)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.33-00FFFF)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)
![Task](https://img.shields.io/badge/Task-Instance%20Segmentation-purple)

A simple **YOLOv8 instance segmentation** project for detecting and segmenting:

- ✏️ `kalem1`
- ✏️ `kalem2`
- 📓 `defter`

The project includes:
- dataset structure
- training script
- evaluation script
- real-time webcam segmentation
- reproducible environment notes

---

# 📁 Project Structure

```text
yolo-seg-notebook-project/
│
├─ data/
│  └─ dataset/
│     ├─ data.yaml
│     ├─ train/
│     ├─ valid/
│     └─ test/
│
├─ models/
│  ├─ pretrained/
│  └─ trained/
│
├─ src/
│  ├─ config.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ webcam.py
│
├─ outputs/
├─ runs/
├─ requirements.txt
├─ .gitignore
└─ README.md
````

---

# 🧠 Model Information

This project was trained with:

* `yolov8n-seg.pt`

That means:

* `n` = lightweight / nano variant
* good for fast experiments and real-time webcam demos

---

# 🖥️ Tested Environment

This project was tested with:

* Python `3.12`
* PyTorch `2.11.0+cu128`
* Ultralytics `8.4.33`
* CUDA `12.8`
* GPU: NVIDIA GeForce RTX 3060 Laptop GPU

---

# ✅ Version Check

Before running the project, check your environment:

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
python -c "import ultralytics; print(ultralytics.__version__)"
```

Expected:

* Python 3.12.x
* CUDA available = `True`
* PyTorch CUDA build active

---

# 🚀 Installation

## Option 1 — If Python is already installed

Create virtual environment:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch (CUDA 12.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

## Option 2 — If Python is NOT installed

1. Install Python 3.12
2. Re-open terminal
3. Then run:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

# 📦 Dataset

The dataset is stored in YOLO segmentation format.

Classes:

* `kalem1`
* `kalem2`
* `defter`

If dataset is not included in the repository, place it here:

```text
data/dataset/
```

And make sure `data.yaml` exists.

---

# 🏋️ Training

Run:

```bash
cd src
python train.py
```

This will:

* load `yolov8n-seg.pt`
* train on the custom dataset
* save outputs in `runs/`
* copy the best model to `models/trained/best.pt`

---

# 📊 Evaluation

Run:

```bash
cd src
python evaluate.py
```

This prints:

* Box Precision
* Box Recall
* Box mAP50
* Box mAP50-95
* Mask Precision
* Mask Recall
* Mask mAP50
* Mask mAP50-95

---

# 🎥 Real-Time Webcam Inference

Run:

```bash
cd src
python webcam.py
```

This opens the camera and performs real-time segmentation.

Press:

* `q` → quit

---

# ⚠️ Notes

* Very small datasets can give overly optimistic results.
* If the model predicts unrelated objects as `defter` or `kalem`, increase confidence threshold in `config.py`.
* For better generalization, collect more images with different angles, lighting conditions, and backgrounds.

---

# 📌 Future Improvements

* add more training images
* add augmentation
* add tracking mode
* add confidence filtering and area filtering
* export model for deployment

---

# 👩‍💻 Author

Melike Çakmakoğlu

````