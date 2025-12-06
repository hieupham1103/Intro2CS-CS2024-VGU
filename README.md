# Robust Real-time UAV Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Multiscale Processing & Cross-Head Knowledge Distillation for detecting small UAVs in challenging environments**

📄 **[Project Website](https://hieupham1103.github.io/Intro2CS-CS2024-VGU/)** | 📊 **[Seminar Slides](docs/Seminar_Slide.pdf)**

---

## Abstract

The proliferation of small Unmanned Aerial Vehicles (UAVs) poses significant security challenges to critical infrastructure. This project presents a computer vision system using a **YOLOv8-based architecture** enhanced by:

1. **Multiscale Processing** - 5-crop inference mechanism with Non-Maximum Weighted (NMW) fusion
2. **Cross-Head Knowledge Distillation (CrossKD)** - Efficient transfer of detection-sensitive features from Teacher to Student model


## System Pipeline

![Pipeline](docs/images/pipeline.png)

### Training Pipeline
1. **Transfer Learning** - Pre-train on DUT Anti-UAV Dataset (10k images)
2. **Progressive KD** - YOLOv8-X → YOLOv8-L → YOLOv8-Nano
3. **CrossKD Loss** - Cross-head distillation for detection-sensitive feature transfer

### Inference Pipeline
```
Input Frame → 5-Crop Split → YOLOv8-n (Batch) → NMW Fusion → Result
```


## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/hieupham1103/Intro2CS-CS2024-VGU.git
cd Intro2CS-CS2024-VGU

# Install dependencies
pip install -r requirements.txt
```

**Required packages:**
- ultralytics
- ensemble_boxes
- orjson
- filterpy
- lap
- tqdm


## How to Reproduce

### 0. Dataset Preparation

**Download the dataset from Kaggle:**

```bash
# Download from Kaggle
# https://www.kaggle.com/datasets/hiuphmnhtrung/vipcup-2025-train-dataset
```

Or use Kaggle CLI:
```bash
kaggle datasets download -d hiuphmnhtrung/vipcup-2025-train-dataset
unzip vipcup-2025-train-dataset.zip
```

**Organize the dataset:**

The downloaded dataset has the following structure:
```
dataset_mix/
├── split_A/
│   ├── IR/
│   └── RGB/
└── split_B/
    ├── IR/
    └── RGB/
```

We use **split_A only**. Move the IR and RGB folders to the `data/` directory:

```bash
mkdir -p data
mv dataset_mix/split_A/IR data/
mv dataset_mix/split_A/RGB data/
```

Your `data/` folder should now look like:
```
data/
├── IR/
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   └── labels/
│       ├── train/
│       └── test/
└── RGB/
    ├── images/
    └── labels/
```

**Create videos from image sequences:**

The dataset contains image sequences. Use `create_videos.py` to convert them into video format:

```bash
python create_videos.py
```

This will:
- Process both RGB and IR datasets
- Group images by sequence ID
- Create MP4 videos (24 FPS, 5s duration = 120 frames)
- Generate corresponding label files for each video

Output structure:
```
data/
├── RGB_video/
│   ├── videos/
│   │   ├── train/
│   │   └── test/
│   └── labels/
│       ├── train/
│       └── test/
└── IR_video/
    ├── videos/
    └── labels/
```

### 1. Training Single Model (YOLOv8)

Use `train.py` to train a single YOLOv8 model on RGB or IR datasets.

**Basic Usage:**
```bash
# Train on RGB dataset
python train.py --set RGB --model yolov8n.pt

# Train on IR (Infrared) dataset
python train.py --set IR --model yolov8n.pt
```

**Advanced Options:**
```bash
python train.py --set RGB \
    --model yolov8n.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --output outputs
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--set` | Dataset to use (`RGB` or `IR`) | Required |
| `--model` | Base model path | `yolov8n.pt` |
| `--config` | Training config YAML | `configs/train_{set}.yml` |
| `--epochs` | Number of epochs | From config |
| `--batch` | Batch size | From config |
| `--imgsz` | Image size | From config |
| `--device` | GPU device (0, 1, cpu) | From config |
| `--output` | Output directory | `outputs` |
| `--resume` | Resume from checkpoint | False |

**Configuration Files:**
- `configs/train_RGB.yml` - RGB training hyperparameters
- `configs/train_IR.yml` - IR training hyperparameters
- `configs/data_RGB.yml` - RGB dataset configuration
- `configs/data_IR.yml` - IR dataset configuration


### 2. Knowledge Distillation with CrossKD

For Cross-Head Knowledge Distillation, we use the [CrossKD](https://github.com/jbwang1997/CrossKD) framework. The implementation is in the `CrossKD/` directory.

#### Setup CrossKD Environment

```bash
# Create conda environment
conda create --name crosskd python=3.8 -y
conda activate crosskd

# Install PyTorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install MMEngine and MMCV
pip install -U openmim
mim install "mmengine==0.7.3"
mim install "mmcv==2.0.0rc4"

# Install CrossKD
cd CrossKD
pip install -v -e .
```

#### Training with CrossKD

**Single GPU:**
```bash
python tools/train.py configs/crosskd/${CONFIG_FILE}
```

**Multi GPU:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh \
    configs/crosskd/${CONFIG_FILE} 4
```

#### Evaluation
```bash
python tools/test.py configs/crosskd/${CONFIG_FILE} ${CHECKPOINT_FILE}
```

For detailed CrossKD documentation, see [CrossKD/README.md](CrossKD/README.md).


### 3. Evaluation

Use `evaluate.py` to evaluate trained models:

```bash
python evaluate.py --model checkpoints/rgb.pt --set RGB
```

### 4. Visualization

Generate detection videos:

```bash
python visualize_videos.py --model checkpoints/rgb.pt --input data/test --output outputs/visualizations
```


## Project Structure

```
DroneTracking/
├── train.py                 # Single model training script
├── evaluate.py              # Evaluation script
├── visualize_videos.py      # Video visualization
├── create_videos.py         # Video creation utilities
├── requirements.txt         # Python dependencies
├── configs/
│   ├── train_RGB.yml        # RGB training config
│   ├── train_IR.yml         # IR training config
│   ├── data_RGB.yml         # RGB dataset config
│   ├── data_IR.yml          # IR dataset config
│   └── bytetrack.yml        # Tracking config
├── checkpoints/             # Pre-trained model weights
│   ├── rgb.pt
│   └── ir.pt
├── models/
│   ├── model.py             # Base model
│   ├── multiscale_model.py  # Multiscale inference
│   ├── track_model.py       # Tracking model
│   └── compensation_tracker.py
├── CrossKD/                 # CrossKD distillation framework
│   ├── configs/crosskd/     # Distillation configs
│   ├── tools/               # Training/testing tools
│   └── ...
├── docs/
│   ├── index.html           # Project website
│   ├── Seminar_Slide.pdf    # Presentation slides
│   └── images/
│       └── pipeline.png     # Pipeline diagram
└── outputs/                 # Training outputs
```


## Methodology

### Multiscale Processing (5-Crop)

To address small object detection:
- Image divided into **4 corner crops + 1 center crop**
- Each crop covers ~55-65% of original frame
- Creates "zoom" effect preserving small details
- Results merged via **Non-Maximum Weighted (NMW)** fusion

### Cross-Head Knowledge Distillation

Instead of traditional feature mimicking, CrossKD establishes cross-connections:
- **Student Features → Teacher Head**: Forces student backbone to learn robust features
- **Teacher Features → Student Head**: Trains student head to process complex features

**Progressive Pipeline:**
```
YOLOv8-X (Teacher) → YOLOv8-L (Intermediate) → YOLOv8-Nano (Student)
```


## Results

| Configuration | mAP@0.5 | mAP@0.5-0.9 | FPS |
|---------------|---------|-------------|-----|
| YOLOv8-Nano (Baseline) | 0.55 | 0.22 | 77 |
| YOLOv8-Nano (KD) | 0.65 | 0.25 | 77 |
| YOLOv8-Nano (Pretrained) | 0.79 | 0.48 | 77 |
| YOLOv8-Nano (Multiscale) | 0.61 | 0.23 | 28 |
| **YOLOv8-Nano (Full Pipeline)** | **0.84** | **0.51** | **28** |

## References

1. [CrossKD: Cross-Head Knowledge Distillation for Object Detection](https://arxiv.org/abs/2306.11369) - CVPR 2024
2. [High-Speed Drone Detection Based On Yolo-V8](https://doi.org/10.1109/ICASSP49357.2023.10095516) - ICASSP 2023
3. [Vision-based Anti-UAV Detection and Tracking](https://arxiv.org/abs/2205.10851) - IEEE TITS 2022
4. [Improving Small Drone Detection Through Multi-Scale Processing](https://arxiv.org/abs/2504.19347) - IJCNN 2025


## Team

**Introduction to Computer Science Project - VGU (CS2024)**

| Member | Contributions |
|--------|---------------|
| **Pham Dinh Trung Hieu** | Implemented multiscale inference pipeline, applied Cross-Head Knowledge Distillation (CrossKD), created presentation slides and website |
| **Truong Quoc Phong** | Dataset preprocessing, trained and evaluated YOLOv8-P2 model, contributed to slides and website |
| **Pham Minh Thu** | Trained and evaluated YOLOv8 models |
| **Nguyen Hoang Minh** | Trained and evaluated YOLOv11 models |
| **Nguyen Khanh Trang** | Trained and evaluated YOLOv8 models |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [CrossKD](https://github.com/jbwang1997/CrossKD)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- VIP Cup 2025 Dataset
- DUT Anti-UAV Dataset