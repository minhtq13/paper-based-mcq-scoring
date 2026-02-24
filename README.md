# Paper-Based MCQ Scoring System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)]()
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-red.svg)](https://docs.ultralytics.com/vi/models/yolo11/)

An automated optical scoring system for paper-based multiple-choice question (MCQ) answer sheets. The system uses computer vision and deep learning (YOLOv11) to detect alignment markers, extract student/exam information, and recognize selected answers from scanned or photographed answer sheet images — producing structured JSON output suitable for downstream grading pipelines.

---

## Table of Contents

- [Overview](#overview)
- [Versioning](#versioning)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Answer Sheet Template](#answer-sheet-template)
- [Usage](#usage)
  - [Preparing Input Images](#preparing-input-images)
  - [Running the Scoring Pipeline](#running-the-scoring-pipeline)
  - [Output Description](#output-description)
- [Models](#models)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

This system automates the grading of paper-based MCQ exams. Given a folder of answer sheet images (JPEG or PNG), it:

1. **Detects alignment markers** on the answer sheet to correct skew and perspective.
2. **Extracts student information** (class code, student code, exam/test-set code) from the information zone.
3. **Recognizes selected answers** for each question (supporting up to 60 questions per sheet with multi-answer combinations A, B, C, D, AB, AC, …, ABCD).
4. **Writes annotated output images** and structured **JSON result files** per answer sheet.
5. **Logs potentially uncertain predictions** (low-confidence detections) to a warning file.

The pipeline is designed for integration with an e-learning support platform but can also be used as a standalone batch-processing tool.

---

## Versioning

This repository maintains **two branches** corresponding to two distinct implementations:

| Branch                     | Detector | Model strategy                    | Description                                                   |
| -------------------------- | -------- | --------------------------------- | ------------------------------------------------------------- |
| `yolov8` _(paper version)_ | yolov8m  | Single shared model               | As described in the published paper (Tinh & Minh, 2024)       |
| `main` _(this branch)_     | YOLOv11m | Three separate specialized models | Upgraded implementation with improved accuracy and modularity |

### Differences from the Published Paper Version

#### 1. Object Detector: YOLOv8 → YOLOv11

The published paper used **YOLOv8** (released January 2023, Ultralytics). This branch upgrades to **YOLOv11** (released September 2024, Ultralytics), which introduces architectural refinements — particularly the **C3k2** block and **PSAA (Partial Self-Attention Aggregation)** mechanism — resulting in higher accuracy with fewer parameters.

**Comparison of the medium (m) variants used in this project:**

| Metric                             | YOLOv8m | YOLOv11m | Change      |
| ---------------------------------- | ------- | -------- | ----------- |
| Parameters                         | 25.9 M  | 20.1 M   | **−22.4%**  |
| Inference speed (T4 TensorRT FP16) | 5.86 ms | 4.70 ms  | **−19.8%**  |
| COCO mAP50-95                      | 50.2    | 51.5     | **+1.3 pp** |
| FLOPs                              | 78.9 B  | 68.0 B   | **−13.8%**  |

> Source: [Ultralytics YOLOv11 documentation](https://docs.ultralytics.com/models/yolo11/)

In this domain-specific application (answer sheet detection), YOLOv11 achieves higher detection accuracy with a smaller model footprint, making it better suited for deployment.

#### 2. Model Architecture: Single Model → Three Specialized Models

The **paper version** (`yolov8` branch) uses a **single YOLOv8 model** trained on all detection tasks simultaneously (markers, student info digits, and answer bubbles). While this reduces the number of model files to maintain, it requires the model to generalize across visually very different object types.

The **current version** (`main` branch) separates the detection into **three independent specialized models**, each trained exclusively on its own task:

| Model       | Task                                  | Benefit of specialization                      |
| ----------- | ------------------------------------- | ---------------------------------------------- |
| `marker.pt` | Alignment marker detection            | Higher recall on small corner markers          |
| `info.pt`   | Student information digit recognition | Better digit discrimination in dense grids     |
| `answer.pt` | Answer bubble classification          | Improved accuracy on multi-choice combinations |

This specialization allows each model to be fine-tuned independently and retrained without affecting the other tasks, improving both accuracy and maintainability.

---

## Features

- ✅ Automatic perspective correction using marker-based homography
- ✅ Supports 20, 40, and 60 question answer sheets
- ✅ Multi-answer recognition (single and combination choices: AB, AC, AD, BC, BD, CD, ABC, ABD, ACD, BCD, ABCD)
- ✅ Student information zone OCR (class code, student ID, test-set code)
- ✅ Confidence-based warning system for low-certainty predictions
- ✅ JSON output format for easy downstream integration
- ✅ Annotated output images highlighting detected answers

---

## System Architecture

```
Input images (JPG/PNG)
        │
        ▼
┌─────────────────────┐
│  Marker Detection   │  ← marker.pt (YOLOv11)
│  & Image Alignment  │
└────────┬────────────┘
         │  Corrected & cropped document
         ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Info Zone Cropping │ ───► │  Info Recognition   │  ← info.pt (YOLOv11)
│  (Student/Exam ID)  │      │  (digits 0-9, blank)│
└─────────────────────┘      └──────────┬──────────┘
                                        │
┌─────────────────────┐      ┌──────────▼──────────┐
│  Answer Zone        │ ───► │  Answer Recognition │  ← answer.pt (YOLOv11)
│  Column Cropping    │      │  (A/B/C/D combos)   │
└─────────────────────┘      └──────────┬──────────┘
                                        │
                             ┌──────────▼──────────┐
                             │  JSON Output +      │
                             │  Annotated Images   │
                             └─────────────────────┘
```

**Key modules:**

| File                | Description                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `main_algorithm.py` | Main pipeline: marker detection, image alignment, info/answer prediction, output writing |
| `tool_algorithm.py` | Utility functions: geometry, perspective transform, angle calculation, label mapping     |
| `common_main.py`    | Shared helpers: image cropping, image merging                                            |

---

## Requirements

- Python **3.8** or higher
- The following Python packages (see also `requirements.txt`):

| Package                  | Version  | Purpose                           |
| ------------------------ | -------- | --------------------------------- |
| `opencv-python-headless` | 4.9.0.80 | Image processing                  |
| `ultralytics`            | ≥ 8.3    | YOLOv11 model inference           |
| `numpy`                  | ≥ 1.21   | Numerical operations              |
| `Flask`                  | latest   | (Optional) REST API serving       |
| `uwsgi`                  | latest   | (Optional) Production WSGI server |

> **Note:** `Flask` and `uwsgi` are only required if you are deploying the system as a web service. For standalone batch processing, only `opencv-python-headless`, `ultralytics`, and `numpy` are needed.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/paper-based-mcq-scoring.git
cd paper-based-mcq-scoring
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install ultralytics numpy
```

### 4. Verify model files

Ensure the three YOLOv11 model weight files are present in the `Model/` directory:

```
Model/
├── marker.pt      # Alignment marker detector (~5.2 MB)
├── info.pt        # Student information recognizer (~38.6 MB)
└── answer.pt      # Answer choice recognizer (~38.6 MB)
```

> The model files are **not** included in this repository due to their size. Please contact the authors or download them from the provided release assets.

---

## Directory Structure

```
paper-based-mcq-scoring/
│
├── Model/                          # Pre-trained YOLOv11 weights
│   ├── marker.pt
│   ├── info.pt
│   └── answer.pt
│
├── images/
│   └── answer_sheets/
│       └── <exam_class_id>/        # One folder per exam session
│           ├── 1.jpg               # Input answer sheet images
│           ├── 2.jpg
│           ├── ...
│           ├── HandledSheets/      # (auto-created) Annotated output images
│           ├── ScoredSheets/       # (auto-created) JSON result files
│           └── MayBeWrong/         # (auto-created) Low-confidence warning log
│
├── main_algorithm.py               # Main scoring pipeline
├── tool_algorithm.py               # Geometry & label utility functions
├── common_main.py                  # Image crop & merge helpers
├── AnswerSheetTemplateNew.pdf          # Printable answer sheet template
├── requirements.txt
└── README.md
```

---

## Answer Sheet Template

The file `AnswerSheetTemplateNew.pdf` is the official printable template that this system is designed to process. Print it on **A4 paper** before scanning or photographing.

### Layout Overview

```
┌──────────────────────────────────────────────────────────────────┐
│ [■] TL marker          PHIẾU TRẢ LỜI TRẮC NGHIỆM          [■] TR│  ← marker1 (×3)
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────────────────────────┐  │
│  │ Supervisor sign │  │  1. Môn thi  (subject)               │  │
│  │ box 1 & 2       │  │  2. Họ và tên (full name)            │  │
│  └─────────────────┘  │  3. Ngày thi (exam date)             │  │
│                       │  4. Chữ ký   (signature)             │  │
│                       └──────────────────────────────────────┘  │
│                                                                  │
│  5. Mã lớp thi   6. Mã SV (SBD)          7. Mã đề             │
│  ┌──────────┐    ┌──────────────────┐     ┌─────┐              │
│  │6-col OMR │    │10-col OMR (0–9,x) │     │3-col│  ← info zone │
│  └──────────┘    └──────────────────┘     └─────┘              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ║ barcode ║│
│  │ Q1–Q20      │  │ Q21–Q40     │  │ Q41–Q60     │  ║  strip  ║│
│  │ (A B C D)   │  │ (A B C D)   │  │ (A B C D)   │  ║         ║│
│  └─────────────┘  └─────────────┘  └─────────────┘  ║         ║│
│                                                                  │
│ [■] BL marker                                          [⊙] BR   │  ← marker2 (×1)
└──────────────────────────────────────────────────────────────────┘
```

### Marker Positions

| Marker    | Position     | Symbol              | Role                                            |
| --------- | ------------ | ------------------- | ----------------------------------------------- |
| `marker1` | Top-Left     | `■` (filled square) | Alignment — 3 copies                            |
| `marker1` | Top-Right    | `■` (filled square) | Alignment                                       |
| `marker1` | Bottom-Left  | `■` (filled square) | Alignment                                       |
| `marker2` | Bottom-Right | `⊙` (circle-dot)    | Reference corner for rotation angle calculation |

The asymmetric placement of `marker2` at bottom-right allows the algorithm to unambiguously determine the sheet's orientation and calculate the exact skew angle.

### Information Zone (Fields 5, 6, 7)

Located in the upper-right area. Each field is an OMR (Optical Mark Recognition) column grid where students fill in digit bubbles `0`–`9` column by column:

| Field | Label       | Columns | Content                  |
| ----- | ----------- | ------- | ------------------------ |
| 5     | Mã lớp thi  | 6       | Exam class/course code   |
| 6     | Mã SV (SBD) | 10      | Student ID number        |
| 7     | Mã đề       | 3       | Exam set / test-set code |

- A blank or uncircled cell is treated as class `x`.
- In column 1 of field 6, describe the student type (B: bachelor, E: engineer, M: master). The custom code for Hanoi University of Technology students still uses the 0-9 and x labels as in the model above.

### Answer Zone (Fields Q1–Q60)

- **3 vertical columns** of questions, each holding up to **20 questions**
- Each row offers **4 bubbles**: `A`, `B`, `C`, `D`
- Students may fill **one or more bubbles** per question (multi-answer support: `AB`, `ACD`, `ABCD`, etc.)
- A question with no bubble filled is recorded as `unchoice` (unanswered)

### Barcode Strip

A vertical barcode strip on the right edge is a printed identifier for the exam sheet (not processed by this software).

### Printing Notes

- Print at **100% scale** on **A4 (210 × 297 mm)** — do **not** scale to fit
- Use a **laser printer** for best marker contrast
- Ensure all 4 alignment markers are fully printed and not clipped by the page margin

---

## Usage

### Preparing Input Images

1. Create a folder named after the **exam class ID** inside `images/answer_sheets/`:

```bash
mkdir -p images/answer_sheets/<exam_class_id>
```

2. Place all scanned or photographed answer sheet images (`.jpg`, `.jpeg`, or `.png`) inside that folder.

**Image requirements:**

- The answer sheet must contain **4 alignment markers** (3 × `marker1` at top-left, top-right, bottom-left; 1 × `marker2` at bottom-right).
- Recommended image resolution: **≥ 1056 × 1500 px**.
- Supported formats: `JPEG`, `PNG`.

---

### Running the Scoring Pipeline

Run the main script from the project root, passing the exam class folder name as the argument:

```bash
python main_algorithm.py <exam_class_id>
```

**Example:**

```bash
python main_algorithm.py demo1
```

This will process all images inside `images/answer_sheets/demo1/` and write results to the automatically created subdirectories.

---

### Output Description

For each successfully processed answer sheet image (e.g., `1.jpg`), the system produces:

#### 1. JSON Result File — `ScoredSheets/<filename>_data.json`

```json
{
  "examClassCode": "demo1",
  "studentCode": "026983557",
  "testSetCode": "014",
  "answers": [
    { "questionNo": 1, "selectedAnswers": "A" },
    { "questionNo": 2, "selectedAnswers": "BC" },
    { "questionNo": 3, "selectedAnswers": "x" },
    ...
    { "questionNo": 60, "selectedAnswers": "D" }
  ],
  "handledScoredImg": "images/answer_sheets/demo1/HandledSheets/handled_1.jpg",
  "originalImg": "images/answer_sheets/demo1/1.jpg",
  "originalImgFileName": "1.jpg"
}
```

| Field                       | Type      | Description                                                                                                |
| --------------------------- | --------- | ---------------------------------------------------------------------------------------------------------- |
| `examClassCode`             | `string`  | Detected class/course code from the info zone                                                              |
| `studentCode`               | `string`  | Detected student ID number                                                                                 |
| `testSetCode`               | `string`  | Detected test/exam set code                                                                                |
| `answers`                   | `array`   | List of per-question answer objects                                                                        |
| `answers[].questionNo`      | `integer` | Question number (1-indexed)                                                                                |
| `answers[].selectedAnswers` | `string`  | Selected answer(s): `"A"`, `"B"`, `"C"`, `"D"`, combinations like `"AB"`, `"BCD"`, or `"x"` for unanswered |
| `handledScoredImg`          | `string`  | Path to the annotated output image                                                                         |
| `originalImg`               | `string`  | Path to the original input image                                                                           |
| `originalImgFileName`       | `string`  | File name of the original input image                                                                      |

#### 2. Annotated Image — `HandledSheets/handled_<filename>.<ext>`

A copy of the answer sheet with colored bounding boxes drawn over detected answers:

- 🟢 **Green box**: high-confidence prediction
- 🟠 **Orange box**: low-confidence prediction (also logged to warning file)

#### 3. Warning Log — `MayBeWrong/may_be_wrong.txt`

If any detection has a confidence score below the threshold (`0.79` by default), a line is appended:

```
Label "A" question 5;1.jpg;0.71
Label 3 from left to right: "x";1.jpg;0.68
```

Each line contains: `<description>;<filename>;<confidence_score>`.

---

## Models

This branch uses three custom-trained **YOLOv11** object detection models, one dedicated per task:

| Model file  | Task                         | Input region                    | Output classes                                                                                         |
| ----------- | ---------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `marker.pt` | Alignment marker detection   | Full answer sheet image         | `marker1` (×3, at TL/TR/BL), `marker2` (×1, at BR)                                                     |
| `info.pt`   | Student info zone OCR        | Cropped info zone (640×640)     | `0`–`9`, `x` (uncircled/blank)                                                                         |
| `answer.pt` | Answer bubble classification | Cropped answer column (250×640) | `unchoice`, `A`, `B`, `C`, `D`, `AB`, `AC`, `AD`, `BC`, `BD`, `CD`, `ABC`, `ABD`, `ACD`, `BCD`, `ABCD` |

All three models are based on the **YOLOv11m** (nano) architecture, trained on a custom dataset of Vietnamese university MCQ answer sheets.

> For the original single-model implementation as described in the published paper, refer to the `yolov8` branch.

---

## Configuration

Key parameters that can be adjusted directly in the source files:

| Parameter           | Location                         | Default             | Description                                                           |
| ------------------- | -------------------------------- | ------------------- | --------------------------------------------------------------------- |
| `threshold_warning` | `tool_algorithm.py`              | `0.79`              | Confidence threshold below which a prediction is flagged as uncertain |
| `numberAnswer`      | `main_algorithm.py` (main block) | `60`                | Number of questions per answer sheet (supported: `20`, `40`, `60`)    |
| `pWeight_marker`    | `main_algorithm.py`              | `./Model/marker.pt` | Path to the marker detection model                                    |
| `pWeight_info`      | `main_algorithm.py`              | `./Model/info.pt`   | Path to the info recognition model                                    |
| `pWeight_answer`    | `main_algorithm.py`              | `./Model/answer.pt` | Path to the answer recognition model                                  |

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Citation

This software is based on the following peer-reviewed publication. If you use this system in academic work, please cite:

**Pham Doan Tinh and Ta Quang Minh**, "Automated Paper-based Multiple Choice Scoring Framework using Fast Object Detection Algorithm," _International Journal of Advanced Computer Science and Applications (IJACSA)_, vol. 15, no. 1, 2024. DOI: [10.14569/IJACSA.2024.01501115](http://dx.doi.org/10.14569/IJACSA.2024.01501115)

```bibtex
@article{Tinh2024,
  title     = {Automated Paper-based Multiple Choice Scoring Framework using Fast Object Detection Algorithm},
  journal   = {International Journal of Advanced Computer Science and Applications},
  doi       = {10.14569/IJACSA.2024.01501115},
  url       = {http://dx.doi.org/10.14569/IJACSA.2024.01501115},
  year      = {2024},
  publisher = {The Science and Information Organization},
  volume    = {15},
  number    = {1},
  author    = {Pham Doan Tinh and Ta Quang Minh}
}
```

---

## Contact

For questions, issues, or contributions, please open a GitHub Issue or contact the authors:

- **Pham Doan Tinh** — corresponding author
- **Ta Quang Minh** - Email: taminh596@gmail.com - Phone: +84 979047751

Paper available at: [https://thesai.org/Publications/ViewPaper?Volume=15&Issue=1&Code=IJACSA&SerialNo=115](https://thesai.org/Publications/ViewPaper?Volume=15&Issue=1&Code=IJACSA&SerialNo=115)
