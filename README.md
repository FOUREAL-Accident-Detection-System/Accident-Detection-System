# 🚨 Accident Detection System

An AI-based road accident detection system that analyzes traffic videos and classifies them as **Accident** or **Normal** using a deep learning model.

The system combines **MobileNetV2** for spatial feature extraction and **LSTM** for temporal sequence analysis. It also provides automated inference, accident evidence extraction, a web interface, and SMS alert functionality.

---

## 📌 Problem Statement

Road accidents require quick detection and response. Manual monitoring of traffic videos can be slow and difficult, especially when large amounts of video data need to be analyzed.

This project aims to develop an AI-based system that can automatically analyze road traffic videos and identify potential accident events.

---

## 💡 Proposed System

The proposed system follows this workflow:

Video Upload
↓
Frame Sampling
↓
Image Preprocessing
↓
Sequence Formation
↓
MobileNetV2 Feature Extraction
↓
LSTM Temporal Analysis
↓
Accident / Normal Prediction
↓
Confidence & Accident Probability
↓
Approximate Detection Window
↓
Evidence Frame Extraction
↓
SMS Alert

---

## ✨ Key Features

- 🎥 Traffic video upload and preview
- 🤖 AI-based Accident / Normal classification
- 📊 Accident probability and confidence
- ⏱️ Approximate accident detection window
- 📸 Genuine evidence frame extracted from the uploaded video
- 📱 SMS alert using TextBee
- ⚙️ Automated batch inference
- 📄 CSV-based prediction results
- 🌐 Web-based demonstration interface

---

## 🧠 AI Model

The main model uses:

- MobileNetV2
- Global Average Pooling
- LSTM
- Dense layer
- Dropout
- Sigmoid output

### Model Architecture

Input Video
→ 32-frame sequence
→ MobileNetV2
→ Global Average Pooling
→ LSTM (64 units)
→ Dense (32 units)
→ Dropout (0.3)
→ Sigmoid
→ Accident / Normal

MobileNetV2 is used as a pretrained ImageNet feature extractor, while the LSTM captures temporal information across video frames.

---

## ⚙️ Final Model Configuration

| Parameter | Value |
|---|---|
| Model | MobileNetV2 + LSTM |
| FPS | 10 |
| Image Size | 160 × 160 |
| Sequence Length | 32 |
| Stride | 32 |
| LSTM Units | 64 |
| Dense Units | 32 |
| Dropout | 0.3 |
| Learning Rate | 0.0001 |
| Batch Size | 4 |
| Threshold | 0.45 |
| Aggregation | Maximum Accident Probability |

---

## 📊 Final Model Evaluation

The final clean evaluation produced:

| Metric | Result |
|---|---:|
| Accuracy | 93.20% |
| Precision | 90.91% |
| Recall | 62.50% |
| F1 Score | 74.07% |

### Confusion Matrix

|  | Predicted Normal | Predicted Accident |
|---|---:|---:|
| Actual Normal | TN = 86 | FP = 1 |
| Actual Accident | FN = 6 | TP = 10 |

The evaluation metrics above represent the main clean evaluation of the final Seq32 model.

---

## 🔬 Experiments

Several experiments were performed to determine suitable model parameters:

- Frame-rate experiments
- Resolution experiments
- Sequence-length experiments
- Threshold tuning
- Model configuration experiments
- Unseen-video/generalization testing

Sequence length experiments included comparison between:

- Sequence length 32
- Sequence length 64

These experiments were used to study the effect of temporal context on model performance.

---

## ⚙️ Automated Inference

The project includes an automated batch inference pipeline.

The automation:

1. Loads the trained model
2. Processes videos
3. Samples frames at the configured FPS
4. Forms frame sequences
5. Performs model prediction
6. Determines Accident / Normal
7. Calculates confidence
8. Records the detection timestamp
9. Saves results to CSV

---

## 🌐 Web Application

The web application provides an interface for demonstrating the complete system.

### Workflow

1. Upload a traffic video
2. Preview the uploaded video
3. Click **Analyze Video**
4. Backend performs AI inference
5. Display Accident / Normal result
6. Display confidence and accident probability
7. Display approximate detection window
8. Display genuine evidence frame
9. Display SMS status

The web application is intended as a project demonstration and evaluation interface.

---

## 📸 Evidence Extraction

When an accident is detected, the backend extracts a genuine frame from the uploaded video around the detected sequence.

Because the model operates on a sequence of frames rather than a single frame, the system reports an **approximate detection window** rather than claiming an exact accident instant.

---

## 📱 SMS Alert

When an accident is detected, the system can send an SMS alert through **TextBee**.

The web interface reports the actual SMS result as:

- `SMS SENT`
- `SMS FAILED`
- `SMS NOT TRIGGERED`

No SMS status is simulated by the application.

---

## 🗂️ Project Structure

```text
Accident-Detection-System/
│
├── arthis backend/
│   └── main.py
│
├── Frontend/
│   └── index.html
│
├── scripts/
│   ├── web_inference.py
│   ├── batch_inference_seq32.py
│   ├── evaluate_101_unseen.py
│   └── ...
│
├── models/
│   └── accident_detection_clean_10fps_160x160_seq32_best.keras
│
├── results/
│   └── ...
│
├── detection_frames/
│   └── ...
│
├── uploads/
│
├── dataset/
│   ├── accident/
│   └── normal/
│
└── README.md

🛠️ Technologies Used
Programming
Python
Deep Learning
TensorFlow
Keras
MobileNetV2
LSTM
Computer Vision
OpenCV
NumPy

Web Application
FastAPI
HTML
CSS
JavaScript
Communication
TextBee SMS API
Development
Visual Studio Code
Git
GitHub

💻 Environment

The project was developed and tested using:

Python 3.11
TensorFlow 2.20.0
OpenCV
NumPy
scikit-learn
Windows
CPU-based execution environment

👥 Team Contributions
AI/ML
Problem definition
Dataset preparation
Preprocessing
Model development
Model training
FPS and resolution experiments
Sequence-length experiments
Threshold tuning
Automation
Model evaluation
Application
Frontend development
Backend development
AI/application integration
Web interface
Result & Alert Handling
Result handling/storage
SMS integration using TextBee
Failure Analysis
Failure-case analysis
Generalization testing
Unseen-video evaluation
Limitations analysis
⚠️ Limitations
Performance depends on the quality and diversity of the training dataset.
Generalization to completely unseen real-world conditions requires further testing.
The system is designed as a video-analysis and demonstration system rather than a production-grade live surveillance system.
Detection time is represented as an approximate window based on the detected video sequence.
🚀 Future Work

Possible future improvements include:

Larger and more diverse accident datasets
More extensive real-world validation
Improved generalization to unseen environments
More detailed accident classification
Severity classification
Improved localization of accident regions
Deployment on suitable real-time hardware
Further optimization for inference speed

▶️ Running the Web Application

Start the backend from the project root:

uvicorn main:app --reload

The web application can then communicate with the FastAPI backend for video analysis.

📌 Note

This project is developed as an academic/training project for demonstrating AI-based accident detection and should not be treated as a certified safety-critical system.

