# Human Activity Recognition

**UCI Smartphone Sensor Based Activity Prediction**

A complete full-stack machine learning web application that trains on the official UCI HAR Dataset and classifies 6 physical activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying) from 561-dimensional smartphone sensor features.

---

## Features

- **Questionnaire Mode** — Describe physical characteristics to find matching UCI test recordings and get ML predictions
- **UCI Sample Mode** — Select any test sample (1–2947) and see the model's prediction vs ground truth
- **Live Sensor Mode** — Use your smartphone's accelerometer and gyroscope for real-time activity classification via nearest-neighbor UCI matching
- **Model Dashboard** — Live accuracy metrics, confusion matrix, model comparison chart
- **Premium UI** — Minimal, research-grade web interface with responsive design

---

## Setup

### 1. Download the UCI HAR Dataset

Download from: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Place the extracted folder at:

```
dataset/UCI HAR Dataset/
├── activity_labels.txt
├── features.txt
├── features_info.txt
├── README.txt
├── train/
│   ├── X_train.txt
│   ├── y_train.txt
│   └── subject_train.txt
└── test/
    ├── X_test.txt
    ├── y_test.txt
    └── subject_test.txt
```

### 2. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Train the ML Model

```bash
cd backend
python train_model.py
```

This will:
- Verify dataset (7352 train, 2947 test, 561 features, 6 classes)
- Train Logistic Regression, Decision Tree, Random Forest, and SVM
- Select the best model by test accuracy
- Save model and metadata to `models/`
- Generate charts in `results/`

### 4. Start the Application

```bash
cd backend
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Project Structure

```
Human_Activity_Recognition/
├── backend/
│   ├── app.py              Flask API server
│   ├── train_model.py      ML training pipeline
│   ├── prediction.py       Prediction engine
│   ├── preprocessing.py    Descriptor mapping engine
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── dataset/                ← Add UCI HAR Dataset here
├── models/                 ← Generated after training
├── results/                ← Charts and CSV reports
└── README.md
```

---

## ML Pipeline

| Model                  | Notes                            |
|------------------------|----------------------------------|
| Logistic Regression    | ~96% accuracy on test set         |
| Decision Tree          | ~86% accuracy                    |
| Random Forest          | ~92% accuracy                    |
| Support Vector Machine | ~96% accuracy, selected as best  |

---

## Live Sensor Mode

When opened on a smartphone, the app uses the browser's `DeviceMotion` API to capture accelerometer and gyroscope readings. The browser computes aggregate statistics from a sliding 50-sample window. These statistics are sent to the backend, where a nearest-neighbor search finds the 30 most similar UCI test recordings. Those recordings' full 561-feature vectors are classified by the trained ML model.

**Raw sensor values never enter the 561-feature ML model directly.**

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Backend and model health |
| `/api/model` | GET | Model metadata and accuracy |
| `/api/predict` | POST | Questionnaire → UCI match → prediction |
| `/api/sample/<id>` | GET | Single UCI test sample prediction |
| `/api/live-predict` | POST | Live sensor stats → UCI match → prediction |
| `/api/activities` | GET | Activity descriptions |
| `/api/results` | GET | Model comparison metrics and chart paths |
| `/api/train` | POST | Retrain model pipeline |

---

## Technical Notes

- Data leakage prevention: `y_test` is only accessed after model prediction for evaluation
- Training never uses `X_test` data
- All confidence values are real model probabilities
- No hardcoded predictions or fake readings
