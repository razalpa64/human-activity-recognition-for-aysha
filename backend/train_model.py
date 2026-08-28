import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from preprocessing import find_dataset_path
from features import process_raw_window

def load_raw_signals(dataset_path, subset="train"):
    """
    Loads raw inertial signals for total acceleration and gyroscope.
    Returns:
        total_acc: (N, 128, 3)
        body_gyro: (N, 128, 3)
    """
    signals_dir = os.path.join(dataset_path, subset, "Inertial Signals")

    tax = np.loadtxt(os.path.join(signals_dir, f"total_acc_x_{subset}.txt"))
    tay = np.loadtxt(os.path.join(signals_dir, f"total_acc_y_{subset}.txt"))
    taz = np.loadtxt(os.path.join(signals_dir, f"total_acc_z_{subset}.txt"))
    total_acc = np.stack([tax, tay, taz], axis=2)

    bgx = np.loadtxt(os.path.join(signals_dir, f"body_gyro_x_{subset}.txt"))
    bgy = np.loadtxt(os.path.join(signals_dir, f"body_gyro_y_{subset}.txt"))
    bgz = np.loadtxt(os.path.join(signals_dir, f"body_gyro_z_{subset}.txt"))
    body_gyro = np.stack([bgx, bgy, bgz], axis=2)

    return total_acc, body_gyro

def extract_features_dataset(total_acc, body_gyro):
    """
    Extracts features for all sample windows in a dataset.
    """
    N = total_acc.shape[0]
    X_features = []
    feature_names = None

    print(f"Extracting features from {N} raw sensor windows...")
    for i in range(N):
        feat_vec, names = process_raw_window(total_acc[i], body_gyro[i], fs=50)
        if feature_names is None:
            feature_names = names
        X_features.append(feat_vec)
        if (i + 1) % 1000 == 0 or (i + 1) == N:
            print(f"  Processed {i + 1}/{N} windows...")

    return np.array(X_features), feature_names

def train_and_evaluate():
    print("Locating dataset...")
    try:
        dataset_path = find_dataset_path()
        print(f"Dataset found at: {dataset_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    print("Loading activity labels...")
    activity_labels_path = os.path.join(dataset_path, "activity_labels.txt")
    activity_map = {}
    activity_names = []
    with open(activity_labels_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0])
                name = parts[1]
                activity_map[idx] = name
                activity_names.append(name)

    print("Loading raw training signals...")
    total_acc_train, body_gyro_train = load_raw_signals(dataset_path, "train")
    y_train = np.loadtxt(os.path.join(dataset_path, "train", "y_train.txt"))

    print("Loading raw test signals...")
    total_acc_test, body_gyro_test = load_raw_signals(dataset_path, "test")
    y_test = np.loadtxt(os.path.join(dataset_path, "test", "y_test.txt"))

    # Extract or load cached extracted features
    train_cache_path = "../results/X_train_features.npy"
    test_cache_path = "../results/X_test_features.npy"

    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print("Loading cached extracted feature matrices...")
        X_train_raw = np.load(train_cache_path)
        X_test_raw = np.load(test_cache_path)
        _, feature_names = process_raw_window(total_acc_train[0], body_gyro_train[0])
    else:
        print("Extracting features for training set...")
        X_train_raw, feature_names = extract_features_dataset(total_acc_train, body_gyro_train)
        np.save(train_cache_path, X_train_raw)

        print("Extracting features for testing set...")
        X_test_raw, _ = extract_features_dataset(total_acc_test, body_gyro_test)
        np.save(test_cache_path, X_test_raw)

    print(f"Extracted feature matrix shapes: Train={X_train_raw.shape}, Test={X_test_raw.shape}")

    # Fit StandardScaler on training features
    print("Fitting StandardScaler on training feature matrix...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Save Activity Distribution graph
    print("Generating activity distribution plot...")
    plt.figure(figsize=(10, 5))
    activity_series = pd.Series(y_train).map(activity_map)
    sns.countplot(y=activity_series, order=activity_names, palette="viridis")
    plt.title("UCI HAR Dataset - Training Samples Activity Distribution")
    plt.xlabel("Number of Samples")
    plt.ylabel("Activity")
    plt.tight_layout()
    plt.savefig("../results/activity_distribution.png", dpi=150)
    plt.close()

    # Define models to train and compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        print(f"  Training took {train_time:.2f} seconds.")

        print(f"Evaluating {name}...")
        start_time = time.time()
        y_pred = model.predict(X_test_scaled)
        pred_time = time.time() - start_time

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        print(f"  Test Accuracy: {acc:.4%}")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "Training_Time_s": train_time,
            "Prediction_Time_s": pred_time,
            "model_object": model
        })

    # Compare models and select best
    df_results = pd.DataFrame(results)
    best_row = df_results.sort_values(by="Accuracy", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model_obj = best_row["model_object"]

    print(f"\n==================================================")
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(f"Accuracy: {best_row['Accuracy']:.4%}")
    print(f"==================================================")

    # Save results CSV
    df_results_save = df_results.drop(columns=["model_object"])
    df_results_save.to_csv("../results/final_results.csv", index=False)

    # Plot model comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Accuracy", y="Model", data=df_results_save, palette="Blues_d")
    plt.title("Model Comparison - Test Accuracy (Raw Signal Pipeline)")
    plt.xlim(0.7, 1.0)
    plt.xlabel("Accuracy")
    plt.ylabel("Model Algorithm")
    for idx, row in df_results_save.iterrows():
        plt.text(row["Accuracy"] + 0.005, idx, f"{row['Accuracy']:.2%}", va='center')
    plt.tight_layout()
    plt.savefig("../results/model_comparison.png", dpi=150)
    plt.close()

    # Save model and scaler
    print("Saving best model to ../models/har_model.pkl...")
    joblib.dump(best_model_obj, "../models/har_model.pkl")

    print("Saving scaler to ../models/scaler.pkl...")
    joblib.dump(scaler, "../models/scaler.pkl")

    # Generate confusion matrix for best model
    y_pred_best = best_model_obj.predict(X_test_scaled)
    report_dict = classification_report(y_test, y_pred_best, target_names=activity_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv("../results/classification_report.csv")

    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=activity_names, yticklabels=activity_names)
    plt.title(f"Confusion Matrix\n{best_model_name} (Acc: {best_row['Accuracy']:.2%})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("../results/confusion_matrix.png", dpi=150)
    plt.close()

    metadata = {
        "training_samples": int(X_train_scaled.shape[0]),
        "testing_samples": int(X_test_scaled.shape[0]),
        "features_count": int(X_train_scaled.shape[1]),
        "activities_count": int(len(activity_names)),
        "best_model_name": best_model_name,
        "accuracy": float(best_row["Accuracy"]),
        "precision": float(best_row["Precision"]),
        "recall": float(best_row["Recall"]),
        "f1_score": float(best_row["F1_Score"]),
        "training_time_s": float(best_row["Training_Time_s"]),
        "prediction_time_s": float(best_row["Prediction_Time_s"]),
        "features": feature_names,
        "activity_names": activity_names,
        "activity_map": activity_map,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print("Saving model metadata to ../models/model_metadata.pkl...")
    joblib.dump(metadata, "../models/model_metadata.pkl")
    print("Training pipeline complete.")
    return True

if __name__ == "__main__":
    train_and_evaluate()
