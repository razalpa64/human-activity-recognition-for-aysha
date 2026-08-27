import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Import our dataset locator
from preprocessing import find_dataset_path, load_features_list

def train_and_evaluate():
    # 1. Verify and locate dataset
    print("Locating dataset...")
    try:
        dataset_path = find_dataset_path()
        print(f"Dataset found at: {dataset_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

    # 2. Create models/ and results/ directories if they don't exist
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    # 3. Load feature names
    features = load_features_list(dataset_path)
    print(f"Loaded {len(features)} feature names.")

    # 4. Load training and testing data
    print("Loading training data...")
    X_train = np.loadtxt(os.path.join(dataset_path, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(dataset_path, "train", "y_train.txt"))
    subject_train = np.loadtxt(os.path.join(dataset_path, "train", "subject_train.txt"))

    print("Loading test data...")
    X_test = np.loadtxt(os.path.join(dataset_path, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(dataset_path, "test", "y_test.txt"))
    subject_test = np.loadtxt(os.path.join(dataset_path, "test", "subject_test.txt"))

    # 5. Validate dimensions
    print("\nValidating dataset dimensions:")
    print(f"  Training samples: {X_train.shape[0]} (Expected: 7352)")
    print(f"  Testing samples: {X_test.shape[0]} (Expected: 2947)")
    print(f"  Feature count: {X_train.shape[1]} (Expected: 561)")
    print(f"  Number of classes: {len(np.unique(y_train))} (Expected: 6)")

    assert X_train.shape[0] == 7352, "X_train row count mismatch!"
    assert X_test.shape[0] == 2947, "X_test row count mismatch!"
    assert X_train.shape[1] == 561, "X_train feature count mismatch!"
    assert X_test.shape[1] == 561, "X_test feature count mismatch!"

    # Load activity labels
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
    print(f"Activity labels: {activity_map}")

    # 6. Save Activity Distribution graph
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

    # 7. Define models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42)
    }

    results = []

    # 8. Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"  Training took {train_time:.2f} seconds.")

        print(f"Evaluating {name}...")
        start_time = time.time()
        y_pred = model.predict(X_test)
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

    # 9. Compare models and select the best one
    df_results = pd.DataFrame(results)
    best_row = df_results.sort_values(by="Accuracy", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model_obj = best_row["model_object"]

    print(f"\n==================================================")
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(f"Accuracy: {best_row['Accuracy']:.4%}")
    print(f"==================================================")

    # Save model comparison table
    df_results_save = df_results.drop(columns=["model_object"])
    df_results_save.to_csv("../results/final_results.csv", index=False)

    # Plot model comparisons
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Accuracy", y="Model", data=df_results_save, palette="Blues_d")
    plt.title("Model Comparison - Test Accuracy")
    plt.xlim(0.8, 1.0)
    plt.xlabel("Accuracy")
    plt.ylabel("Model Algorithm")
    # Annotate bars
    for idx, row in df_results_save.iterrows():
        plt.text(row["Accuracy"] + 0.005, idx, f"{row['Accuracy']:.2%}", va='center')
    plt.tight_layout()
    plt.savefig("../results/model_comparison.png", dpi=150)
    plt.close()

    # 10. Save the best model
    print(f"Saving best model to ../models/har_model.pkl...")
    joblib.dump(best_model_obj, "../models/har_model.pkl")

    # Generate predictions using the best model to construct confusion matrix and classification report
    y_pred_best = best_model_obj.predict(X_test)

    # Save classification report as CSV
    report_dict = classification_report(y_test, y_pred_best, target_names=activity_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv("../results/classification_report.csv")

    # Save confusion matrix plot
    print("Generating confusion matrix plot...")
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

    # 11. Save model metadata
    metadata = {
        "training_samples": int(X_train.shape[0]),
        "testing_samples": int(X_test.shape[0]),
        "features_count": int(X_train.shape[1]),
        "activities_count": int(len(activity_names)),
        "best_model_name": best_model_name,
        "accuracy": float(best_row["Accuracy"]),
        "precision": float(best_row["Precision"]),
        "recall": float(best_row["Recall"]),
        "f1_score": float(best_row["F1_Score"]),
        "training_time_s": float(best_row["Training_Time_s"]),
        "prediction_time_s": float(best_row["Prediction_Time_s"]),
        "features": features,
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
