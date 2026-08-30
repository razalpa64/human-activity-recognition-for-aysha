import os
import numpy as np
import joblib
from preprocessing import find_dataset_path, DescriptorEngine
from features import process_raw_window, compute_rotation_matrix, TRAIN_STANDING_GRAVITY

class Predictor:
    def __init__(self):
        self.backend_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(os.path.dirname(self.backend_dir), "models")
        self.model_path = os.path.join(self.models_dir, "har_model.pkl")
        self.scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        self.metadata_path = os.path.join(self.models_dir, "model_metadata.pkl")

        self.model = None
        self.scaler = None
        self.metadata = None
        self.X_test = None
        self.y_test = None
        self.desc_engine = None
        self.indexed_samples = []

        self.initialize()

    def initialize(self):
        if os.path.exists(self.model_path) and os.path.exists(self.metadata_path):
            self.model = joblib.load(self.model_path)
            self.metadata = joblib.load(self.metadata_path)
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            else:
                self.scaler = None
        else:
            raise FileNotFoundError("Trained model or metadata is missing. Please train the model first.")

        dataset_path = find_dataset_path()
        try:
            self.X_test = np.loadtxt(os.path.join(dataset_path, "test", "X_test.txt"))
            self.y_test = np.loadtxt(os.path.join(dataset_path, "test", "y_test.txt"))
            self.desc_engine = DescriptorEngine(dataset_path)
            self.indexed_samples = self.desc_engine.build_test_index(self.X_test)
        except Exception as e:
            print(f"Warning: could not load raw UCI test samples: {e}")

    def predict_from_raw_window(self, acc_raw, gyro_raw, rotation_matrix=None):
        """
        Accepts raw accelerometer readings (N x 3) and raw gyroscope readings (N x 3).
        Processes them through the exact digital signal processing + feature extraction pipeline,
        scales them with the training StandardScaler, and runs inference with the ML model.
        """
        # 1. Feature extraction & rotation
        feat_vec, feat_names = process_raw_window(acc_raw, gyro_raw, fs=50, rotation_matrix=rotation_matrix)
        feat_matrix = feat_vec.reshape(1, -1)

        # 2. Scaling (never fit live data, use fitted scaler from training)
        if self.scaler is not None:
            feat_scaled = self.scaler.transform(feat_matrix)
        else:
            feat_scaled = feat_matrix

        # 3. Model prediction
        probs = self.model.predict_proba(feat_scaled)[0]
        best_class_idx = np.argmax(probs)

        model_classes = self.model.classes_
        predicted_class_val = model_classes[best_class_idx]

        activity_name = self.metadata['activity_map'][int(predicted_class_val)]
        confidence = float(probs[best_class_idx])

        top_predictions = []
        for idx, class_val in enumerate(model_classes):
            top_predictions.append({
                "activity": self.metadata['activity_map'][int(class_val)],
                "probability": float(probs[idx])
            })
        top_predictions.sort(key=lambda x: x['probability'], reverse=True)

        return {
            "activity": activity_name,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "model": self.metadata.get('best_model_name', 'Trained Classifier'),
            "features_extracted_count": int(len(feat_vec)),
            "window_samples": len(acc_raw),
            "scaler_applied": self.scaler is not None
        }

    def predict_questionnaire(self, intensity, stability, body_position, rotation, movement_pattern):
        matched_indices = []
        for sample in self.indexed_samples:
            if (sample['intensity'] == intensity and
                sample['stability'] == stability and
                sample['body_position'] == body_position and
                sample['rotation'] == rotation and
                sample['movement_pattern'] == movement_pattern):
                matched_indices.append(sample['sample_id'])

        # Fallback to weighted partial matching if 0 exact matches found
        if not matched_indices and self.indexed_samples:
            scored_samples = []
            for sample in self.indexed_samples:
                score = 0
                if sample['body_position'] == body_position: score += 4
                if sample['intensity'] == intensity: score += 3
                if sample['stability'] == stability: score += 2
                if sample['rotation'] == rotation: score += 2
                if sample['movement_pattern'] == movement_pattern: score += 2
                scored_samples.append((score, sample['sample_id']))
            
            scored_samples.sort(key=lambda x: x[0], reverse=True)
            max_score = scored_samples[0][0]
            matched_indices = [sid for sc, sid in scored_samples if sc >= max(1, max_score - 2)]

        if not matched_indices or self.X_test is None:
            return None

        matched_X = self.X_test[matched_indices]
        probs = self.model.predict_proba(matched_X)
        avg_probs = np.mean(probs, axis=0)

        best_class_idx = np.argmax(avg_probs)
        model_classes = self.model.classes_
        predicted_class_val = model_classes[best_class_idx]

        activity_name = self.metadata['activity_map'][int(predicted_class_val)]
        confidence = float(avg_probs[best_class_idx])

        top_predictions = []
        for idx, class_val in enumerate(model_classes):
            top_predictions.append({
                "activity": self.metadata['activity_map'][int(class_val)],
                "probability": float(avg_probs[idx])
            })
        top_predictions.sort(key=lambda x: x['probability'], reverse=True)

        return {
            "activity": activity_name,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "matched_samples": len(matched_indices),
            "model": self.metadata.get('best_model_name', 'Trained Classifier')
        }

    def predict_sample(self, sample_id):
        if self.X_test is None or sample_id < 0 or sample_id >= len(self.X_test):
            raise IndexError(f"Sample ID must be between 0 and {len(self.X_test) - 1}.")

        X_row = self.X_test[sample_id].reshape(1, -1)
        probs = self.model.predict_proba(X_row)[0]
        best_class_idx = np.argmax(probs)

        model_classes = self.model.classes_
        predicted_class_val = model_classes[best_class_idx]

        predicted_activity = self.metadata['activity_map'][int(predicted_class_val)]
        confidence = float(probs[best_class_idx])

        actual_class_val = self.y_test[sample_id]
        actual_activity = self.metadata['activity_map'][int(actual_class_val)]

        is_correct = bool(int(predicted_class_val) == int(actual_class_val))

        top_predictions = []
        for idx, class_val in enumerate(model_classes):
            top_predictions.append({
                "activity": self.metadata['activity_map'][int(class_val)],
                "probability": float(probs[idx])
            })
        top_predictions.sort(key=lambda x: x['probability'], reverse=True)

        return {
            "sample_id": int(sample_id),
            "features_count": int(self.X_test.shape[1]),
            "predicted": predicted_activity,
            "confidence": confidence,
            "actual": actual_activity,
            "correct": is_correct,
            "model": self.metadata.get('best_model_name', 'Trained Classifier'),
            "top_predictions": top_predictions
        }
