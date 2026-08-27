import os
import numpy as np
import joblib
from preprocessing import find_dataset_path, DescriptorEngine

class Predictor:
    def __init__(self):
        # 1. Paths
        self.backend_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(os.path.dirname(self.backend_dir), "models")
        self.model_path = os.path.join(self.models_dir, "har_model.pkl")
        self.metadata_path = os.path.join(self.models_dir, "model_metadata.pkl")
        
        self.model = None
        self.metadata = None
        self.X_test = None
        self.y_test = None
        self.desc_engine = None
        self.indexed_samples = []
        
        # Load everything
        self.initialize()

    def initialize(self):
        # Load model and metadata
        if os.path.exists(self.model_path) and os.path.exists(self.metadata_path):
            self.model = joblib.load(self.model_path)
            self.metadata = joblib.load(self.metadata_path)
        else:
            raise FileNotFoundError("Trained model or metadata is missing. Please train the model first.")

        # Load X_test and y_test
        dataset_path = find_dataset_path()
        self.X_test = np.loadtxt(os.path.join(dataset_path, "test", "X_test.txt"))
        self.y_test = np.loadtxt(os.path.join(dataset_path, "test", "y_test.txt"))

        # Initialize descriptor engine and build index
        self.desc_engine = DescriptorEngine(dataset_path)
        self.indexed_samples = self.desc_engine.build_test_index(self.X_test)

    def predict_questionnaire(self, intensity, stability, body_position, rotation, movement_pattern):
        """
        Processes questionnaire responses, finds matching observations,
        runs them through the ML model, and returns aggregated predictions.
        """
        # Find matching samples in index
        matched_indices = []
        for sample in self.indexed_samples:
            if (sample['intensity'] == intensity and
                sample['stability'] == stability and
                sample['body_position'] == body_position and
                sample['rotation'] == rotation and
                sample['movement_pattern'] == movement_pattern):
                matched_indices.append(sample['sample_id'])
                
        if not matched_indices:
            # No matching records
            return None

        # Retrieve matched X vectors
        matched_X = self.X_test[matched_indices]
        
        # Run prediction probabilities
        probs = self.model.predict_proba(matched_X)
        
        # Aggregate probabilities by mean
        avg_probs = np.mean(probs, axis=0)
        
        # Find class index with highest probability
        best_class_idx = np.argmax(avg_probs)
        
        # SVM models inside scikit-learn store classes as float/int values (e.g. 1.0, 2.0...)
        # We need to map class values to activity labels.
        model_classes = self.model.classes_ # e.g. [1., 2., 3., 4., 5., 6.]
        predicted_class_val = model_classes[best_class_idx]
        
        # Map class value to activity label name
        activity_name = self.metadata['activity_map'][int(predicted_class_val)]
        confidence = avg_probs[best_class_idx]
        
        # Build top predictions list
        top_predictions = []
        for idx, class_val in enumerate(model_classes):
            top_predictions.append({
                "activity": self.metadata['activity_map'][int(class_val)],
                "probability": float(avg_probs[idx])
            })
            
        # Sort top predictions by probability descending
        top_predictions.sort(key=lambda x: x['probability'], reverse=True)

        return {
            "activity": activity_name,
            "confidence": float(confidence),
            "top_predictions": top_predictions,
            "matched_samples": len(matched_indices),
            "model": self.metadata['best_model_name']
        }

    def predict_sample(self, sample_id):
        """
        Runs prediction for a specific sample ID from the test dataset.
        Returns predicted activity, probability confidence, actual label, and correctness.
        """
        if sample_id < 0 or sample_id >= len(self.X_test):
            raise IndexError(f"Sample ID must be between 0 and {len(self.X_test) - 1}.")

        X_row = self.X_test[sample_id].reshape(1, -1)
        
        # Run model prediction
        probs = self.model.predict_proba(X_row)[0]
        best_class_idx = np.argmax(probs)
        
        model_classes = self.model.classes_
        predicted_class_val = model_classes[best_class_idx]
        
        # Get activity names
        predicted_activity = self.metadata['activity_map'][int(predicted_class_val)]
        confidence = probs[best_class_idx]
        
        # Retrieve actual label from y_test
        actual_class_val = self.y_test[sample_id]
        actual_activity = self.metadata['activity_map'][int(actual_class_val)]
        
        is_correct = bool(int(predicted_class_val) == int(actual_class_val))
        
        # Top predictions
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
            "confidence": float(confidence),
            "actual": actual_activity,
            "correct": is_correct,
            "model": self.metadata['best_model_name'],
            "top_predictions": top_predictions
        }

    def predict_from_live_stats(self, live_stats):
        """
        Accepts computed aggregate statistics from a live browser sensor window.
        Performs nearest-neighbor search on key UCI feature dimensions derived from the same
        physical quantities (acc_sma, gyr_sma, gravity_x, gravity_y, acc_std_x/y/z).
        The top-k closest UCI test sample vectors are then passed through the trained ML model.
        
        Raw sensor values are NEVER fed directly into the 561-feature model.
        The model always receives real UCI 561-feature vectors.
        """
        # Resolve the specific feature indices we'll use for nearest-neighbor matching
        # These match what the live sensor window computes
        features = self.metadata.get('features', self.desc_engine.features)
        
        # Build the feature indices lookup
        idx_acc_sma = features.index("tBodyAcc-sma()")
        idx_gyr_sma = features.index("tBodyGyro-sma()")
        idx_grav_x  = features.index("tGravityAcc-mean()-X")
        idx_grav_y  = features.index("tGravityAcc-mean()-Y")
        idx_acc_std_x = features.index("tBodyAcc-std()-X")
        idx_acc_std_y = features.index("tBodyAcc-std()-Y")
        idx_acc_std_z = features.index("tBodyAcc-std()-Z")
        
        # Build a compact feature matrix of these 7 dimensions from all X_test rows
        # Shape: (2947, 7)
        X_compact = self.X_test[:, [
            idx_acc_sma, idx_gyr_sma, idx_grav_x, idx_grav_y,
            idx_acc_std_x, idx_acc_std_y, idx_acc_std_z
        ]]
        
        # Normalise the live_stats to the same UCI feature space.
        # UCI features are normalised to [-1, 1] range by their preprocessing.
        # We apply the same clipping to the live window aggregates to keep them comparable.
        def clip_unit(v):
            return max(-1.0, min(1.0, float(v)))
        
        live_vec = np.array([
            clip_unit(live_stats.get('acc_sma',  0)),
            clip_unit(live_stats.get('gyr_sma',  0)),
            clip_unit(live_stats.get('gravity_x', 0)),
            clip_unit(live_stats.get('gravity_y', 0)),
            clip_unit(live_stats.get('acc_std_x', 0)),
            clip_unit(live_stats.get('acc_std_y', 0)),
            clip_unit(live_stats.get('acc_std_z', 0)),
        ])
        
        # Compute Euclidean distances to all test samples on these 7 dimensions
        diffs = X_compact - live_vec
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        
        # Retrieve top-k nearest neighbours
        k = 30
        top_k_indices = np.argsort(distances)[:k]
        
        # Only keep samples whose distance is within a reasonable threshold
        # If the closest match is too far, return None (no sufficient match)
        if distances[top_k_indices[0]] > 2.0:  # max possible distance in 7D normalised space ≈ sqrt(7)*2 ≈ 5.3
            return None
        
        # Retrieve the full 561-feature UCI vectors for the nearest neighbours
        matched_X = self.X_test[top_k_indices]
        
        # Run through trained ML model
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
            "matched_samples": int(k),
            "nearest_distance": float(distances[top_k_indices[0]]),
            "model": self.metadata['best_model_name'],
            "live_stats_received": {k: float(v) for k, v in live_stats.items()}
        }
