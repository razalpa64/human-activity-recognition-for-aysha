import os
import numpy as np

def find_dataset_path():
    """
    Finds the UCI HAR Dataset path in common locations.
    Returns the absolute path, or raises FileNotFoundError.
    """
    candidates = [
        "dataset/UCI HAR Dataset",
        "../dataset/UCI HAR Dataset",
        "../../dataset/UCI HAR Dataset",
        "Human_Activity_Recognition/dataset/UCI HAR Dataset",
        "../Human_Activity_Recognition/dataset/UCI HAR Dataset",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError("UCI HAR Dataset directory not found in candidate paths.")

def load_features_list(dataset_path):
    """
    Loads feature names from features.txt.
    """
    features_path = os.path.join(dataset_path, "features.txt")
    features = []
    with open(features_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                # Store the feature name
                features.append(parts[1])
            else:
                features.append(line.strip())
    return features

class DescriptorEngine:
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            dataset_path = find_dataset_path()
        self.dataset_path = dataset_path
        self.features = load_features_list(self.dataset_path)
        
        # Resolve feature indices
        try:
            self.acc_sma_idx = self.features.index("tBodyAcc-sma()")
            self.jerk_sma_idx = self.features.index("tBodyAccJerk-sma()")
            self.gyro_sma_idx = self.features.index("tBodyGyro-sma()")
            self.grav_x_idx = self.features.index("tGravityAcc-mean()-X")
            self.grav_y_idx = self.features.index("tGravityAcc-mean()-Y")
            self.ar_coeff_x_idx = self.features.index("tGravityAcc-arCoeff()-X,1")
        except ValueError as e:
            raise KeyError(f"Required feature missing from features list: {str(e)}")

    def extract_from_vector(self, X_row):
        """
        Extracts descriptors from a 561-feature row.
        Returns a dictionary of descriptors.
        """
        # 1. MOVEMENT INTENSITY (Low, Medium, High)
        acc_sma_val = X_row[self.acc_sma_idx]
        if acc_sma_val < -0.9:
            intensity = 'Low'
        elif acc_sma_val < -0.15:
            intensity = 'Medium'
        else:
            intensity = 'High'

        # 2. MOVEMENT STABILITY (Low, Medium, High)
        jerk_sma_val = X_row[self.jerk_sma_idx]
        if jerk_sma_val < -0.95:
            stability = 'High'
        elif jerk_sma_val < -0.2:
            stability = 'Medium'
        else:
            stability = 'Low'

        # 3. BODY POSITION (Lying, Sitting, Standing / Upright)
        gx = X_row[self.grav_x_idx]
        gy = X_row[self.grav_y_idx]
        if gx < 0.25:
            body_pos = 'Lying'
        elif gy > 0.05:
            body_pos = 'Sitting'
        else:
            body_pos = 'Standing'

        # 4. ROTATION (Low, Moderate, High)
        gyro_sma_val = X_row[self.gyro_sma_idx]
        if gyro_sma_val < -0.9:
            rotation = 'Low'
        elif gyro_sma_val < -0.2:
            rotation = 'Moderate'
        else:
            rotation = 'High'

        # 5. MOVEMENT PATTERN (Still, Regular, Rhythmic)
        if intensity == 'Low':
            pattern = 'Still'
        else:
            ar_val = X_row[self.ar_coeff_x_idx]
            if ar_val >= -0.52:
                pattern = 'Rhythmic'
            else:
                pattern = 'Regular'

        return {
            'intensity': intensity,
            'stability': stability,
            'body_position': body_pos,
            'rotation': rotation,
            'movement_pattern': pattern
        }

    def build_test_index(self, X_test):
        """
        Builds a searchable index mapping each test sample index to its descriptors.
        """
        index = []
        for i in range(len(X_test)):
            desc = self.extract_from_vector(X_test[i])
            desc['sample_id'] = i
            index.append(desc)
        return index
