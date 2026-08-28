import numpy as np
from scipy.signal import butter, filtfilt

# Reference gravity vector during standing in the UCI dataset training reference frame
TRAIN_STANDING_GRAVITY = np.array([0.9430911, -0.18963752, 0.02040092])

def compute_rotation_matrix(v_from, v_to=TRAIN_STANDING_GRAVITY):
    """
    Computes 3x3 rotation matrix R that rotates 3D vector v_from to v_to.
    Both vectors are normalized internally. Uses Rodrigues' rotation formula.
    """
    norm_from = np.linalg.norm(v_from)
    norm_to = np.linalg.norm(v_to)
    if norm_from == 0 or norm_to == 0:
        return np.eye(3)

    a = v_from / norm_from
    b = v_to / norm_to

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-6:
        if c > 0:
            return np.eye(3)
        else:
            orthogonal = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            v = np.cross(a, orthogonal)
            v = v / np.linalg.norm(v)
            return 2.0 * np.outer(v, v) - np.eye(3)

    v_x = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + v_x + np.matmul(v_x, v_x) * ((1.0 - c) / (s ** 2))
    return R

def butter_lowpass_filter(data, cutoff, fs, order=3):
    """
    Applies Butterworth low-pass filter to a signal array (N x C).
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    padlen = min(6, data.shape[0] - 1)
    y = filtfilt(b, a, data, axis=0, padlen=padlen)
    return y

def extract_signal_features(signal, prefix):
    """
    Extracts time-domain features from a 3D signal (N x 3) or 1D signal (N,).
    """
    features = {}
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    n_axes = signal.shape[1]
    axis_names = ['X', 'Y', 'Z'] if n_axes == 3 else ['Mag']

    # 1. Mean
    means = np.mean(signal, axis=0)
    for i in range(n_axes):
        features[f"{prefix}-mean()-{axis_names[i]}" if n_axes == 3 else f"{prefix}-mean()"] = float(means[i])

    # 2. Std
    stds = np.std(signal, axis=0, ddof=1) if signal.shape[0] > 1 else np.zeros(n_axes)
    for i in range(n_axes):
        features[f"{prefix}-std()-{axis_names[i]}" if n_axes == 3 else f"{prefix}-std()"] = float(stds[i])

    # 3. MAD
    mads = np.median(np.abs(signal - np.median(signal, axis=0)), axis=0)
    for i in range(n_axes):
        features[f"{prefix}-mad()-{axis_names[i]}" if n_axes == 3 else f"{prefix}-mad()"] = float(mads[i])

    # 4. Max
    maxs = np.max(signal, axis=0)
    for i in range(n_axes):
        features[f"{prefix}-max()-{axis_names[i]}" if n_axes == 3 else f"{prefix}-max()"] = float(maxs[i])

    # 5. Min
    mins = np.min(signal, axis=0)
    for i in range(n_axes):
        features[f"{prefix}-min()-{axis_names[i]}" if n_axes == 3 else f"{prefix}-min()"] = float(mins[i])

    # 6. SMA
    if n_axes == 3:
        sma = np.mean(np.sum(np.abs(signal), axis=1))
        features[f"{prefix}-sma()"] = float(sma)
    else:
        sma = np.mean(np.abs(signal))
        features[f"{prefix}-sma()"] = float(sma)

    # 7. Energy
    energy = np.mean(signal ** 2, axis=0)
    for i in range(n_axes):
        features[f"{prefix}-energy()-{axis_names[i]}" if n_axes == 3 else f"{prefix}-energy()"] = float(energy[i])

    # 8. IQR
    iqr = np.percentile(signal, 75, axis=0) - np.percentile(signal, 25, axis=0)
    for i in range(n_axes):
        features[f"{prefix}-iqr()-{axis_names[i]}" if n_axes == 3 else f"{prefix}-iqr()"] = float(iqr[i])

    # 9. Correlation (only for 3D)
    if n_axes == 3:
        for idx1, idx2, pair in [(0, 1, 'X,Y'), (0, 2, 'X,Z'), (1, 2, 'Y,Z')]:
            c = np.corrcoef(signal[:, idx1], signal[:, idx2])[0, 1]
            features[f"{prefix}-correlation()-{pair}"] = float(c if not np.isnan(c) else 0.0)

    return features

def extract_fft_features(signal, prefix):
    """
    Computes FFT spectral features for 3D signal (N x 3) or 1D signal (N,).
    """
    features = {}
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    n_axes = signal.shape[1]
    axis_names = ['X', 'Y', 'Z'] if n_axes == 3 else ['Mag']

    fft_vals = np.abs(np.fft.rfft(signal, axis=0))

    means = np.mean(fft_vals, axis=0)
    for i in range(n_axes):
        features[f"f{prefix}-mean()-{axis_names[i]}" if n_axes == 3 else f"f{prefix}-mean()"] = float(means[i])

    stds = np.std(fft_vals, axis=0)
    for i in range(n_axes):
        features[f"f{prefix}-std()-{axis_names[i]}" if n_axes == 3 else f"f{prefix}-std()"] = float(stds[i])

    energy = np.mean(fft_vals ** 2, axis=0)
    for i in range(n_axes):
        features[f"f{prefix}-energy()-{axis_names[i]}" if n_axes == 3 else f"f{prefix}-energy()"] = float(energy[i])

    psd = fft_vals ** 2
    sum_psd = np.sum(psd, axis=0, keepdims=True)
    sum_psd[sum_psd == 0] = 1e-12
    p = psd / sum_psd
    entropy = -np.sum(p * np.log2(p + 1e-12), axis=0)
    for i in range(n_axes):
        features[f"f{prefix}-entropy()-{axis_names[i]}" if n_axes == 3 else f"f{prefix}-entropy()"] = float(entropy[i])

    max_inds = np.argmax(fft_vals, axis=0)
    for i in range(n_axes):
        features[f"f{prefix}-maxInds-{axis_names[i]}" if n_axes == 3 else f"f{prefix}-maxInds"] = float(max_inds[i])

    freqs = np.linspace(0, 1, fft_vals.shape[0])
    mean_freq = np.sum(fft_vals * freqs[:, None], axis=0) / (np.sum(fft_vals, axis=0) + 1e-12)
    for i in range(n_axes):
        features[f"f{prefix}-meanFreq()-{axis_names[i]}" if n_axes == 3 else f"f{prefix}-meanFreq()"] = float(mean_freq[i])

    return features

def extract_window_features(total_acc, body_acc, body_gyro, fs=50):
    """
    Extracts complete feature dictionary from a 128-sample window.
    """
    body_acc_jerk = np.gradient(body_acc, axis=0) * fs
    body_gyro_jerk = np.gradient(body_gyro, axis=0) * fs

    total_acc_mag = np.linalg.norm(total_acc, axis=1)
    body_acc_mag = np.linalg.norm(body_acc, axis=1)
    gravity_acc_mag = np.linalg.norm(total_acc - body_acc, axis=1)
    body_acc_jerk_mag = np.linalg.norm(body_acc_jerk, axis=1)
    body_gyro_mag = np.linalg.norm(body_gyro, axis=1)
    body_gyro_jerk_mag = np.linalg.norm(body_gyro_jerk, axis=1)

    all_features = {}

    all_features.update(extract_signal_features(body_acc, "tBodyAcc"))
    all_features.update(extract_signal_features(total_acc - body_acc, "tGravityAcc"))
    all_features.update(extract_signal_features(body_acc_jerk, "tBodyAccJerk"))
    all_features.update(extract_signal_features(body_gyro, "tBodyGyro"))
    all_features.update(extract_signal_features(body_gyro_jerk, "tBodyGyroJerk"))

    all_features.update(extract_signal_features(body_acc_mag, "tBodyAccMag"))
    all_features.update(extract_signal_features(gravity_acc_mag, "tGravityAccMag"))
    all_features.update(extract_signal_features(body_acc_jerk_mag, "tBodyAccJerkMag"))
    all_features.update(extract_signal_features(body_gyro_mag, "tBodyGyroMag"))
    all_features.update(extract_signal_features(body_gyro_jerk_mag, "tBodyGyroJerkMag"))

    all_features.update(extract_fft_features(body_acc, "BodyAcc"))
    all_features.update(extract_fft_features(body_acc_jerk, "BodyAccJerk"))
    all_features.update(extract_fft_features(body_gyro, "BodyGyro"))
    all_features.update(extract_fft_features(body_acc_mag, "BodyAccMag"))
    all_features.update(extract_fft_features(body_acc_jerk_mag, "BodyAccJerkMag"))
    all_features.update(extract_fft_features(body_gyro_mag, "BodyGyroMag"))
    all_features.update(extract_fft_features(body_gyro_jerk_mag, "BodyGyroJerkMag"))

    grav = total_acc - body_acc
    mean_grav = np.mean(grav, axis=0)
    mean_body_acc = np.mean(body_acc, axis=0)

    norm_mg = np.linalg.norm(mean_grav) + 1e-12
    norm_mba = np.linalg.norm(mean_body_acc) + 1e-12

    angle_body_acc_grav = np.arccos(np.clip(np.dot(mean_body_acc, mean_grav) / (norm_mba * norm_mg), -1.0, 1.0))
    all_features["angle(tBodyAccMean,gravity)"] = float(angle_body_acc_grav)

    for i, ax in enumerate(['X', 'Y', 'Z']):
        vec = np.zeros(3)
        vec[i] = 1.0
        angle_ax = np.arccos(np.clip(np.dot(vec, mean_grav) / norm_mg, -1.0, 1.0))
        all_features[f"angle({ax},gravityMean)"] = float(angle_ax)

    return all_features

def process_raw_window(acc_raw, gyro_raw, fs=50, rotation_matrix=None):
    """
    Takes raw accelerometer (N x 3) and raw gyroscope (N x 3).
    If rotation_matrix is provided, rotates both 3D signals.
    Applies noise filter (20 Hz lowpass) and gravity filter (0.3 Hz lowpass).
    Returns 138-dimensional feature vector as a numpy 1D array, and feature names list.
    """
    acc = np.array(acc_raw, dtype=float)
    gyro = np.array(gyro_raw, dtype=float)

    # Convert m/s^2 to g if needed
    if np.max(np.abs(acc)) > 4.0:
        acc = acc / 9.80665

    if rotation_matrix is not None:
        R = np.array(rotation_matrix, dtype=float)
        acc = np.matmul(acc, R.T)
        gyro = np.matmul(gyro, R.T)

    total_acc_filtered = butter_lowpass_filter(acc, cutoff=20.0, fs=fs, order=3)
    body_gyro_filtered = butter_lowpass_filter(gyro, cutoff=20.0, fs=fs, order=3)

    gravity_acc = butter_lowpass_filter(total_acc_filtered, cutoff=0.3, fs=fs, order=3)
    body_acc = total_acc_filtered - gravity_acc

    feature_dict = extract_window_features(total_acc_filtered, body_acc, body_gyro_filtered, fs=fs)
    feature_names = list(feature_dict.keys())
    feature_vector = np.array([feature_dict[k] for k in feature_names], dtype=float)

    return feature_vector, feature_names
