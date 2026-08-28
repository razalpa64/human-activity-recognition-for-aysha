import os
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from prediction import Predictor
from train_model import train_and_evaluate
from features import compute_rotation_matrix, TRAIN_STANDING_GRAVITY

app = Flask(__name__, static_folder="../frontend", static_url_path="")

predictor = None
init_error = None

def init_predictor():
    global predictor, init_error
    try:
        predictor = Predictor()
        init_error = None
        print("Predictor initialized successfully.")
    except Exception as e:
        predictor = None
        init_error = str(e)
        print(f"Failed to initialize predictor: {e}")
        traceback.print_exc()

init_predictor()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    global predictor, init_error
    dataset_verified = False
    try:
        from preprocessing import find_dataset_path
        find_dataset_path()
        dataset_verified = True
    except FileNotFoundError:
        dataset_verified = False

    if predictor is not None:
        return jsonify({
            "status": "ready",
            "model_loaded": True,
            "dataset_verified": dataset_verified,
            "error": None
        })
    else:
        model_exists = os.path.exists(os.path.join(app.root_path, "..", "models", "har_model.pkl"))
        return jsonify({
            "status": "not_initialized",
            "model_loaded": model_exists,
            "dataset_verified": dataset_verified,
            "error": init_error or "Trained model files not found. Please train the model."
        })

@app.route('/api/model', methods=['GET'])
def get_model_metadata():
    global predictor
    if predictor is None or predictor.metadata is None:
        return jsonify({"error": "Model metadata is not loaded. Train the model first."}), 404

    meta = dict(predictor.metadata)
    if 'features' in meta:
        meta['features_preview'] = meta['features'][:10]
        meta['features_count'] = len(meta['features'])
        del meta['features']

    return jsonify(meta)

@app.route('/api/predict', methods=['POST'])
def predict_questionnaire():
    global predictor
    if predictor is None:
        return jsonify({"error": "Predictor is not initialized. Train the model first."}), 503

    data = request.get_json() or {}
    intensity = data.get('intensity')
    stability = data.get('stability')
    body_position = data.get('body_position')
    rotation = data.get('rotation')
    movement_pattern = data.get('movement_pattern')

    required = [intensity, stability, body_position, rotation, movement_pattern]
    if any(v is None for v in required):
        return jsonify({"error": "Invalid input. All 5 descriptors are required."}), 400

    try:
        res = predictor.predict_questionnaire(
            intensity=intensity,
            stability=stability,
            body_position=body_position,
            rotation=rotation,
            movement_pattern=movement_pattern
        )
        if res is None:
            return jsonify({
                "error": "INSUFFICIENT_MATCHES",
                "message": "No matching recordings found."
            }), 200
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500

@app.route('/api/sample/<int:sample_id>', methods=['GET'])
def get_sample_prediction(sample_id):
    global predictor
    if predictor is None:
        return jsonify({"error": "Predictor is not initialized. Train the model first."}), 503

    try:
        res = predictor.predict_sample(sample_id)
        return jsonify(res)
    except IndexError as e:
        return jsonify({"error": "Invalid sample ID", "message": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500

@app.route('/api/activities', methods=['GET'])
def get_activities():
    activities = [
        {"id": "WALKING", "name": "Walking", "icon": "🚶", "description": "Dynamic movement with a cyclic, rhythmic gait, characterized by regular acceleration peaks."},
        {"id": "WALKING_UPSTAIRS", "name": "Walking Upstairs", "icon": "⬆️", "description": "Vigorous upward traversal showing elevated vertical acceleration."},
        {"id": "WALKING_DOWNSTAIRS", "name": "Walking Downstairs", "icon": "⬇️", "description": "Descent with high impact acceleration peaks and lower stability."},
        {"id": "SITTING", "name": "Sitting", "icon": "🪑", "description": "Static seated posture marked by negligible body acceleration and stable gravity alignment."},
        {"id": "STANDING", "name": "Standing", "icon": "🧍", "description": "Static upright posture with minimal motion amplitude."},
        {"id": "LAYING", "name": "Lying", "icon": "🛏️", "description": "Resting horizontal posture with gravity vector aligned along horizontal phone axes."}
    ]
    return jsonify(activities)

@app.route('/api/results', methods=['GET'])
def get_results_summary():
    results_csv_path = os.path.join(app.root_path, "..", "results", "final_results.csv")
    comparison = []
    if os.path.exists(results_csv_path):
        import csv
        with open(results_csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                comparison.append({
                    "model": row["Model"],
                    "accuracy": float(row["Accuracy"]),
                    "precision": float(row["Precision"]),
                    "recall": float(row["Recall"]),
                    "f1_score": float(row["F1_Score"]),
                    "training_time": float(row["Training_Time_s"]),
                    "prediction_time": float(row["Prediction_Time_s"])
                })
    return jsonify({
        "comparison": comparison,
        "plots": {
            "confusion_matrix": "/results/confusion_matrix.png",
            "model_comparison": "/results/model_comparison.png",
            "activity_distribution": "/results/activity_distribution.png"
        }
    })

@app.route('/results/<path:filename>')
def serve_results(filename):
    results_dir = os.path.abspath(os.path.join(app.root_path, "..", "results"))
    return send_from_directory(results_dir, filename)

@app.route('/api/calibrate', methods=['POST'])
def calibrate_orientation():
    """
    Accepts gravity vector from user standing/upright position:
    { "x": float, "y": float, "z": float }
    Computes and returns the 3x3 rotation matrix R to align with training standing orientation.
    """
    data = request.get_json() or {}
    x = data.get('x', 0.0)
    y = data.get('y', 0.0)
    z = data.get('z', 0.0)

    v_gravity = np.array([x, y, z], dtype=float)
    R = compute_rotation_matrix(v_from=v_gravity, v_to=TRAIN_STANDING_GRAVITY)

    return jsonify({
        "status": "success",
        "rotation_matrix": R.tolist(),
        "calibrated_gravity": (np.dot(R, v_gravity / np.linalg.norm(v_gravity))).tolist() if np.linalg.norm(v_gravity) > 0 else [0, 0, 0]
    })

@app.route('/api/live-predict', methods=['POST'])
def predict_live_sensor():
    """
    Accepts raw window of accelerometer and gyroscope readings captured in the browser.
    Expected JSON:
    {
        "acc": [ {"x": float, "y": float, "z": float}, ... ],
        "gyro": [ {"x": float, "y": float, "z": float}, ... ],
        "rotation_matrix": [[...], [...], [...]] (optional)
    }
    """
    global predictor
    if predictor is None:
        return jsonify({"error": "Predictor not initialized. Train the model first."}), 503

    data = request.get_json() or {}

    acc_list = data.get('acc', [])
    gyro_list = data.get('gyro', [])
    rotation_matrix = data.get('rotation_matrix', None)

    if not acc_list or len(acc_list) < 10:
        return jsonify({"error": "INSUFFICIENT_SAMPLES", "message": "Need at least 10 samples in window."}), 400

    # Format arrays
    acc_raw = np.array([[s.get('x', 0), s.get('y', 0), s.get('z', 0)] for s in acc_list], dtype=float)
    if gyro_list and len(gyro_list) == len(acc_list):
        gyro_raw = np.array([[s.get('x', 0), s.get('y', 0), s.get('z', 0)] for s in gyro_list], dtype=float)
    else:
        # Fallback if gyro not supplied
        gyro_raw = np.zeros_like(acc_raw)

    # Pad or truncate window to 128 samples if needed
    target_samples = 128
    if len(acc_raw) != target_samples:
        t_orig = np.linspace(0, 1, len(acc_raw))
        t_target = np.linspace(0, 1, target_samples)
        acc_interp = np.zeros((target_samples, 3))
        gyro_interp = np.zeros((target_samples, 3))
        for col in range(3):
            acc_interp[:, col] = np.interp(t_target, t_orig, acc_raw[:, col])
            gyro_interp[:, col] = np.interp(t_target, t_orig, gyro_raw[:, col])
        acc_raw = acc_interp
        gyro_raw = gyro_interp

    try:
        result = predictor.predict_from_raw_window(acc_raw, gyro_raw, rotation_matrix=rotation_matrix)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Live prediction failed", "message": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def run_training():
    try:
        success = train_and_evaluate()
        if success:
            init_predictor()
            return jsonify({"status": "success", "message": "Model retrained successfully."})
        else:
            return jsonify({"status": "error", "message": "Model training failed."}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
