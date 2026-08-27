import os
import traceback
from flask import Flask, request, jsonify, send_from_directory
from prediction import Predictor
from train_model import train_and_evaluate

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# Global Predictor instance
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

# Initialize predictor at startup
init_predictor()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    global predictor, init_error
    
    # Verify dataset exists
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
        # Check if the files are just missing
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
    
    # We strip out the list of features to avoid huge JSON payload unless requested
    meta = dict(predictor.metadata)
    if 'features' in meta:
        meta['features_preview'] = meta['features'][:10]
        # Keep features count, but exclude the full 561 features list to save bandwidth
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
    
    # Validate inputs
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
                "message": "No sufficiently similar real UCI sensor recordings were found. Try changing one or more characteristics."
            }), 200 # Return 200 with error code to handle gracefully
            
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
        {"id": "WALKING", "name": "Walking", "icon": "🚶", "description": "Dynamic movement with a cyclic, rhythmic gait, typically characterized by regular horizontal acceleration peaks."},
        {"id": "WALKING_UPSTAIRS", "name": "Walking Upstairs", "icon": "⬆️", "description": "Vigorous upward traversal, showing elevated vertical acceleration and moderate rotation instability."},
        {"id": "WALKING_DOWNSTAIRS", "name": "Walking Downstairs", "icon": "⬇️", "description": "Descent with high impact acceleration peaks, exhibiting lower stability and rhythmic rotation patterns."},
        {"id": "SITTING", "name": "Sitting", "icon": "🪑", "description": "Static seated posture with the phone held vertical or tilted, marked by negligible body acceleration and stable gravity alignment."},
        {"id": "STANDING", "name": "Standing", "icon": "🧍", "description": "Static upright posture, exhibiting minimal motion intensity and highly stable gravity alignment."},
        {"id": "LAYING", "name": "Lying", "icon": "🛏️", "description": "Resting horizontal posture, characterized by distinct gravity orientation along the phone's horizontal axes and lack of motion."}
    ]
    return jsonify(activities)

@app.route('/api/results', methods=['GET'])
def get_results_summary():
    # Load comparison results from CSV if it exists
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

@app.route('/api/live-predict', methods=['POST'])
def predict_live_sensor():
    """
    Accepts computed aggregate statistics from a live sensor window captured in the browser.
    These statistics are used to find the nearest matching UCI test observations, which are
    then passed through the trained ML model. Raw sensor values never enter the model.
    
    Expected JSON body:
    {
        "acc_mean_x": float, "acc_mean_y": float, "acc_mean_z": float,
        "acc_std_x": float, "acc_std_y": float, "acc_std_z": float,
        "gyr_mean_x": float, "gyr_mean_y": float, "gyr_mean_z": float,
        "gyr_std_x": float, "gyr_std_y": float, "gyr_std_z": float,
        "acc_sma": float,
        "gyr_sma": float,
        "gravity_x": float, "gravity_y": float, "gravity_z": float
    }
    """
    global predictor
    if predictor is None:
        return jsonify({"error": "Predictor not initialized. Train the model first."}), 503

    data = request.get_json() or {}

    required_fields = [
        'acc_sma', 'gyr_sma', 'gravity_x', 'gravity_y',
        'acc_std_x', 'acc_std_y', 'acc_std_z'
    ]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        result = predictor.predict_from_live_stats(data)
        if result is None:
            return jsonify({
                "error": "INSUFFICIENT_MATCHES",
                "message": "No similar UCI recordings found for this sensor window. Try moving for a longer period."
            }), 200
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Live prediction failed", "message": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def run_training():
    try:
        success = train_and_evaluate()
        if success:
            # Re-initialize predictor to load the new model and metadata
            init_predictor()
            return jsonify({"status": "success", "message": "Model retrained successfully."})
        else:
            return jsonify({"status": "error", "message": "Model training failed. Check dataset path."}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Host on all interfaces on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
