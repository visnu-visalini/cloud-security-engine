"""
Flask API and Web Application for Internet Firewall AI Engine
Dataset: log2.csv (Author: Fatih Ertam, Firat University, Turkey)
Model: Random Forest Classifier
"""

import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from data_loader import (
    load_dataset,
    resolve_dataset_path,
    get_dataset_metadata,
    get_sample_records_by_class,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    EXPECTED_CLASSES
)

# Initialize Flask application
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

MODEL_PATH = os.path.join('.', 'models', 'random_forest_firewall.joblib')
METADATA_PATH = os.path.join('.', 'models', 'model_metadata.json')

# Global handles
rf_model = None
model_metadata = None
dataset_cache = None


def get_model():
    """Lazily loads and returns the trained Random Forest model."""
    global rf_model
    if rf_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at '{MODEL_PATH}'. "
                "Please run 'python train_model.py' first."
            )
        rf_model = joblib.load(MODEL_PATH)
        logging.info("Loaded Random Forest model successfully.")
    return rf_model


def get_metadata():
    """Lazily loads and returns the model evaluation metadata."""
    global model_metadata
    if model_metadata is None:
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(
                f"Model metadata not found at '{METADATA_PATH}'. "
                "Please run 'python train_model.py' first."
            )
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            model_metadata = json.load(f)
    return model_metadata


def get_cached_dataset():
    """Loads and caches the real dataset from log2.csv."""
    global dataset_cache
    if dataset_cache is None:
        dataset_cache = load_dataset()
    return dataset_cache


@app.route('/')
def index():
    """Renders the main interactive dashboard."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """System health check and dataset/model status."""
    try:
        dataset_path = resolve_dataset_path()
        dataset_exists = True
    except FileNotFoundError:
        dataset_path = None
        dataset_exists = False

    model_exists = os.path.exists(MODEL_PATH)
    meta_exists = os.path.exists(METADATA_PATH)

    status = "healthy" if (dataset_exists and model_exists and meta_exists) else "degraded"

    return jsonify({
        "status": status,
        "dataset": {
            "found": dataset_exists,
            "path": dataset_path,
            "expected_file": "log2.csv",
            "author": "Fatih Ertam, Firat University, Turkey"
        },
        "model": {
            "trained": model_exists,
            "path": MODEL_PATH,
            "algorithm": "RandomForestClassifier (100 estimators)"
        }
    }), (200 if status == "healthy" else 503)


@app.route('/api/dataset/stats', methods=['GET'])
def dataset_stats():
    """Returns dataset summary metrics and statistical distributions."""
    try:
        df = get_cached_dataset()
        stats = get_dataset_metadata(df)
        return jsonify(stats), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e), "code": "DATASET_MISSING"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to compute dataset stats: {str(e)}"}), 500


@app.route('/api/model/metrics', methods=['GET'])
def model_metrics():
    """Returns model evaluation results, confusion matrix, and feature importances."""
    try:
        meta = get_metadata()
        return jsonify(meta), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e), "code": "MODEL_MISSING"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to load metrics: {str(e)}"}), 500


@app.route('/api/dataset/sample', methods=['GET'])
def dataset_sample():
    """Returns representative and browsable samples from log2.csv."""
    try:
        df = get_cached_dataset()
        mode = request.args.get('mode', 'class_reps')
        limit = int(request.args.get('limit', 10))

        if mode == 'class_reps':
            samples = get_sample_records_by_class(df, n_per_class=1)
            # Add descriptions for dashboard scenarios
            descriptions = {
                'allow': "DNS Resolution on UDP 53 with valid NAT state translation",
                'deny': "Suspicious inbound SMB traffic blocked by firewall access rule",
                'drop': "Silent drop on unmapped external port probe",
                'reset-both': "Host terminated TCP handshake with dual RST packet emission"
            }
            for s in samples:
                act = s.get('Action')
                s['scenario_name'] = f"Actual {act.upper()} Sample"
                s['scenario_description'] = descriptions.get(act, "Network traffic log entry")
            return jsonify({"samples": samples, "total_records": len(df)}), 200
        else:
            # Paginated or slice sample
            subset = df.head(min(limit, 100))
            records = []
            for idx, row in subset.iterrows():
                r = row.to_dict()
                r['row_id'] = int(idx)
                records.append(r)
            return jsonify({"samples": records, "total_records": len(df)}), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e), "code": "DATASET_MISSING"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to load samples: {str(e)}"}), 500


def parse_and_validate_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Validates that all 11 required numeric features are present and numeric."""
    features = {}
    missing = []
    invalid = []

    for col in FEATURE_COLUMNS:
        # Check direct or normalized key name
        val = data.get(col)
        if val is None:
            # Check snake_case or stripped
            alt_key = col.lower().replace(' ', '_').replace('(', '').replace(')', '')
            val = data.get(alt_key)

        if val is None:
            missing.append(col)
        else:
            try:
                features[col] = [float(val)]
            except (ValueError, TypeError):
                invalid.append(col)

    if missing:
        raise ValueError(f"Missing required feature(s): {missing}. Expected: {FEATURE_COLUMNS}")
    if invalid:
        raise ValueError(f"Feature(s) must be numeric: {invalid}")

    return pd.DataFrame(features)[FEATURE_COLUMNS]


def determine_risk_level(action: str, confidence: float) -> Dict[str, str]:
    """Generates a human-friendly risk categorization and firewall explanation."""
    if action == 'allow':
        return {
            "risk_level": "Low / Authorized",
            "badge_color": "success",
            "verdict": "Traffic Allowed",
            "action_advice": "Session matches permitted firewall access policy and NAT state tables."
        }
    elif action == 'deny':
        return {
            "risk_level": "Medium / Policy Rejection",
            "badge_color": "danger",
            "verdict": "Traffic Denied",
            "action_advice": "Connection explicitly rejected by access control list (ACL). RST/ICMP sent to source."
        }
    elif action == 'drop':
        return {
            "risk_level": "High / Silent Discard",
            "badge_color": "warning",
            "verdict": "Traffic Dropped",
            "action_advice": "Packets silently dropped without response. Characteristic of stealth drops, scans, or floods."
        }
    elif action == 'reset-both':
        return {
            "risk_level": "Critical / Connection Reset",
            "badge_color": "purple",
            "verdict": "Reset Both Ends",
            "action_advice": "Firewall actively terminated session by sending TCP RST packets to both client and server."
        }
    return {
        "risk_level": "Unknown",
        "badge_color": "secondary",
        "verdict": action,
        "action_advice": "Firewall state inspected."
    }


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predicts firewall action for a single network log record.
    Returns predicted action, confidence score, and per-class probability distribution.
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    data = request.get_json()

    try:
        input_df = parse_and_validate_input(data)
        model = get_model()

        # Predict
        predicted_action = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        prob_dict = {
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }
        confidence = float(np.max(probabilities))

        risk_info = determine_risk_level(predicted_action, confidence)

        return jsonify({
            "status": "success",
            "predicted_action": predicted_action,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "probabilities": prob_dict,
            "analysis": risk_info,
            "input_features": data
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e), "code": "VALIDATION_ERROR"}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e), "code": "MODEL_MISSING"}), 500
    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Predicts firewall action for an array of network log records."""
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    payload = request.get_json()
    records = payload.get('records') if isinstance(payload, dict) else payload

    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "Payload must contain a non-empty list of 'records'"}), 400

    try:
        model = get_model()
        classes = list(model.classes_)
        results = []

        for idx, rec in enumerate(records):
            input_df = parse_and_validate_input(rec)
            pred = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            confidence = float(np.max(probs))
            results.append({
                "record_index": idx,
                "predicted_action": pred,
                "confidence": round(confidence, 4),
                "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, probs)},
                "risk": determine_risk_level(pred, confidence)
            })

        return jsonify({
            "status": "success",
            "count": len(results),
            "predictions": results
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e), "code": "VALIDATION_ERROR"}), 400
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    # Ensure model and metadata exist on boot
    try:
        get_model()
        get_metadata()
    except Exception as e:
        logging.warning(f"Note: Model not loaded on startup ({e}). Run train_model.py if not done yet.")

    print("\n========================================================")
    print("  FIREWALL AI INSPECTION ENGINE API & DASHBOARD ONLINE")
    print("  Dataset: log2.csv (Fatih Ertam, Firat University)")
    print("  URL: http://127.0.0.1:5000")
    print("========================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
