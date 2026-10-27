"""
Comprehensive Automated Test Suite for Internet Firewall AI Pipeline
Tests dataset loading, attribute verification, model evaluation, and Flask REST API.
Dataset: log2.csv (Author: Fatih Ertam, Firat University, Turkey)
"""

import os
import json
import pytest
import pandas as pd

from data_loader import (
    load_dataset,
    resolve_dataset_path,
    get_dataset_metadata,
    get_sample_records_by_class,
    EXPECTED_ATTRIBUTES,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    EXPECTED_CLASSES
)
from app import app, parse_and_validate_input, get_model, get_metadata


def test_dataset_file_and_locations():
    """Verify that log2.csv exists in preferred or supported locations."""
    path = resolve_dataset_path()
    assert os.path.exists(path), f"Resolved path '{path}' does not exist on disk."
    assert "log2.csv" in path.lower()


def test_dataset_missing_path_raises_informative_error():
    """Verify that an invalid path fails with FileNotFoundError and explicit instructions."""
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_dataset_path(custom_path="./non_existent_folder/missing.csv")
    err_msg = str(exc_info.value)
    assert "Dataset 'log2.csv' not found" in err_msg
    assert "Fatih Ertam" in err_msg
    assert "Never use synthetic or fabricated data" in err_msg


def test_dataset_attributes_and_shape():
    """Verify that all 12 attributes exist, Action is target, and records are loaded."""
    df = load_dataset()
    assert len(df) == 65532, f"Expected 65,532 records, found {len(df)}"
    assert len(df.columns) == 12, f"Expected 12 columns, found {len(df.columns)}"

    for attr in EXPECTED_ATTRIBUTES:
        assert attr in df.columns, f"Attribute '{attr}' missing from dataset."

    assert TARGET_COLUMN in df.columns
    unique_actions = set(df[TARGET_COLUMN].unique())
    for exp_cls in EXPECTED_CLASSES:
        assert exp_cls in unique_actions, f"Expected class '{exp_cls}' missing from target."


def test_dataset_metadata_computation():
    """Verify metadata and summary statistics."""
    meta = get_dataset_metadata()
    assert meta['total_records'] == 65532
    assert meta['dataset_author'] == "Fatih Ertam, Firat University, Turkey"
    assert len(meta['feature_columns']) == 11
    assert 'allow' in meta['class_distribution']
    assert meta['class_distribution']['allow'] == 37640


def test_model_and_metadata_files_exist():
    """Verify trained Random Forest model and metadata exist."""
    model_path = os.path.join('.', 'models', 'random_forest_firewall.joblib')
    meta_path = os.path.join('.', 'models', 'model_metadata.json')
    assert os.path.exists(model_path), f"Model file '{model_path}' not found. Run train_model.py."
    assert os.path.exists(meta_path), f"Metadata file '{meta_path}' not found."

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    assert meta['metrics']['accuracy'] >= 0.99, f"Accuracy {meta['metrics']['accuracy']} below 99%."
    assert len(meta['feature_importance']) == 11
    assert meta['confusion_matrix'] is not None


def test_flask_health_endpoint():
    """Test /api/health endpoint."""
    with app.test_client() as client:
        res = client.get('/api/health')
        assert res.status_code == 200
        data = res.get_json()
        assert data['status'] == "healthy"
        assert data['dataset']['found'] is True
        assert data['model']['trained'] is True


def test_flask_dataset_stats_endpoint():
    """Test /api/dataset/stats endpoint."""
    with app.test_client() as client:
        res = client.get('/api/dataset/stats')
        assert res.status_code == 200
        data = res.get_json()
        assert data['total_records'] == 65532
        assert "Fatih Ertam" in data['dataset_author']


def test_flask_model_metrics_endpoint():
    """Test /api/model/metrics endpoint."""
    with app.test_client() as client:
        res = client.get('/api/model/metrics')
        assert res.status_code == 200
        data = res.get_json()
        assert 'metrics' in data
        assert data['metrics']['accuracy'] > 0.99
        assert 'confusion_matrix' in data


def test_flask_dataset_sample_endpoint():
    """Test /api/dataset/sample endpoint."""
    with app.test_client() as client:
        res = client.get('/api/dataset/sample?mode=class_reps')
        assert res.status_code == 200
        data = res.get_json()
        assert 'samples' in data
        assert len(data['samples']) == 4


def test_flask_predict_endpoint_valid():
    """Test /api/predict with a real legitimate traffic sample."""
    sample_payload = {
        "Source Port": 57222,
        "Destination Port": 53,
        "NAT Source Port": 57222,
        "NAT Destination Port": 53,
        "Bytes": 177,
        "Bytes Sent": 94,
        "Bytes Received": 83,
        "Packets": 2,
        "Elapsed Time (sec)": 30,
        "pkts_sent": 1,
        "pkts_received": 1
    }
    with app.test_client() as client:
        res = client.post('/api/predict', json=sample_payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data['status'] == "success"
        assert data['predicted_action'] in EXPECTED_CLASSES
        assert 'confidence' in data
        assert 'probabilities' in data
        assert 'analysis' in data


def test_flask_predict_endpoint_invalid():
    """Test /api/predict handles missing features properly."""
    invalid_payload = {
        "Source Port": 57222
        # Missing other 10 features
    }
    with app.test_client() as client:
        res = client.post('/api/predict', json=invalid_payload)
        assert res.status_code == 400
        data = res.get_json()
        assert 'error' in data
        assert data['code'] == "VALIDATION_ERROR"


def test_flask_predict_batch():
    """Test /api/predict/batch with multiple samples."""
    samples = [
        {
            "Source Port": 57222, "Destination Port": 53,
            "NAT Source Port": 57222, "NAT Destination Port": 53,
            "Bytes": 177, "Bytes Sent": 94, "Bytes Received": 83,
            "Packets": 2, "Elapsed Time (sec)": 30,
            "pkts_sent": 1, "pkts_received": 1
        },
        {
            "Source Port": 56258, "Destination Port": 3389,
            "NAT Source Port": 56258, "NAT Destination Port": 3389,
            "Bytes": 4760, "Bytes Sent": 3370, "Bytes Received": 1390,
            "Packets": 19, "Elapsed Time (sec)": 17,
            "pkts_sent": 10, "pkts_received": 9
        }
    ]
    with app.test_client() as client:
        res = client.post('/api/predict/batch', json={"records": samples})
        assert res.status_code == 200
        data = res.get_json()
        assert data['status'] == "success"
        assert data['count'] == 2
        assert len(data['predictions']) == 2


if __name__ == '__main__':
    pytest.main(["-v", "test_pipeline.py"])
