"""
Random Forest Model Training and Evaluation Pipeline
for Internet Firewall Action Classification
Dataset: log2.csv (Fatih Ertam, Firat University, Turkey)
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from data_loader import (
    load_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    EXPECTED_CLASSES
)

MODEL_DIR = os.path.join('.', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'random_forest_firewall.joblib')
METADATA_FILE = os.path.join(MODEL_DIR, 'model_metadata.json')


def train_and_evaluate(
    n_estimators: int = 100,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a Random Forest Classifier on the actual Internet Firewall dataset,
    evaluates its performance comprehensively, and saves artifacts.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("==================================================")
    print("  FIREWALL ACTION CLASSIFICATION: RF TRAINING")
    print("==================================================")

    # 1. Load real dataset
    print("[1/5] Loading actual dataset via data_loader...")
    start_load = time.time()
    df = load_dataset()
    load_duration = time.time() - start_load
    print(f"      Loaded {len(df)} records in {load_duration:.2f}s")
    print(f"      Features: {FEATURE_COLUMNS}")
    print(f"      Target: '{TARGET_COLUMN}' (Classes: {list(df[TARGET_COLUMN].unique())})")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 2. Stratified train/test split
    print(f"[2/5] Creating stratified train/test split ({int((1-test_size)*100)}/{int(test_size*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"      Training samples: {len(X_train)}")
    print(f"      Testing samples:  {len(X_test)}")

    # 3. Train Random Forest Classifier
    print(f"[3/5] Training RandomForestClassifier (n_estimators={n_estimators}, n_jobs=-1)...")
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )
    start_train = time.time()
    rf_model.fit(X_train, y_train)
    train_duration = time.time() - start_train
    print(f"      Model training completed in {train_duration:.2f}s")

    # 4. Model Evaluation on Held-Out Test Set
    print("[4/5] Evaluating model on test set...")
    start_eval = time.time()
    y_pred = rf_model.predict(X_test)
    eval_duration = time.time() - start_eval

    # Overall Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Detailed Classification Report
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Classes present in model
    classes = list(rf_model.classes_)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_dict = {
        'labels': classes,
        'matrix': cm.tolist()
    }

    # Feature Importances
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_importance_list = [
        {
            'feature': FEATURE_COLUMNS[i],
            'importance': round(float(importances[i]), 5),
            'percentage': round(float(importances[i]) * 100, 2)
        }
        for i in sorted_idx
    ]

    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy:          {acc * 100:.3f}%")
    print(f"Macro F1-Score:    {f1_macro * 100:.3f}%")
    print(f"Weighted F1-Score: {f1_weighted * 100:.3f}%")
    print("\nPer-class breakdown:")
    for cls in classes:
        cls_data = clf_report.get(cls, {})
        print(f"  - {cls.ljust(10)}: Precision={cls_data.get('precision', 0)*100:.2f}%, "
              f"Recall={cls_data.get('recall', 0)*100:.2f}%, F1={cls_data.get('f1-score', 0)*100:.2f}%, "
              f"Support={cls_data.get('support', 0)}")

    print("\nTop 5 Most Important Features:")
    for fi in feature_importance_list[:5]:
        print(f"  1. {fi['feature'].ljust(22)}: {fi['percentage']}%")

    # 5. Save Artifacts
    print(f"\n[5/5] Saving model artifacts to '{MODEL_DIR}'...")
    joblib.dump(rf_model, MODEL_FILE, compress=3)
    print(f"      Saved model: {MODEL_FILE}")

    metadata = {
        'model_name': 'Random Forest Classifier',
        'n_estimators': n_estimators,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': FEATURE_COLUMNS,
        'classes': classes,
        'metrics': {
            'accuracy': round(float(acc), 5),
            'precision_macro': round(float(prec_macro), 5),
            'recall_macro': round(float(rec_macro), 5),
            'f1_macro': round(float(f1_macro), 5),
            'precision_weighted': round(float(prec_weighted), 5),
            'recall_weighted': round(float(rec_weighted), 5),
            'f1_weighted': round(float(f1_weighted), 5)
        },
        'per_class_metrics': {
            cls: {
                'precision': round(float(clf_report[cls]['precision']), 4),
                'recall': round(float(clf_report[cls]['recall']), 4),
                'f1_score': round(float(clf_report[cls]['f1-score']), 4),
                'support': int(clf_report[cls]['support'])
            }
            for cls in classes if cls in clf_report
        },
        'confusion_matrix': cm_dict,
        'feature_importance': feature_importance_list,
        'training_duration_seconds': round(train_duration, 2),
        'evaluation_duration_seconds': round(eval_duration, 4),
        'dataset_info': {
            'total_records': len(df),
            'author': "Fatih Ertam, Firat University, Turkey",
            'file': "log2.csv"
        }
    }

    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"      Saved metadata: {METADATA_FILE}")
    print("==================================================")
    print("  TRAINING & EVALUATION COMPLETE")
    print("==================================================")

    return metadata


if __name__ == '__main__':
    train_and_evaluate()
