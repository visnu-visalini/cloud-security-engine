"""
Dataset Loader and Verification Module for Internet Firewall Dataset
Author: Fatih Ertam, Firat University, Turkey
CSV File: log2.csv (~2.88 MB, 65,532 records, 12 attributes)
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Expected 12 attributes in the dataset
EXPECTED_ATTRIBUTES: List[str] = [
    'Source Port',
    'Destination Port',
    'NAT Source Port',
    'NAT Destination Port',
    'Action',
    'Bytes',
    'Bytes Sent',
    'Bytes Received',
    'Packets',
    'Elapsed Time (sec)',
    'pkts_sent',
    'pkts_received'
]

TARGET_COLUMN: str = 'Action'
FEATURE_COLUMNS: List[str] = [col for col in EXPECTED_ATTRIBUTES if col != TARGET_COLUMN]

EXPECTED_CLASSES: List[str] = ['allow', 'deny', 'drop', 'reset-both']

# Candidate paths in order of preference
CANDIDATE_PATHS: List[str] = [
    os.path.join('.', 'data', 'log2.csv'),
    os.path.join('.', 'log2.csv'),
]


def resolve_dataset_path(custom_path: Optional[str] = None) -> str:
    """
    Resolves the actual path to log2.csv.
    Checks custom path if provided, then ./data/log2.csv, then ./log2.csv.
    Fails with a clear, informative error message if missing.
    """
    if custom_path is not None:
        if os.path.exists(custom_path):
            return custom_path
        raise FileNotFoundError(
            f"CRITICAL ERROR: Dataset 'log2.csv' not found at requested path: '{custom_path}'\n\n"
            "Dataset Information:\n"
            "  - Name: Internet Firewall Dataset (log2.csv, ~2.88 MB)\n"
            "  - Author: Fatih Ertam, Firat University, Turkey\n"
            "  - Target: 'Action' ('allow', 'deny', 'drop', 'reset-both')\n"
            "  - Expected Attributes: 12 network firewall attributes\n\n"
            "Troubleshooting Steps:\n"
            "  1. Download 'log2.csv' from Kaggle (Internet Firewall Data by Fatih Ertam).\n"
            "  2. Place the file into './data/log2.csv' or the workspace root './log2.csv'.\n"
            "  3. If using the Kaggle CLI: kaggle datasets download -d <dataset-identifier> -p ./data --unzip\n"
            "Never use synthetic or fabricated data in place of the real dataset."
        )

    for candidate in CANDIDATE_PATHS:
        if os.path.exists(candidate):
            return candidate

    # If missing, provide explicit failure instructions
    raise FileNotFoundError(
        "CRITICAL ERROR: Dataset 'log2.csv' not found!\n"
        "Checked paths:\n"
        f"  1. {CANDIDATE_PATHS[0]} (Preferred location)\n"
        f"  2. {CANDIDATE_PATHS[1]} (Alternate location)\n\n"
        "Dataset Information:\n"
        "  - Name: Internet Firewall Dataset (log2.csv, ~2.88 MB)\n"
        "  - Author: Fatih Ertam, Firat University, Turkey\n"
        "  - Target: 'Action' ('allow', 'deny', 'drop', 'reset-both')\n"
        "  - Expected Attributes: 12 network firewall attributes\n\n"
        "Troubleshooting Steps:\n"
        "  1. Download 'log2.csv' from Kaggle (Internet Firewall Data by Fatih Ertam).\n"
        "  2. Place the file into './data/log2.csv' or the workspace root './log2.csv'.\n"
        "  3. If using the Kaggle CLI: kaggle datasets download -d <dataset-identifier> -p ./data --unzip\n"
        "Never use synthetic or fabricated data in place of the real dataset."
    )


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and validates the Internet Firewall dataset from log2.csv.
    Ensures that all 12 expected attributes exist and Action is present.
    Preserves original dataset integrity.
    """
    resolved_path = resolve_dataset_path(file_path)
    df = pd.read_csv(resolved_path)

    # Validate that all 12 expected attributes are present
    missing_cols = [col for col in EXPECTED_ATTRIBUTES if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset validation failed for '{resolved_path}'. Missing required attributes: {missing_cols}. "
            f"Expected 12 attributes: {EXPECTED_ATTRIBUTES}"
        )

    # Validate target column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from '{resolved_path}'. Columns found: {list(df.columns)}"
        )

    return df


def get_dataset_metadata(df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Computes statistical and structural metadata about the loaded dataset.
    """
    if df is None:
        df = load_dataset()

    total_records = len(df)
    class_distribution = df[TARGET_COLUMN].value_counts().to_dict()
    class_percentages = {
        cls: round((count / total_records) * 100, 2)
        for cls, count in class_distribution.items()
    }

    feature_stats = {}
    for col in FEATURE_COLUMNS:
        feature_stats[col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': round(float(df[col].mean()), 2),
            'median': float(df[col].median()),
            'std': round(float(df[col].std()), 2)
        }

    return {
        'total_records': total_records,
        'num_attributes': len(EXPECTED_ATTRIBUTES),
        'attributes': EXPECTED_ATTRIBUTES,
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'class_distribution': class_distribution,
        'class_percentages': class_percentages,
        'feature_stats': feature_stats,
        'dataset_author': "Fatih Ertam, Firat University, Turkey",
        'dataset_name': "Internet Firewall Dataset (Kaggle log2.csv)"
    }


def get_sample_records_by_class(df: Optional[pd.DataFrame] = None, n_per_class: int = 1) -> List[Dict]:
    """
    Retrieves representative sample records from each Action class for testing.
    """
    if df is None:
        df = load_dataset()

    samples = []
    for cls in EXPECTED_CLASSES:
        subset = df[df[TARGET_COLUMN] == cls]
        if not subset.empty:
            sample_rows = subset.head(n_per_class)
            for idx, row in sample_rows.iterrows():
                record = row.to_dict()
                record['row_id'] = int(idx)
                samples.append(record)
    return samples


if __name__ == '__main__':
    print("Testing Dataset Loader...")
    try:
        path = resolve_dataset_path()
        print(f"[OK] Resolved dataset path: {path}")
        df_loaded = load_dataset()
        print(f"[OK] Successfully loaded {len(df_loaded)} records from {path}")
        meta = get_dataset_metadata(df_loaded)
        print(f"[OK] Target distribution: {meta['class_distribution']}")
        print(f"[OK] Feature count: {len(meta['feature_columns'])}")
        samples = get_sample_records_by_class(df_loaded)
        print(f"[OK] Retrieved {len(samples)} sample records across classes")
    except Exception as e:
        print(f"[ERROR] {e}")
