
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def test_heart_disease_dataset():
    path = DATA_DIR / "heart_disease.csv"
    assert path.exists(), "Heart Disease dataset is missing"

    data = pd.read_csv(path)
    assert data.shape == (303, 14)
    assert "target" in data.columns


def test_boston_housing_dataset():
    path = DATA_DIR / "boston_housing.csv"
    assert path.exists(), "Boston Housing dataset is missing"

    data = pd.read_csv(path)
    assert len(data) == 506
    assert "MEDV" in data.columns
