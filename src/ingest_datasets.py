
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEART_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

BOSTON_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/housing/"
    "housing.data"
)


# --------------------------------------------------
# 2. Download and prepare Heart Disease data
# --------------------------------------------------

heart_columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

print("Downloading Heart Disease dataset...")

heart = pd.read_csv(
    HEART_URL,
    header=None,
    names=heart_columns,
    na_values="?",
)

# Convert columns to numeric and mark invalid values as missing.
for column in heart_columns:
    heart[column] = pd.to_numeric(
        heart[column],
        errors="coerce",
    )

# Remove rows that do not have a valid diagnosis.
heart = heart.dropna(subset=["num"]).copy()

# Original diagnosis values:
# 0 = absence of disease
# 1, 2, 3, 4 = presence of disease
heart["target"] = (heart["num"] > 0).astype(int)

# Keep the binary target, not the original diagnosis column.
heart = heart.drop(columns=["num"])

heart_path = DATA_DIR / "heart_disease.csv"
heart.to_csv(heart_path, index=False)


# --------------------------------------------------
# 3. Download and prepare Boston Housing data
# --------------------------------------------------

boston_columns = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]

print("Downloading Boston Housing dataset...")

boston = pd.read_csv(
    BOSTON_URL,
    sep=r"\s+",
    header=None,
    names=boston_columns,
)

boston = boston.apply(
    pd.to_numeric,
    errors="coerce",
)

boston = boston.dropna(subset=["MEDV"]).copy()

boston_path = DATA_DIR / "boston_housing.csv"
boston.to_csv(boston_path, index=False)


# --------------------------------------------------
# 4. Verify and report the downloaded datasets
# --------------------------------------------------

print("\nDATA INGESTION COMPLETED")
print("=" * 45)

print("\nHeart Disease dataset")
print("Shape:", heart.shape)
print("Features:", heart.shape[1] - 1)
print("Missing values:", int(heart.isna().sum().sum()))
print("Target counts:")
print(heart["target"].value_counts().sort_index())
print("Saved to:", heart_path)
print("\nSample records:")
print(heart.head(3).to_string(index=False))

print("\nBoston Housing dataset")
print("Shape:", boston.shape)
print("Features:", boston.shape[1] - 1)
print("Missing values:", int(boston.isna().sum().sum()))
print("Target: MEDV")
print("Saved to:", boston_path)
print("\nSample records:")
print(boston.head(3).to_string(index=False))