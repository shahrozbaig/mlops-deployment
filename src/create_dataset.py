
import csv
import random
from pathlib import Path

# Keep the generated dataset reproducible
random.seed(42)

# Save the CSV inside the project's data folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "passengers.csv"

columns = [
    "passenger_id",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "survived",
]

rows = []

for passenger_id in range(1, 1001):
    pclass = random.choices(
        [1, 2, 3], weights=[24, 21, 55]
    )[0]

    sex = random.choices(
        ["female", "male"], weights=[38, 62]
    )[0]

    age = round(max(1, min(80, random.gauss(30, 14))), 1)

    sibsp = random.choices(
        [0, 1, 2, 3, 4],
        weights=[68, 21, 6, 3, 2]
    )[0]

    parch = random.choices(
        [0, 1, 2, 3, 4],
        weights=[70, 15, 10, 3, 2]
    )[0]

    embarked = random.choices(
        ["S", "C", "Q"], weights=[72, 19, 9]
    )[0]

    # Generate a fare according to ticket class
    fare_ranges = {
        1: (30, 180),
        2: (10, 60),
        3: (4, 40),
    }

    low, high = fare_ranges[pclass]
    fare = round(random.uniform(low, high), 2)

    # Simulate a survival outcome for this practice dataset
    probability = 0.18

    if sex == "female":
        probability += 0.55

    if pclass == 1:
        probability += 0.14
    elif pclass == 2:
        probability += 0.07

    if age <= 14:
        probability += 0.12

    if age >= 60:
        probability -= 0.08

    if fare >= 50:
        probability += 0.03

    family_size = sibsp + parch + 1

    if 2 <= family_size <= 4:
        probability += 0.05
    elif family_size >= 6:
        probability -= 0.12

    probability = max(0.02, min(0.95, probability))
    survived = int(random.random() < probability)

    rows.append({
        "passenger_id": passenger_id,
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked,
        "survived": survived,
    })

# Write all records to CSV
with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

survivors = sum(row["survived"] for row in rows)

print(f"Dataset created: {OUTPUT_FILE}")
print(f"Total passengers: {len(rows)}")
print(f"Survived: {survivors}")
print(f"Did not survive: {len(rows) - survivors}")