
import json
import math
import os
from urllib.request import Request, urlopen

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

HEART = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}

BOSTON = {
    "CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31, "CHAS": 0,
    "NOX": 0.538, "RM": 6.575, "AGE": 65.2, "DIS": 4.09,
    "RAD": 1, "TAX": 296, "PTRATIO": 15.3, "B": 396.9,
    "LSTAT": 4.98,
}

TESTS = [
    ("/predict/heart/random-forest", HEART),
    ("/predict/heart/logistic", HEART),
    ("/predict/boston", BOSTON),
]


def request_json(url, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers)
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    health = request_json(f"{BASE_URL}/health")
    assert health.get("status") == "ok"
    print("PASS: API health")

    for endpoint, features in TESTS:
        result = request_json(BASE_URL + endpoint, features)
        prediction = result.get("prediction")
        assert isinstance(prediction, (int, float))
        assert math.isfinite(prediction)
        print(f"PASS: {endpoint}; prediction={prediction}")

    print("ALL DEPLOYMENT TESTS PASSED")


if __name__ == "__main__":
    main()
