# sends two wines from the dataset to the service

import json
import sys
from urllib.request import Request, urlopen

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696/predict"

wines = {
    # row 13 of winequality-white.csv, quality 7
    "white, rated 7": {
        "fixed_acidity": 6.6,
        "volatile_acidity": 0.16,
        "citric_acid": 0.4,
        "residual_sugar": 1.5,
        "chlorides": 0.044,
        "free_sulfur_dioxide": 48.0,
        "total_sulfur_dioxide": 143.0,
        "density": 0.9912,
        "pH": 3.54,
        "sulphates": 0.52,
        "alcohol": 12.4,
        "wine_type": "white",
    },
    # row 0 of winequality-red.csv, quality 5
    "red, rated 5": {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "wine_type": "red",
    },
}

for name, wine in wines.items():
    request = Request(
        url,
        data=json.dumps(wine).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        print(f"{name}: {json.load(response)}")
