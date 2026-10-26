"""Question 3: load pipeline.bin with pickle and score one lead."""

import hashlib
import pickle
from pathlib import Path

from model import normalize_record

MODEL_PATH = Path(__file__).with_name("pipeline.bin")
EXPECTED_SHA256 = "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1"

# Only unpickle the artifact if it is the exact file published by the course
actual_sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
assert actual_sha256 == EXPECTED_SHA256, f"unexpected model checksum: {actual_sha256}"

with MODEL_PATH.open("rb") as f:
    pipeline = pickle.load(f)

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

record = normalize_record(lead)
probability = pipeline.predict_proba([record])[0, 1]

print("probability:", probability)
print("rounded:    ", round(probability, 3))
