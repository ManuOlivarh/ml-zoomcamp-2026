import pickle
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).with_name("model.bin")
THRESHOLD = 0.5

with MODEL_PATH.open("rb") as f:
    pipeline = pickle.load(f)

app = FastAPI(title="Wine quality prediction")


class Wine(BaseModel):
    fixed_acidity: float = Field(ge=0)
    volatile_acidity: float = Field(ge=0)
    citric_acid: float = Field(ge=0)
    residual_sugar: float = Field(ge=0)
    chlorides: float = Field(ge=0)
    free_sulfur_dioxide: float = Field(ge=0)
    total_sulfur_dioxide: float = Field(ge=0)
    density: float = Field(gt=0)
    pH: float = Field(ge=0, le=14)
    sulphates: float = Field(ge=0)
    alcohol: float = Field(ge=0, le=100)
    wine_type: Literal["red", "white"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(wine: Wine) -> dict:
    probability = float(pipeline.predict_proba([wine.model_dump()])[0, 1])
    return {
        "good_probability": round(probability, 4),
        "good_wine": probability >= THRESHOLD,
    }
