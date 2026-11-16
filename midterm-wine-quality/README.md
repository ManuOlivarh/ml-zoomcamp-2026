# Wine quality prediction

Midterm project for ML Zoomcamp 2026.

## Problem

Wine quality is normally rated by expert tasters. That takes time and money, while lab measurements
like acidity, sugar, sulphates or alcohol are cheap and available as soon as the wine is produced.

The idea of this project is to predict if a wine is good (quality 7 or more, on a 0-10 scale) using
only those measurements and the type of wine (red or white). A winery could use it to decide which
batches are worth sending to the tasters or to a premium line, and skip the ones that are very
likely average.

Only about 20% of the wines are good, so accuracy is not a good metric here (always predicting
"not good" already gives 80%). I used ROC AUC to compare the models.

## Dataset

Wine Quality dataset from the UCI repository: https://archive.ics.uci.edu/dataset/186/wine+quality

It has 1599 red and 4898 white Portuguese wines (vinho verde), with 11 lab measurements and the
quality score given by tasters. The two CSV files are in the `data` folder. If you want to download
them again:

```bash
curl -L -o data/winequality-red.csv https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
curl -L -o data/winequality-white.csv https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
```

## What I did

Everything is in `notebook.ipynb`:

- data cleaning (no missing values, 1177 duplicated rows that I kept, see the notebook)
- EDA: ranges, distributions, the target, correlations, AUC of each feature and mutual information
- I trained and tuned 4 models on a 60/20/20 split

| Model | Parameters tuned | Validation AUC |
|---|---|---|
| Logistic regression | C | 0.809 |
| Decision tree | max_depth, min_samples_leaf | 0.834 |
| XGBoost | eta, max_depth, min_child_weight, number of rounds | 0.899 |
| Random forest | n_estimators, max_depth, min_samples_leaf | 0.912 |

The random forest was the best one. After retraining it on train + validation it got 0.915 AUC on
the test set. The most important feature is alcohol, then density, volatile acidity and chlorides.

Files:

- `notebook.ipynb` - EDA and model selection
- `train.py` - trains the final model and saves it to `model.bin`
- `predict.py` - web service (FastAPI)
- `test.py` - sends two example wines to the service
- `pyproject.toml` and `uv.lock` - dependencies
- `Dockerfile`

## How to run it

I used [uv](https://docs.astral.sh/uv/) to manage the virtual environment and the dependencies
(Python 3.12). Install it with `pip install uv` if you don't have it.

```bash
uv sync
```

This creates a `.venv` folder with everything installed. You can run commands inside it with
`uv run ...`, or activate it:

```bash
source .venv/bin/activate      # Linux / Mac
.venv\Scripts\activate         # Windows
```

Train the model:

```bash
uv run python train.py
```

Start the service:

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

You can also open http://localhost:9696/docs to try it from the browser.

Then, in another terminal:

```bash
uv run python test.py
```

You should get something like this:

```
white, rated 7: {'good_probability': 0.505, 'good_wine': True}
red, rated 5: {'good_probability': 0.0, 'good_wine': False}
```

Or with curl:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"fixed_acidity": 6.6, "volatile_acidity": 0.16, "citric_acid": 0.4,
       "residual_sugar": 1.5, "chlorides": 0.044, "free_sulfur_dioxide": 48.0,
       "total_sulfur_dioxide": 143.0, "density": 0.9912, "pH": 3.54,
       "sulphates": 0.52, "alcohol": 12.4, "wine_type": "white"}'
```

## Docker

The model is trained when the image is built, so you don't need to run `train.py` first.

```bash
docker build -t wine-quality .
docker run -it --rm -p 9696:9696 wine-quality
```

And test it the same way, with `test.py` or curl.
