# Trains the final model (random forest, parameters chosen in the notebook)
# and saves it to model.bin

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATA_DIR = Path(__file__).with_name("data")
MODEL_PATH = Path(__file__).with_name("model.bin")

NUMERICAL = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
FEATURES = NUMERICAL + ["wine_type"]

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": 1,
    "n_jobs": -1,
}


def load_data() -> pd.DataFrame:
    frames = []
    for wine_type in ["red", "white"]:
        df = pd.read_csv(DATA_DIR / f"winequality-{wine_type}.csv", sep=";")
        df["wine_type"] = wine_type
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.replace(" ", "_")
    df["good"] = (df["quality"] >= 7).astype(int)
    return df


def main() -> None:
    df = load_data()
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1, stratify=df["good"]
    )

    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        RandomForestClassifier(**RF_PARAMS),
    )
    pipeline.fit(
        df_full_train[FEATURES].to_dict(orient="records"),
        df_full_train["good"].values,
    )

    y_pred = pipeline.predict_proba(df_test[FEATURES].to_dict(orient="records"))[:, 1]
    print(f"test AUC: {roc_auc_score(df_test['good'], y_pred):.4f}")

    dv, model = pipeline.named_steps.values()
    importance = pd.Series(model.feature_importances_, index=dv.get_feature_names_out())
    print("top features:")
    print(importance.sort_values(ascending=False).head(5).round(3).to_string())

    with MODEL_PATH.open("wb") as f:
        pickle.dump(pipeline, f)
    print(f"model saved to {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
