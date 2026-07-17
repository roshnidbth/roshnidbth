"""Preprocess Falcon 9 features, tune classifiers, and report test metrics."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def train_model(frame: pd.DataFrame, target: str = "Class") -> tuple[GridSearchCV, dict]:
    X = frame.drop(columns=[target])
    y = frame[target].astype(int)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.columns.difference(numeric).tolist()

    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                      ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    pipeline = Pipeline([("prepare", preprocessing), ("model", SVC())])
    search = GridSearchCV(
        pipeline,
        {"model__kernel": ["linear", "rbf", "sigmoid"],
         "model__C": [0.1, 1, 10],
         "model__gamma": ["scale", "auto"]},
        cv=5,
        scoring="accuracy",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2, stratify=y
    )
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "best_parameters": search.best_params_,
    }
    return search, metrics


if __name__ == "__main__":
    data = pd.read_csv("dataset_part_2.csv")
    _, results = train_model(data)
    print(results)
