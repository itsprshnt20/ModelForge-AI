"""CSV validation, reproducible training, and validation-only evaluation."""
import csv
from dataclasses import dataclass
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_BYTES = 3 * 1024 * 1024
MAX_ROWS = 20_000
MAX_COLUMNS = 50
MAX_CATEGORIES = 100
MAX_NUMERIC = 1e12
SEED = 42
VALIDATION_SIZE = 0.25
MODELS = ("Majority baseline", "Logistic regression", "Random forest")
MISSING = {"", "na", "n/a", "null"}


class ValidationError(ValueError):
    pass


def parse_csv(raw):
    if len(raw) > MAX_BYTES:
        raise ValidationError("CSV exceeds 3 MiB. Export a smaller dataset.")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValidationError("Export the CSV as UTF-8 and try again.") from exc
    if "\x00" in content:
        raise ValidationError("CSV contains binary data. Export it as UTF-8 text.")
    reader = csv.reader(io.StringIO(content, newline=""), strict=True)
    try:
        headers = [header.strip() for header in next(reader, [])]
        if not 2 <= len(headers) <= MAX_COLUMNS:
            raise ValidationError("CSV needs 2–50 columns, including a target and at least one feature.")
        if len(set(headers)) != len(headers) or any(
            not item or len(item) > 100 or not item.isprintable() for item in headers
        ):
            raise ValidationError("Use unique, printable column names with 1–100 characters.")
        rows = []
        for row in reader:
            if len(rows) >= MAX_ROWS:
                raise ValidationError("CSV exceeds 20,000 data rows. Export a smaller sample.")
            if len(row) != len(headers):
                raise ValidationError(f"CSV record ending on line {reader.line_num} has "
                                      f"{len(row)} fields. Expected {len(headers)}.")
            rows.append([value.strip() if value.strip().casefold() not in MISSING else None
                         for value in row])
    except csv.Error as exc:
        raise ValidationError(f"Malformed CSV near line {reader.line_num}. Check quoting and field lengths.") from exc
    if not rows:
        raise ValidationError("CSV contains headers only. Add labeled data rows.")
    return pd.DataFrame(rows, columns=headers, dtype=object)


def target_labels(frame, target):
    if target not in frame:
        raise ValidationError("Select a target column in the dataset.")
    series = frame[target]
    if series.isna().any():
        raise ValidationError("Target contains missing values. Fill or remove unlabeled rows.")
    labels = tuple(sorted(series.unique()))
    if len(labels) != 2:
        raise ValidationError(f"Target must contain exactly two classes. Found {len(labels)}.")
    if any(len(label) > 100 or not label.isprintable() for label in labels):
        raise ValidationError("Use printable target labels with at most 100 characters.")
    return labels


def split_labels(labels, seed=SEED):
    counts = labels.value_counts()
    size = math.ceil(len(labels) * VALIDATION_SIZE)
    if len(counts) != 2 or counts.min() < 2 or min(size, len(labels) - size) < 2:
        raise ValidationError("Too few labeled examples for a stratified split. Add rows to both classes.")
    train, validation = train_test_split(np.arange(len(labels)), test_size=VALIDATION_SIZE,
                                         stratify=labels, random_state=seed)
    if labels.iloc[train].nunique() != 2 or labels.iloc[validation].nunique() != 2:
        raise ValidationError("The stratified split leaves one class absent from a partition. "
                              "Add more examples of the smaller class.")
    return train, validation


def feature_columns(frame, target, excluded):
    columns = [name for name in frame.columns if name != target and name not in excluded]
    if not columns:
        raise ValidationError("Select at least one feature besides the target.")
    return columns


def prepare_features(train, validation):
    """Infer feature types using training rows only; apply that schema to validation."""
    numeric, categorical = [], []
    train, validation = train.copy(), validation.copy()
    for name in train:
        original = train[name]
        present = original.dropna()
        parsed = pd.to_numeric(present, errors="coerce")
        if not len(present) or parsed.notna().all():
            numeric.append(name)
            for part in (train, validation):
                converted = pd.to_numeric(part[name], errors="coerce").astype(float)
                # Unexpected numeric fields in validation are treated as missing.
                part[name] = converted.where(converted.abs() <= MAX_NUMERIC)
            if (parsed.astype(float).abs() > MAX_NUMERIC).any():
                raise ValidationError(f"Feature {name!r} contains a number outside ±1e12. "
                                      "Rescale the feature or exclude it.")
        else:
            if present.nunique() > MAX_CATEGORIES or present.map(len).max() > 120:
                raise ValidationError(
                    f"Feature {name!r} is unsupported: above 100 training categories or 120 characters. "
                    "Exclude this ID or free-text field."
                )
            categorical.append(name)
            for part in (train, validation):
                part[name] = part[name].where(part[name].notna(), np.nan)
    return train, validation, numeric, categorical


@dataclass(frozen=True)
class Run:
    source: str
    target: str
    positive: str
    negative: str
    features: tuple
    excluded: tuple
    train_size: int
    validation_rows: tuple
    observed: tuple
    probabilities: dict


def fit_models(frame, target, positive, excluded, source):
    labels = target_labels(frame, target)
    if positive not in labels:
        raise ValidationError("Explicitly choose which of the two target classes is positive.")
    features = feature_columns(frame, target, excluded)
    y = frame[target].eq(positive).astype(int)
    train_ids, val_ids = split_labels(y)
    x = frame.loc[:, features]
    train, validation, numeric, categorical = prepare_features(x.iloc[train_ids], x.iloc[val_ids])
    estimators = (
        DummyClassifier(strategy="most_frequent"),
        LogisticRegression(random_state=SEED, max_iter=1000),
        RandomForestClassifier(random_state=SEED, n_estimators=120, n_jobs=1),
    )
    probabilities = {}
    for name, estimator in zip(MODELS, estimators):
        numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if name == MODELS[1]:
            numeric_steps.append(("scale", StandardScaler()))
        preprocess = ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ])
        pipeline = Pipeline([("prepare", preprocess), ("model", estimator)])
        pipeline.fit(train, y.iloc[train_ids])
        positive_position = list(pipeline.classes_).index(1)
        probabilities[name] = tuple(float(value) for value in pipeline.predict_proba(validation)[:, positive_position])
    negative = next(label for label in labels if label != positive)
    return Run(source, target, positive, negative, tuple(features), tuple(excluded), len(train_ids),
               tuple(int(index) for index in val_ids), tuple(y.iloc[val_ids].tolist()), probabilities)


def evaluate(observed, probabilities, threshold):
    predicted = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(observed, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        observed, predicted, average="binary", zero_division=0
    )
    return {
        "Balanced accuracy": float(balanced_accuracy_score(observed, predicted)),
        "Precision": float(precision), "Recall": float(recall), "F1": float(f1),
        "Average precision": float(average_precision_score(observed, probabilities)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def model_card(run, model, threshold):
    metric = evaluate(run.observed, run.probabilities[model], threshold)

    def literal(value):
        return "\n".join("    " + line for line in value.splitlines())

    context = "\n".join([
        f"Dataset source: {run.source}", f"Target: {run.target}",
        f"Positive class (1): {run.positive}", f"Negative class (0): {run.negative}",
        "Selected features: " + ", ".join(run.features),
        "Excluded fields: " + (", ".join(run.excluded) or "None"),
    ])
    lines = ["# ModelForge AI model card", "", literal(context), "",
             "## Training configuration", "", f"- Model: {model}",
             f"- Seed: {SEED}. Stratified validation: {VALIDATION_SIZE:.0%}.",
             f"- Training rows: {run.train_size}. Validation rows: {len(run.observed)}.",
             "- Training-only median and mode imputation; one-hot categorical encoding.",
             "- Logistic regression scales numeric features. Random forest uses 120 trees.",
             "", "## Validation results", "", f"- Threshold: {threshold:.2f} (positive when score ≥ threshold)",
             *[f"- {name}: {metric[name]:.3f}" for name in
               ("Balanced accuracy", "Precision", "Recall", "F1", "Average precision")],
             "", "| Actual / predicted | Negative | Positive |", "| --- | ---: | ---: |",
             f"| Negative | TN {metric['TN']} | FP {metric['FP']} |",
             f"| Positive | FN {metric['FN']} | TP {metric['TP']} |", "", "## Limitations", "",
             "- Metrics use the validation set. A threshold selected here needs an independent test set.",
             "- Undefined precision, recall, and F1 terms use zero. Average precision uses saved scores.",
             "- Use domain knowledge to exclude leakage, duplicates, and post-outcome fields.",
             "- Synthetic demo results do not establish performance on real customers."]
    return "\n".join(lines) + "\n"


def main():
    path = Path(__file__).resolve().parent / "samples" / "synthetic_churn.csv"
    data = parse_csv(path.read_bytes())
    run = fit_models(data, "churn", "Yes", ("customer_id", "cancellation_ticket"),
                     "Synthetic customer churn")
    print(f"{len(data)} rows · class counts {data['churn'].value_counts().to_dict()}")
    print(f"{run.train_size} training rows · {len(run.observed)} validation rows")
    for name in MODELS:
        values = evaluate(run.observed, run.probabilities[name], 0.50)
        metrics = ("Balanced accuracy", "Precision", "Recall", "F1", "Average precision")
        print(name + " · " + " · ".join(f"{metric}: {values[metric]:.3f}" for metric in metrics))


if __name__ == "__main__":
    main()
