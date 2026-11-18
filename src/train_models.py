
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = ROOT / "outputs" / "model_results"
MODEL_DIR = OUTPUT_DIR / "saved_models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Features used for candidate scoring.
# Identifiers, timestamps, arc IDs and candidate rank are excluded.
BASE_FEATURES = [
    "distance_m",
    "gps_speed_mps",
    "gps_displacement_m",
    "time_gap_s",
    "segment_length_m",
    "heading_difference",
    "projection_fraction",
]


# ============================================================
# 2. LOAD FEATURES AND CREATE LABELS IN MEMORY
# ============================================================

def load_data():

    feature_files = sorted(
        DATA_DIR.glob("trip_*_features.csv")
    )

    frames = []

    print("Searching for processed trips...")

    for feature_file in feature_files:

        trip_id = feature_file.stem.replace(
            "trip_", ""
        ).replace("_features", "")

        alignment_file = (
            DATA_DIR /
            f"trip_{trip_id}_monotonic_alignment.csv"
        )

        if not alignment_file.exists():

            print(
                f"Skipping {trip_id}: "
                "alignment file not found."
            )
            continue

        features = pd.read_csv(feature_file)
        alignment = pd.read_csv(alignment_file)

        required_alignment_columns = {
            "trip_id",
            "gps_index",
            "aligned_arc_id"
        }

        if not required_alignment_columns.issubset(
            alignment.columns
        ):
            print(
                f"Skipping {trip_id}: invalid alignment columns."
            )
            continue

        labelled = features.merge(
            alignment[
                ["trip_id", "gps_index", "aligned_arc_id"]
            ],
            on=["trip_id", "gps_index"],
            how="inner",
            validate="many_to_one"
        )

        labelled["label"] = (
            labelled["arc_id"] ==
            labelled["aligned_arc_id"]
        ).astype(int)

        # Remove GPS observations for which no candidate
        # matches the provisional aligned arc.
        positive_by_observation = (
            labelled.groupby(
                ["trip_id", "gps_index"]
            )["label"].transform("max")
        )

        labelled = labelled[
            positive_by_observation == 1
        ].copy()

        frames.append(labelled)

        print(
            f"{trip_id}: {len(features)} candidates, "
            f"{labelled['gps_index'].nunique()} "
            "usable GPS observations"
        )

    if not frames:
        raise RuntimeError(
            "No labelled feature data found. "
            "Generate monotonic alignment files first."
        )

    data = pd.concat(
        frames,
        ignore_index=True
    )

    print("\nCombined candidate data:", data.shape)
    print("Trips available:", data["trip_id"].nunique())

    print("\nLabel distribution:")
    print(data["label"].value_counts())

    if data["trip_id"].nunique() < 2:
        raise RuntimeError(
            "At least two aligned trips are required "
            "for a trip-level train/validation split."
        )

    return data


# ============================================================
# 3. FEATURE PREPARATION
# ============================================================

def prepare_features(data):

    X = data[BASE_FEATURES].copy()

    # Encode circular directions.
    # This avoids treating 359 degrees and 1 degree as far apart.
    gps_radians = np.radians(data["gps_bearing"])

    X["gps_bearing_sin"] = np.sin(gps_radians)
    X["gps_bearing_cos"] = np.cos(gps_radians)

    segment_radians = np.radians(
        data["segment_bearing"]
    )

    X["segment_bearing_sin"] = np.sin(
        segment_radians
    )

    X["segment_bearing_cos"] = np.cos(
        segment_radians
    )

    return X


# ============================================================
# 4. DEFINE FIVE MODELS
# ============================================================

def create_models():

    models = {

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Gaussian Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GaussianNB())
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ]),

        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=300,
                early_stopping=True,
                random_state=RANDOM_STATE
            ))
        ])
    }

    # LightGBM is an optional dependency.
    try:

        from lightgbm import LGBMClassifier

        models["LightGBM"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                class_weight="balanced",
                verbosity=-1,
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])

    except ImportError:

        print(
            "\nLightGBM not installed. "
            "Run: pip install lightgbm"
        )

    return models


# ============================================================
# 5. OBSERVATION-LEVEL TOP-1 ACCURACY
# ============================================================

def calculate_top1_accuracy(
    test_data,
    probabilities
):

    results = test_data[
        ["trip_id", "gps_index", "arc_id", "label",
         "distance_m"]
    ].copy()

    results["predicted_probability"] = probabilities

    # Select the highest-scoring candidate for each GPS point.
    # Distance is used only as a deterministic tie-breaker.
    results = results.sort_values(
        ["predicted_probability", "distance_m"],
        ascending=[False, True]
    )

    best_candidates = results.drop_duplicates(
        subset=["trip_id", "gps_index"]
    )

    return accuracy_score(
        best_candidates["label"],
        np.ones(len(best_candidates), dtype=int)
    )


# ============================================================
# 6. EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    test_data
):

    print("\n" + "=" * 65)
    print("TRAINING:", name)
    print("=" * 65)

    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    start = time.perf_counter()

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    inference_time = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    binary_f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=[0, 1]
    )

    top1_accuracy = calculate_top1_accuracy(
        test_data,
        probabilities
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    result = {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "binary_f1": binary_f1,
        "macro_f1": macro_f1,
        "log_loss": loss,
        "observation_top1_accuracy": top1_accuracy,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time
    }

    print("\nMetrics:")

    for key, value in result.items():

        if key != "model":
            print(f"{key}: {value:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    safe_name = name.lower().replace(" ", "_")

    pd.DataFrame(
        matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    ).to_csv(
        OUTPUT_DIR / f"{safe_name}_confusion_matrix.csv"
    )

    joblib.dump(
        model,
        MODEL_DIR / f"{safe_name}.joblib"
    )

    return result


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PATHSNAP - MODEL TRAINING AND COMPARISON")
    print("=" * 65)

    data = load_data()

    X = prepare_features(data)

    y = data["label"].astype(int)

    groups = data["trip_id"]

    # Split by trip, not by individual candidate rows.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    test_data = data.iloc[test_indices].copy()

    print("\nTraining trips:")
    print(sorted(data.iloc[train_indices]["trip_id"].unique()))

    print("\nValidation trips:")
    print(sorted(data.iloc[test_indices]["trip_id"].unique()))

    print("\nTraining rows:", len(X_train))
    print("Validation rows:", len(X_test))

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError(
            "Train or validation split contains only one class. "
            "More aligned trips are needed."
        )

    models = create_models()

    results = []

    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            test_data
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "macro_f1",
        ascending=False
    )

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df.to_string(index=False)
    )

    print("\nResults saved to:", OUTPUT_DIR)
    print("Models saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()