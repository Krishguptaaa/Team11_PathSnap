
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "feature_selection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORRELATION_THRESHOLD = 0.95
VIF_THRESHOLD = 10

MAX_ANALYSIS_ROWS = 200_000


# ============================================================
# 1. LOAD ALL FEATURE FILES
# ============================================================

def load_all_features():

    files = sorted(
        PROCESSED_DIR.glob("trip_*_features.csv")
    )

    if not files:
        raise FileNotFoundError(
            "No trip feature CSV files found. "
            "Run batch_feature_extraction.py first."
        )

    frames = []

    for file in files:

        print("Loading:", file.name)

        df = pd.read_csv(file)
        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    print("\nCombined feature table:", combined.shape)

    return combined


# ============================================================
# 2. MISSING VALUE ANALYSIS
# ============================================================

def analyse_missing_values(df):

    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percentage": (
            df.isnull().mean() * 100
        ).round(3)
    })

    missing.to_csv(
        OUTPUT_DIR / "missing_value_analysis.csv"
    )

    print("\nMISSING VALUE ANALYSIS")
    print(missing.to_string())

    return missing


# ============================================================
# 3. FEATURE ENGINEERING FOR FEATURE SELECTION
# ============================================================

def prepare_features(df):

    data = pd.DataFrame(index=df.index)

    # Spatial features
    data["distance_m"] = df["distance_m"]

    data["projection_fraction"] = (
        df["projection_fraction"]
    )

    data["segment_length_m"] = (
        df["segment_length_m"]
    )

    # GPS movement features
    data["gps_speed_mps"] = df["gps_speed_mps"]

    data["gps_displacement_m"] = (
        df["gps_displacement_m"]
    )

    data["time_gap_s"] = df["time_gap_s"]

    # Directional difference
    data["heading_difference"] = (
        df["heading_difference"]
    )

    # Circular encoding of GPS bearing
    gps_radians = np.radians(df["gps_bearing"])

    data["gps_bearing_sin"] = np.sin(gps_radians)
    data["gps_bearing_cos"] = np.cos(gps_radians)

    # Circular encoding of road segment bearing
    segment_radians = np.radians(
        df["segment_bearing"]
    )

    data["segment_bearing_sin"] = np.sin(
        segment_radians
    )

    data["segment_bearing_cos"] = np.cos(
        segment_radians
    )

    # Check whether distance and projection distance
    # are effectively duplicates.
    difference = (
        df["distance_m"] - df["projection_distance_m"]
    ).abs()

    print(
        "\nMaximum difference between distance_m and "
        "projection_distance_m:",
        difference.max()
    )

    # Missing-value indicators preserve information
    # about unavailable temporal features.
    data["gps_speed_missing"] = (
        df["gps_speed_mps"].isna().astype(int)
    )

    data["gps_bearing_missing"] = (
        df["gps_bearing"].isna().astype(int)
    )

    # Exclude identifiers, arc IDs, timestamps and candidate rank.
    # These are not included as ordinary numerical features.

    return data


# ============================================================
# 4. FEATURE DISTRIBUTION ANALYSIS
# ============================================================

def plot_feature_distributions(data):

    columns = [
        col for col in data.columns
        if data[col].nunique(dropna=True) > 1
    ]

    for column in columns:

        plt.figure(figsize=(7, 4))

        data[column].dropna().hist(bins=40)

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{column}_distribution.png",
            dpi=150
        )

        plt.close()

    print("\nFeature distribution plots saved.")


# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================

def analyse_correlations(data):

    correlation = data.corr(
        method="spearman"
    )

    correlation.to_csv(
        OUTPUT_DIR / "feature_correlation_matrix.csv"
    )

    # Plot heatmap using Matplotlib.
    plt.figure(figsize=(12, 9))

    plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    plt.colorbar(label="Spearman correlation")

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=180
    )

    plt.close()

    # Identify highly correlated pairs.
    pairs = []

    columns = correlation.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            value = correlation.iloc[i, j]

            if abs(value) >= CORRELATION_THRESHOLD:

                pairs.append({
                    "feature_1": columns[i],
                    "feature_2": columns[j],
                    "correlation": value
                })

    high_corr = pd.DataFrame(pairs)

    high_corr.to_csv(
        OUTPUT_DIR / "high_correlation_pairs.csv",
        index=False
    )

    print("\nHIGHLY CORRELATED FEATURE PAIRS")
    print(
        high_corr.to_string(index=False)
        if not high_corr.empty
        else "No pairs above threshold."
    )

    return correlation, high_corr


# ============================================================
# 6. VARIANCE THRESHOLD
# ============================================================

def remove_low_variance_features(data):

    variances = data.var()

    variance_report = pd.DataFrame({
        "variance": variances
    }).sort_values("variance")

    variance_report.to_csv(
        OUTPUT_DIR / "feature_variance.csv"
    )

    # Remove constant features only.
    selected = [
        column for column in data.columns
        if data[column].nunique(dropna=True) > 1
    ]

    removed = [
        column for column in data.columns
        if column not in selected
    ]

    print("\nConstant features removed:", removed)

    return data[selected]


# ============================================================
# 7. VIF ANALYSIS
# ============================================================

def calculate_vif(data):

    numeric = data.copy()

    # Median imputation for analysis only.
    numeric = numeric.fillna(
        numeric.median()
    )

    # Remove any remaining constant columns.
    numeric = numeric.loc[
        :, numeric.nunique() > 1
    ]

    # VIF can be unstable with perfectly duplicated columns.
    # Iteratively remove the feature with highest VIF.
    selected = list(numeric.columns)

    while len(selected) > 1:

        current = numeric[selected].astype(float)

        # Standardize for numerical stability.
        std = current.std(ddof=0)

        current = (
            current - current.mean()
        ) / std.replace(0, 1)

        vif_values = []

        for i, column in enumerate(selected):

            try:
                value = variance_inflation_factor(
                    current.to_numpy(),
                    i
                )
            except Exception:
                value = np.inf

            vif_values.append({
                "feature": column,
                "VIF": value
            })

        vif_df = pd.DataFrame(vif_values)

        highest = vif_df.loc[
            vif_df["VIF"].idxmax()
        ]

        if highest["VIF"] <= VIF_THRESHOLD:
            break

        if not np.isfinite(highest["VIF"]):
            worst_feature = highest["feature"]
        else:
            worst_feature = highest["feature"]

        print(
            f"Removing {worst_feature}: "
            f"VIF={highest['VIF']:.2f}"
        )

        selected.remove(worst_feature)

    final_data = numeric[selected]

    final_vif = []

    if len(selected) > 1:

        current = final_data.astype(float)

        std = current.std(ddof=0)

        current = (
            current - current.mean()
        ) / std.replace(0, 1)

        for i, column in enumerate(selected):

            final_vif.append({
                "feature": column,
                "VIF": variance_inflation_factor(
                    current.to_numpy(),
                    i
                )
            })

    elif len(selected) == 1:

        final_vif.append({
            "feature": selected[0],
            "VIF": 1.0
        })

    vif_report = pd.DataFrame(final_vif)

    vif_report.to_csv(
        OUTPUT_DIR / "final_vif_report.csv",
        index=False
    )

    return selected, vif_report


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PATHSNAP - FEATURE SELECTION")
    print("=" * 65)

    df = load_all_features()

    # Sample for analysis if the combined table is very large.
    if len(df) > MAX_ANALYSIS_ROWS:

        df = df.sample(
            n=MAX_ANALYSIS_ROWS,
            random_state=42
        )

        print(
            "\nUsing a reproducible sample of",
            MAX_ANALYSIS_ROWS,
            "rows for feature analysis."
        )

    # Save the raw combined feature table.
    df.to_csv(
        PROCESSED_DIR / "all_training_features.csv",
        index=False
    )

    analyse_missing_values(df)

    features = prepare_features(df)

    # Save feature-selection input before imputation.
    features.to_csv(
        OUTPUT_DIR / "feature_selection_input.csv",
        index=False
    )

    # Median imputation for correlation and VIF analysis.
    features = features.fillna(
        features.median()
    )

    plot_feature_distributions(features)

    correlation, high_corr = analyse_correlations(
        features
    )

    features = remove_low_variance_features(
        features
    )

    selected, vif_report = calculate_vif(
        features
    )

    # Save recommended feature list.
    pd.DataFrame({
        "selected_feature": selected
    }).to_csv(
        OUTPUT_DIR / "selected_features.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 65)

    print("Initial engineered features:", len(features.columns))
    print("Selected features:", len(selected))

    print("\nSelected feature list:")

    for feature in selected:
        print("-", feature)

    print("\nFinal VIF report:")
    print(vif_report.to_string(index=False))

    print("\nOutputs saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
