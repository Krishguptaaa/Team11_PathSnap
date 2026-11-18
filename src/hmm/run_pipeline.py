"""Run the HMM map-matching pipeline on a deliberately small trip batch."""

import argparse
import gc
import time
from pathlib import Path

import pandas as pd

from . import emission_probabilities
from . import evaluate_viterbi
from . import transition_probabilities
from . import viterbi


ROOT = Path(__file__).resolve().parents[2]
HMM_DIR = ROOT / "outputs" / "hmm_results"
EVALUATION_DIR = HMM_DIR / "evaluation"
SUMMARY_PATH = EVALUATION_DIR / "pilot_batch_summary.csv"
PILOT_TRIPS = (5, 13, 16)


def _trip_output_files(trip_id):
    trip_name = f"{trip_id:08d}"
    return [
        HMM_DIR / f"trip_{trip_name}_emissions.csv",
        HMM_DIR / f"trip_{trip_name}_transitions.csv",
        HMM_DIR / f"trip_{trip_name}_transition_matrices.npz",
        HMM_DIR / f"trip_{trip_name}_viterbi_predictions.csv",
        HMM_DIR / f"trip_{trip_name}_viterbi_transition_quality.csv",
        EVALUATION_DIR / f"trip_{trip_name}_evaluation_summary.csv",
        EVALUATION_DIR / f"trip_{trip_name}_rf_vs_viterbi_comparison.csv",
    ]


def process_trip(trip_id, model, model_load_seconds):
    trip_name = f"{trip_id:08d}"
    started = time.perf_counter()

    emissions = emission_probabilities.process_trip(
        trip_name,
        model=model,
    )
    observation_count = emissions["gps_index"].nunique()
    candidate_count = len(emissions)
    del emissions

    transition_probabilities.main(trip_id)
    viterbi.main(trip_id)
    summary, comparison = evaluate_viterbi.evaluate_trip(trip_id)

    prediction_path = (
        HMM_DIR / f"trip_{trip_name}_viterbi_predictions.csv"
    )
    predictions = pd.read_csv(prediction_path, usecols=["gps_index"])
    if len(predictions) != observation_count:
        raise ValueError(
            f"Trip {trip_name}: expected {observation_count} predictions "
            f"for GPS observations, got {len(predictions)}."
        )

    quality_path = (
        HMM_DIR / f"trip_{trip_name}_viterbi_transition_quality.csv"
    )
    quality = pd.read_csv(quality_path)
    if len(quality) != max(0, observation_count - 1):
        raise ValueError(
            f"Trip {trip_name}: expected {max(0, observation_count - 1)} "
            f"transition diagnostics, got {len(quality)}."
        )

    selected_reachability = (
        float(quality["reachable"].astype(bool).mean())
        if not quality.empty
        else float("nan")
    )
    runtime_seconds = time.perf_counter() - started
    output_bytes = sum(path.stat().st_size for path in _trip_output_files(trip_id))

    summary["candidate_records"] = candidate_count
    summary["prediction_observations"] = len(predictions)
    summary["selected_transition_reachability"] = selected_reachability
    summary["runtime_seconds"] = runtime_seconds
    summary["model_load_seconds"] = model_load_seconds
    summary["output_size_mb"] = output_bytes / (1024 * 1024)

    print(
        f"Trip {trip_name}: {observation_count:,} observations, "
        f"{candidate_count:,} candidates, "
        f"{runtime_seconds:.1f}s, "
        f"{output_bytes / (1024 * 1024):.1f} MiB output."
    )

    return summary, comparison, quality


def build_aggregate_summary(comparisons, qualities):
    comparison = pd.concat(comparisons, ignore_index=True)
    valid = comparison.dropna(
        subset=[
            "aligned_arc_id",
            "classifier_top1_arc_id",
            "viterbi_arc_id",
        ]
    ).copy()

    valid["aligned_arc_id"] = valid["aligned_arc_id"].astype(int)
    valid["classifier_top1_arc_id"] = (
        valid["classifier_top1_arc_id"].astype(int)
    )
    valid["viterbi_arc_id"] = valid["viterbi_arc_id"].astype(int)

    # Arc IDs are only unique within a trip's road network.
    y_true = valid["trip_id"] + ":" + valid["aligned_arc_id"].astype(str)
    rf_pred = valid["trip_id"] + ":" + valid[
        "classifier_top1_arc_id"
    ].astype(str)
    hmm_pred = valid["trip_id"] + ":" + valid["viterbi_arc_id"].astype(str)

    rows = []
    for method, y_pred, distance_column in [
        ("Random Forest Top-1", rf_pred, "classifier_top1_distance_m"),
        ("HMM Viterbi", hmm_pred, "distance_m"),
    ]:
        metrics = evaluate_viterbi.calculate_metrics(y_true, y_pred)
        distance = evaluate_viterbi.distance_statistics(
            valid[distance_column]
        )
        reference = evaluate_viterbi.distance_statistics(
            valid["distance_to_aligned_arc_m"]
        )
        rows.append({
            "trip_id": "ALL_PILOT_TRIPS",
            "method": method,
            "evaluation_observations": len(valid),
            **metrics,
            "mean_selected_distance_m": distance["mean"],
            "median_selected_distance_m": distance["median"],
            "reference_mean_distance_m": reference["mean"],
            "reference_median_distance_m": reference["median"],
            "prediction_changes": int(
                (valid["classifier_top1_arc_id"] != valid["viterbi_arc_id"]).sum()
            ),
            "prediction_change_percentage": 100 * (
                valid["classifier_top1_arc_id"] != valid["viterbi_arc_id"]
            ).mean(),
            "viterbi_improved_count": int(
                ((valid["classifier_top1_arc_id"] != valid["aligned_arc_id"])
                 & (valid["viterbi_arc_id"] == valid["aligned_arc_id"])).sum()
            ),
            "viterbi_worsened_count": int(
                ((valid["classifier_top1_arc_id"] == valid["aligned_arc_id"])
                 & (valid["viterbi_arc_id"] != valid["aligned_arc_id"])).sum()
            ),
            "both_correct_count": int(
                ((valid["classifier_top1_arc_id"] == valid["aligned_arc_id"])
                 & (valid["viterbi_arc_id"] == valid["aligned_arc_id"])).sum()
            ),
            "both_wrong_count": int(
                ((valid["classifier_top1_arc_id"] != valid["aligned_arc_id"])
                 & (valid["viterbi_arc_id"] != valid["aligned_arc_id"])).sum()
            ),
            "selected_transition_reachability": (
                sum(int(item["reachable"].astype(bool).sum()) for item in qualities)
                / sum(len(item) for item in qualities)
                if sum(len(item) for item in qualities)
                else float("nan")
            ),
        })

    return pd.DataFrame(rows)


def run(trip_ids):
    HMM_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    model_started = time.perf_counter()
    model = emission_probabilities.load_random_forest()
    model_load_seconds = time.perf_counter() - model_started

    summaries = []
    comparisons = []
    qualities = []
    for trip_id in trip_ids:
        summary, comparison, quality = process_trip(
            trip_id,
            model,
            model_load_seconds,
        )
        summaries.append(summary)
        comparisons.append(comparison)
        qualities.append(quality)
        gc.collect()

    per_trip_summary = pd.concat(summaries, ignore_index=True)
    aggregate_summary = build_aggregate_summary(comparisons, qualities)
    pd.concat(
        [per_trip_summary, aggregate_summary],
        ignore_index=True,
    ).to_csv(SUMMARY_PATH, index=False)

    print("\nPilot batch summary:")
    print(per_trip_summary[
        [
            "trip_id",
            "method",
            "evaluation_observations",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "selected_transition_reachability",
            "runtime_seconds",
            "output_size_mb",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nAggregate summary saved to {SUMMARY_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trip-ids",
        nargs="+",
        type=int,
        default=PILOT_TRIPS,
        help="Trip IDs to process (defaults to the 5, 13, 16 pilot batch).",
    )
    args = parser.parse_args()
    run(args.trip_ids)


if __name__ == "__main__":
    main()
