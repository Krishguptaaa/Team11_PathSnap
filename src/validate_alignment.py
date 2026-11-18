
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRIP_ID = "00000001"


def main():

    candidates = pd.read_csv(
        PROCESSED_DIR /
        f"trip_{TRIP_ID}_candidates.csv"
    )

    alignment = pd.read_csv(
        PROCESSED_DIR /
        f"trip_{TRIP_ID}_monotonic_alignment.csv"
    )

    # Match each aligned GPS observation to its candidate set
    merged = alignment.merge(
        candidates[
            [
                "gps_index",
                "arc_id",
                "candidate_rank",
                "distance_m"
            ]
        ],
        left_on=["gps_index", "aligned_arc_id"],
        right_on=["gps_index", "arc_id"],
        how="left"
    )

    merged["candidate_found"] = (
        merged["candidate_rank"].notna()
    )

    total = len(merged)
    found = int(merged["candidate_found"].sum())

    coverage = (
        found / total * 100
        if total else 0
    )

    print("=" * 60)
    print("ALIGNMENT VS CANDIDATE VALIDATION")
    print("=" * 60)

    print("GPS observations:", total)
    print("Aligned arcs found in Top-10:", found)
    print("Aligned arcs missing from Top-10:", total - found)
    print(f"Top-10 alignment coverage: {coverage:.2f}%")

    print("\nCandidate rank distribution:")

    print(
        merged["candidate_rank"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nAlignment distance comparison:")

    print(
        merged[
            [
                "distance_to_aligned_arc_m",
                "distance_m"
            ]
        ].describe().round(3)
    )

    output_file = (
        PROCESSED_DIR /
        f"trip_{TRIP_ID}_alignment_validation.csv"
    )

    merged.to_csv(output_file, index=False)

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
