
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRIP_ID = "00000001"

alignment = pd.read_csv(
    PROCESSED_DIR / f"trip_{TRIP_ID}_monotonic_alignment.csv"
)

candidates = pd.read_csv(
    PROCESSED_DIR / f"trip_{TRIP_ID}_candidates.csv"
)

# Identify aligned arcs absent from the candidate list
merged = alignment.merge(
    candidates[["gps_index", "arc_id", "candidate_rank", "distance_m"]],
    left_on=["gps_index", "aligned_arc_id"],
    right_on=["gps_index", "arc_id"],
    how="left"
)

missing = merged[merged["candidate_rank"].isna()].copy()

print("=" * 65)
print("MISSING CANDIDATE ANALYSIS")
print("=" * 65)

print("Missing observations:", len(missing))

print(
    missing[
        [
            "gps_index",
            "timestamp",
            "aligned_route_position",
            "aligned_arc_id",
            "distance_to_aligned_arc_m"
        ]
    ].to_string(index=False)
)

output_file = (
    PROCESSED_DIR /
    f"trip_{TRIP_ID}_missing_candidates.csv"
)

missing.to_csv(output_file, index=False)

print("\nSaved:", output_file)
