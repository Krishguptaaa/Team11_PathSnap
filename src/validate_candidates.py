
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(
    r"C:\Users\Krishitha\OneDrive\Desktop\Student folder"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRIP_ID = "00000001"

trip_dir = DATA_ROOT / "train" / TRIP_ID
processed_dir = PROJECT_ROOT / "data" / "processed"

# Load ground-truth route
route = pd.read_csv(
    trip_dir / f"{TRIP_ID}.route",
    sep=r"\s+",
    header=None,
    names=["arc_id"]
)

# Load generated candidates
candidates = pd.read_csv(
    processed_dir / f"trip_{TRIP_ID}_candidates.csv"
)

# All candidate arc IDs
candidate_arc_ids = set(
    candidates["arc_id"].astype(int)
)

# Ground-truth route arc IDs
route_arc_ids = route["arc_id"].astype(int).tolist()

# Route-level coverage:
# Does each reference route arc appear anywhere
# in the candidate set for this trip?
covered = [
    arc_id in candidate_arc_ids
    for arc_id in route_arc_ids
]

coverage_count = sum(covered)
total_route_arcs = len(route_arc_ids)

coverage_percentage = (
    coverage_count / total_route_arcs * 100
    if total_route_arcs else 0
)

print("=" * 60)
print("GROUND-TRUTH ROUTE CANDIDATE COVERAGE")
print("=" * 60)

print("Total route arcs:", total_route_arcs)
print("Covered route arcs:", coverage_count)
print("Missing route arcs:", total_route_arcs - coverage_count)
print(f"Route-level candidate coverage: {coverage_percentage:.2f}%")

# Save individual route arc coverage
coverage_df = pd.DataFrame({
    "route_position": range(len(route_arc_ids)),
    "arc_id": route_arc_ids,
    "found_in_candidates": covered
})

coverage_file = (
    processed_dir / f"trip_{TRIP_ID}_route_coverage.csv"
)

coverage_df.to_csv(coverage_file, index=False)

print("\nMissing route arc IDs:")
print(
    coverage_df.loc[
        ~coverage_df["found_in_candidates"],
        "arc_id"
    ].tolist()[:30]
)

print("\nCoverage report saved to:", coverage_file)

# Candidate distance statistics by GPS observation
print("\nCandidate distance statistics:")

print(
    candidates.groupby("gps_index")["distance_m"]
    .min()
    .describe()
)

print("\nCandidate coverage validation completed.")
