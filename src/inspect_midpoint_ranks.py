
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:\Users\Krishitha\OneDrive\Desktop\Student folder")

TRIP_ID = "00000001"

track = pd.read_csv(
    DATA / "train" / TRIP_ID / f"{TRIP_ID}.track",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude", "timestamp"]
)

nodes = pd.read_csv(
    DATA / "train" / TRIP_ID / f"{TRIP_ID}.nodes",
    sep=r"\s+",
    header=None,
    names=["longitude", "latitude"]
)

arcs = pd.read_csv(
    DATA / "train" / TRIP_ID / f"{TRIP_ID}.arcs",
    sep=r"\s+",
    header=None,
    names=["from_node_id", "to_node_id"]
)

missing = pd.read_csv(
    ROOT / "data" / "processed" /
    f"trip_{TRIP_ID}_missing_candidates.csv"
)

# Construct road-segment midpoints.
start = nodes.iloc[arcs["from_node_id"].to_numpy()][
    ["longitude", "latitude"]
].to_numpy()

end = nodes.iloc[arcs["to_node_id"].to_numpy()][
    ["longitude", "latitude"]
].to_numpy()

midpoints = (start + end) / 2

# BallTree expects coordinates in radians, ordered latitude, longitude.
midpoint_coords = np.radians(midpoints[:, [1, 0]])
tree = BallTree(midpoint_coords, metric="haversine")

results = []

for _, row in missing.iterrows():
    gps = track.iloc[int(row["gps_index"])]

    gps_coord = np.radians([[
        gps["latitude"], gps["longitude"]
    ]])

    # Find the rank of the aligned arc among ALL arc midpoints.
    distances, indices = tree.query(
        gps_coord,
        k=len(arcs)
    )

    aligned_arc_id = int(row["aligned_arc_id"])
    positions = np.where(indices[0] == aligned_arc_id)[0]

    midpoint_rank = int(positions[0] + 1) if len(positions) else -1
    midpoint_distance_m = (
        distances[0][midpoint_rank - 1] * 6371000
        if midpoint_rank > 0 else np.nan
    )

    results.append({
        "gps_index": int(row["gps_index"]),
        "aligned_arc_id": aligned_arc_id,
        "segment_distance_m": row["distance_to_aligned_arc_m"],
        "midpoint_rank": midpoint_rank,
        "midpoint_distance_m": midpoint_distance_m,
        "inside_top_200": midpoint_rank <= 200
    })

result = pd.DataFrame(results)

print("\nMIDPOINT RETRIEVAL DIAGNOSTICS")
print("=" * 75)
print(result.to_string(index=False))

print("\nSummary:")
print("Missing arcs inside Top-200 midpoint pool:",
      result["inside_top_200"].sum(), "/", len(result))
print("Missing arcs outside Top-200 midpoint pool:",
      (~result["inside_top_200"]).sum(), "/", len(result))

output = ROOT / "data" / "processed" / "trip_00000001_midpoint_rank_diagnostic.csv"
result.to_csv(output, index=False)
print(f"\nSaved: {output}")
