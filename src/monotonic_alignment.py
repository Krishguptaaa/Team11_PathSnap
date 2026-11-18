
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\Krishitha\OneDrive\Desktop\Student folder"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRIP_ID = "00000001"


# ============================================================
# LOAD DATA
# ============================================================

def load_trip_data():

    trip_dir = DATA_ROOT / "train" / TRIP_ID

    track = pd.read_csv(
        trip_dir / f"{TRIP_ID}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        trip_dir / f"{TRIP_ID}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        trip_dir / f"{TRIP_ID}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )

    route = pd.read_csv(
        trip_dir / f"{TRIP_ID}.route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    return track, nodes, arcs, route


# ============================================================
# DISTANCE AND PROJECTION
# ============================================================

def project_point_onto_segment(
    gps_lon,
    gps_lat,
    start_lon,
    start_lat,
    end_lon,
    end_lat
):

    longitude_scale = (
        111320 * np.cos(np.radians(gps_lat))
    )

    latitude_scale = 110574

    start_x = (start_lon - gps_lon) * longitude_scale
    start_y = (start_lat - gps_lat) * latitude_scale

    end_x = (end_lon - gps_lon) * longitude_scale
    end_y = (end_lat - gps_lat) * latitude_scale

    dx = end_x - start_x
    dy = end_y - start_y

    length_squared = dx ** 2 + dy ** 2

    if length_squared == 0:

        distance = np.sqrt(
            start_x ** 2 + start_y ** 2
        )

        return float(distance), 0.0

    fraction = -(
        start_x * dx + start_y * dy
    ) / length_squared

    fraction = float(np.clip(fraction, 0, 1))

    projected_x = start_x + fraction * dx
    projected_y = start_y + fraction * dy

    distance = np.sqrt(
        projected_x ** 2 + projected_y ** 2
    )

    return float(distance), fraction


# ============================================================
# BUILD REFERENCE ROUTE
# ============================================================

def build_route_geometry(nodes, arcs, route):

    coordinates = nodes[
        ["longitude", "latitude"]
    ].to_numpy(dtype=float)

    arc_array = arcs[
        ["from_node_id", "to_node_id"]
    ].to_numpy(dtype=int)

    route_arc_ids = route["arc_id"].to_numpy(dtype=int)

    segments = []

    for position, arc_id in enumerate(route_arc_ids):

        from_node, to_node = arc_array[arc_id]

        start = coordinates[from_node]
        end = coordinates[to_node]

        segments.append({
            "route_position": position,
            "arc_id": int(arc_id),
            "from_node": int(from_node),
            "to_node": int(to_node),
            "start_lon": start[0],
            "start_lat": start[1],
            "end_lon": end[0],
            "end_lat": end[1]
        })

    return pd.DataFrame(segments)


# ============================================================
# BUILD GPS-TO-ROUTE DISTANCE MATRIX
# ============================================================

def build_distance_matrix(track, route_geometry):

    gps_count = len(track)
    route_count = len(route_geometry)

    distances = np.zeros(
        (gps_count, route_count),
        dtype=np.float32
    )

    fractions = np.zeros(
        (gps_count, route_count),
        dtype=np.float32
    )

    print("\nCalculating GPS-to-route distances...")

    for i, gps in track.iterrows():

        for j, segment in route_geometry.iterrows():

            distance, fraction = project_point_onto_segment(
                gps["longitude"],
                gps["latitude"],
                segment["start_lon"],
                segment["start_lat"],
                segment["end_lon"],
                segment["end_lat"]
            )

            distances[i, j] = distance
            fractions[i, j] = fraction

        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{gps_count}"
            )

    return distances, fractions


# ============================================================
# MONOTONIC DYNAMIC PROGRAMMING
# ============================================================

def monotonic_alignment(distances):

    gps_count, route_count = distances.shape

    dp = np.full(
        (gps_count, route_count),
        np.inf,
        dtype=np.float64
    )

    backpointer = np.full(
        (gps_count, route_count),
        -1,
        dtype=np.int32
    )

    # First GPS observation can match any route segment
    dp[0, :] = distances[0, :] ** 2

    # Subsequent observations must stay at or advance
    # beyond the previous route position.
    for i in range(1, gps_count):

        best_cost = np.inf
        best_previous_position = -1

        for j in range(route_count):

            previous_cost = dp[i - 1, j]

            if previous_cost < best_cost:
                best_cost = previous_cost
                best_previous_position = j

            dp[i, j] = (
                distances[i, j] ** 2 + best_cost
            )

            backpointer[i, j] = best_previous_position

    # Find the optimal final route position
    final_position = int(np.argmin(dp[-1, :]))

    aligned_positions = np.zeros(
        gps_count,
        dtype=int
    )

    aligned_positions[-1] = final_position

    # Backtrack through the dynamic programming table
    for i in range(gps_count - 1, 0, -1):

        aligned_positions[i - 1] = (
            backpointer[i, aligned_positions[i]]
        )

    return aligned_positions


# ============================================================
# BUILD ALIGNMENT OUTPUT
# ============================================================

def build_alignment_output(
    track,
    route_geometry,
    distances,
    fractions,
    aligned_positions
):

    records = []

    for i, position in enumerate(aligned_positions):

        segment = route_geometry.iloc[position]

        records.append({
            "trip_id": TRIP_ID,
            "gps_index": i,
            "timestamp": track.iloc[i]["timestamp"],
            "longitude": track.iloc[i]["longitude"],
            "latitude": track.iloc[i]["latitude"],
            "aligned_route_position": int(position),
            "aligned_arc_id": int(segment["arc_id"]),
            "distance_to_aligned_arc_m": float(
                distances[i, position]
            ),
            "projection_fraction": float(
                fractions[i, position]
            )
        })

    return pd.DataFrame(records)


# ============================================================
# EVALUATE ALIGNMENT
# ============================================================

def evaluate_alignment(alignment):

    positions = alignment[
        "aligned_route_position"
    ].to_numpy()

    differences = np.diff(positions)

    print("\n" + "=" * 65)
    print("MONOTONIC ALIGNMENT RESULTS")
    print("=" * 65)

    print("GPS observations:", len(alignment))

    print(
        "Backward transitions:",
        int((differences < 0).sum())
    )

    print(
        "Forward transitions:",
        int((differences > 0).sum())
    )

    print(
        "Unchanged transitions:",
        int((differences == 0).sum())
    )

    print("\nDistance statistics:")

    print(
        alignment[
            "distance_to_aligned_arc_m"
        ].describe().round(3)
    )

    print("\nFirst 20 aligned observations:")

    print(
        alignment.head(20).to_string(index=False)
    )

    print("\nLargest alignment distances:")

    print(
        alignment.nlargest(
            10,
            "distance_to_aligned_arc_m"
        )[
            [
                "gps_index",
                "aligned_route_position",
                "aligned_arc_id",
                "distance_to_aligned_arc_m"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE ALIGNMENT
# ============================================================

def save_alignment(alignment):

    output_file = (
        PROCESSED_DIR /
        f"trip_{TRIP_ID}_monotonic_alignment.csv"
    )

    alignment.to_csv(
        output_file,
        index=False
    )

    print("\nAlignment saved to:")
    print(output_file)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PATHSNAP - MONOTONIC ROUTE ALIGNMENT")
    print("=" * 65)

    track, nodes, arcs, route = load_trip_data()

    route_geometry = build_route_geometry(
        nodes,
        arcs,
        route
    )

    distances, fractions = build_distance_matrix(
        track,
        route_geometry
    )

    aligned_positions = monotonic_alignment(
        distances
    )

    alignment = build_alignment_output(
        track,
        route_geometry,
        distances,
        fractions,
        aligned_positions
    )

    evaluate_alignment(alignment)

    save_alignment(alignment)

    print("\nMonotonic alignment completed.")
    print("These labels are provisional.")


if __name__ == "__main__":
    main()