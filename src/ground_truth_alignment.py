
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\Krishitha\OneDrive\Desktop\Student folder"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRIP_ID = "00000001"


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_trip_data(trip_id):

    trip_dir = DATA_ROOT / "train" / trip_id

    track = pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )

    route = pd.read_csv(
        trip_dir / f"{trip_id}.route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    return track, nodes, arcs, route


# ============================================================
# 3. DISTANCE AND PROJECTION
# ============================================================

def project_point_onto_segment(
    gps_lon,
    gps_lat,
    start_lon,
    start_lat,
    end_lon,
    end_lat
):
    """
    Project a GPS point onto a road segment.

    Returns:
        distance in metres,
        projection fraction from 0 to 1.
    """

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
# 4. BUILD REFERENCE ROUTE GEOMETRY
# ============================================================

def build_reference_route(nodes, arcs, route):

    node_coordinates = nodes[
        ["longitude", "latitude"]
    ].to_numpy(dtype=float)

    arc_array = arcs[
        ["from_node_id", "to_node_id"]
    ].to_numpy(dtype=int)

    route_arc_ids = route["arc_id"].to_numpy(dtype=int)

    route_segments = []

    cumulative_distance = 0.0

    for route_position, arc_id in enumerate(route_arc_ids):

        if arc_id < 0 or arc_id >= len(arc_array):
            raise ValueError(
                f"Invalid route arc ID: {arc_id}"
            )

        from_node, to_node = arc_array[arc_id]

        start = node_coordinates[from_node]
        end = node_coordinates[to_node]

        segment_length = np.sqrt(
            (
                (end[0] - start[0])
                * 111320
                * np.cos(np.radians(
                    (start[1] + end[1]) / 2
                ))
            ) ** 2
            +
            (
                (end[1] - start[1]) * 110574
            ) ** 2
        )

        route_segments.append({
            "route_position": route_position,
            "arc_id": int(arc_id),
            "from_node": int(from_node),
            "to_node": int(to_node),
            "start_lon": start[0],
            "start_lat": start[1],
            "end_lon": end[0],
            "end_lat": end[1],
            "segment_length_m": segment_length,
            "cumulative_start_m": cumulative_distance
        })

        cumulative_distance += segment_length

    route_geometry = pd.DataFrame(route_segments)

    return route_geometry


# ============================================================
# 5. CHECK ROUTE CONNECTIVITY
# ============================================================

def inspect_route_connectivity(route_geometry):

    from_nodes = route_geometry["from_node"].to_numpy()
    to_nodes = route_geometry["to_node"].to_numpy()

    connected = (
        to_nodes[:-1] == from_nodes[1:]
    )

    disconnected_count = int((~connected).sum())

    print("\n" + "=" * 65)
    print("ROUTE CONNECTIVITY")
    print("=" * 65)

    print("Route arcs:", len(route_geometry))
    print("Adjacent arc pairs:", len(connected))
    print("Connected adjacent pairs:", int(connected.sum()))
    print("Disconnected adjacent pairs:", disconnected_count)

    if disconnected_count > 0:

        positions = np.where(~connected)[0]

        print("\nFirst disconnected positions:")

        for position in positions[:20]:

            print(
                f"Route position {position}: "
                f"arc {route_geometry.iloc[position]['arc_id']} "
                f"ends at node {to_nodes[position]}, "
                f"but next arc starts at node "
                f"{from_nodes[position + 1]}"
            )

    return connected


# ============================================================
# 6. FIND CLOSEST REFERENCE ROUTE ARC FOR EACH GPS POINT
# ============================================================

def find_nearest_route_arc(track, route_geometry):

    results = []

    print("\nFinding nearest reference route positions...")

    for gps_index, gps in track.iterrows():

        best_distance = float("inf")
        best_segment = None
        best_fraction = None

        for _, segment in route_geometry.iterrows():

            distance, fraction = project_point_onto_segment(
                gps["longitude"],
                gps["latitude"],
                segment["start_lon"],
                segment["start_lat"],
                segment["end_lon"],
                segment["end_lat"]
            )

            if distance < best_distance:

                best_distance = distance
                best_segment = segment
                best_fraction = fraction

        results.append({
            "gps_index": gps_index,
            "timestamp": gps["timestamp"],
            "longitude": gps["longitude"],
            "latitude": gps["latitude"],
            "nearest_route_position": int(
                best_segment["route_position"]
            ),
            "nearest_route_arc_id": int(
                best_segment["arc_id"]
            ),
            "distance_to_route_m": best_distance,
            "projection_fraction": best_fraction
        })

        if (gps_index + 1) % 100 == 0:

            print(
                f"Processed {gps_index + 1}/"
                f"{len(track)} GPS observations"
            )

    return pd.DataFrame(results)


# ============================================================
# 7. ANALYSE CHRONOLOGICAL PROGRESSION
# ============================================================

def analyse_progression(alignment):

    positions = alignment[
        "nearest_route_position"
    ].to_numpy()

    differences = np.diff(positions)

    backward_steps = int((differences < 0).sum())
    forward_steps = int((differences > 0).sum())
    unchanged_steps = int((differences == 0).sum())

    print("\n" + "=" * 65)
    print("ROUTE PROGRESSION ANALYSIS")
    print("=" * 65)

    print("GPS observations:", len(positions))
    print("Backward route-position steps:", backward_steps)
    print("Forward route-position steps:", forward_steps)
    print("Unchanged route-position steps:", unchanged_steps)

    print("\nRoute position statistics:")

    print(
        alignment["nearest_route_position"]
        .describe()
    )

    print("\nDistance to reference route:")

    print(
        alignment["distance_to_route_m"]
        .describe()
        .round(3)
    )

    print("\nFirst 20 alignment records:")

    print(
        alignment.head(20).to_string(index=False)
    )

    print("\nLargest distances to route:")

    print(
        alignment.nlargest(
            10,
            "distance_to_route_m"
        )[
            [
                "gps_index",
                "nearest_route_position",
                "nearest_route_arc_id",
                "distance_to_route_m"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 8. SAVE DIAGNOSTIC RESULTS
# ============================================================

def save_results(route_geometry, alignment):

    route_file = (
        PROCESSED_DIR /
        f"trip_{TRIP_ID}_reference_route_geometry.csv"
    )

    alignment_file = (
        PROCESSED_DIR /
        f"trip_{TRIP_ID}_alignment_diagnostic.csv"
    )

    route_geometry.to_csv(
        route_file,
        index=False
    )

    alignment.to_csv(
        alignment_file,
        index=False
    )

    print("\nSaved reference route geometry:", route_file)
    print("Saved alignment diagnostic:", alignment_file)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PATHSNAP - GROUND TRUTH ALIGNMENT DIAGNOSTICS")
    print("=" * 65)

    track, nodes, arcs, route = load_trip_data(TRIP_ID)

    print("\nGPS observations:", len(track))
    print("Road nodes:", len(nodes))
    print("Road arcs:", len(arcs))
    print("Reference route arcs:", len(route))

    route_geometry = build_reference_route(
        nodes,
        arcs,
        route
    )

    inspect_route_connectivity(route_geometry)

    alignment = find_nearest_route_arc(
        track,
        route_geometry
    )

    analyse_progression(alignment)

    save_results(
        route_geometry,
        alignment
    )

    print("\nAlignment diagnostics completed.")
    print("No training labels have been assigned yet.")


if __name__ == "__main__":
    main()
