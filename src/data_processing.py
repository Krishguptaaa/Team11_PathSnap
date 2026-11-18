
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import folium


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\Krishitha\OneDrive\Desktop\Student folder"
)

TRAIN_DIR = DATA_ROOT / "train"
STUDENT_TEST_DIR = DATA_ROOT / "student_test"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEMAS = {
    "track": ["longitude", "latitude", "timestamp"],
    "nodes": ["longitude", "latitude"],
    "arcs": ["from_node_id", "to_node_id"],
    "route": ["arc_id"],
}


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_file(file_path, columns):
    """Load a whitespace-separated dataset."""

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=columns
    )

    return df


def load_trip(trip_dir):
    """Load all four files belonging to one trip."""

    trip_id = trip_dir.name

    trip = {
        "trip_id": trip_id,
        "track": load_file(
            trip_dir / f"{trip_id}.track",
            SCHEMAS["track"]
        ),
        "nodes": load_file(
            trip_dir / f"{trip_id}.nodes",
            SCHEMAS["nodes"]
        ),
        "arcs": load_file(
            trip_dir / f"{trip_id}.arcs",
            SCHEMAS["arcs"]
        ),
        "route": load_file(
            trip_dir / f"{trip_id}.route",
            SCHEMAS["route"]
        )
    }

    return trip


def get_trip_directories(partition_dir):
    """Return sorted trip directories from a partition."""

    if not partition_dir.exists():
        raise FileNotFoundError(
            f"Partition directory not found: {partition_dir}"
        )

    return sorted(
        path for path in partition_dir.iterdir()
        if path.is_dir()
    )


# ============================================================
# 3. BASIC DATA INSPECTION
# ============================================================

def inspect_trip(trip):
    """Print basic information about one trip."""

    print("\n" + "=" * 65)
    print(f"TRIP ID: {trip['trip_id']}")
    print("=" * 65)

    for name in ["track", "nodes", "arcs", "route"]:
        df = trip[name]

        print(f"\n{name.upper()}")
        print("-" * 40)
        print("Shape:", df.shape)
        print("\nData types:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())
        print("\nFirst five rows:")
        print(df.head())


# ============================================================
# 4. DATA VALIDATION
# ============================================================

def validate_trip(trip):
    """Check data quality without modifying the original data."""

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]

    # GPS coordinate validation
    valid_gps = (
        np.isfinite(track["longitude"])
        & np.isfinite(track["latitude"])
        & track["longitude"].between(-180, 180)
        & track["latitude"].between(-90, 90)
    )

    invalid_gps = int((~valid_gps).sum())

    # Road node validation
    valid_nodes = (
        np.isfinite(nodes["longitude"])
        & np.isfinite(nodes["latitude"])
        & nodes["longitude"].between(-180, 180)
        & nodes["latitude"].between(-90, 90)
    )

    invalid_nodes = int((~valid_nodes).sum())

    # Arc endpoint validation
    valid_arcs = (
        arcs["from_node_id"].between(0, len(nodes) - 1)
        & arcs["to_node_id"].between(0, len(nodes) - 1)
    )

    invalid_arcs = int((~valid_arcs).sum())

    # Route arc ID validation
    valid_routes = route["arc_id"].between(0, len(arcs) - 1)

    invalid_routes = int((~valid_routes).sum())

    # Duplicate GPS observations
    duplicate_gps = int(
        track.duplicated(
            subset=["longitude", "latitude", "timestamp"]
        ).sum()
    )

    # Timestamp validation
    time_differences = track["timestamp"].diff().dropna()

    non_increasing_timestamps = int(
        (time_differences <= 0).sum()
    )

    gaps_over_1s = int((time_differences > 1).sum())
    gaps_over_2s = int((time_differences > 2).sum())
    gaps_over_5s = int((time_differences > 5).sum())
    gaps_over_10s = int((time_differences > 10).sum())

    result = {
        "trip_id": trip["trip_id"],
        "gps_points": len(track),
        "road_nodes": len(nodes),
        "road_arcs": len(arcs),
        "route_arcs": len(route),
        "invalid_gps": invalid_gps,
        "invalid_nodes": invalid_nodes,
        "invalid_arcs": invalid_arcs,
        "invalid_route_ids": invalid_routes,
        "duplicate_gps": duplicate_gps,
        "non_increasing_timestamps": non_increasing_timestamps,
        "gaps_over_1s": gaps_over_1s,
        "gaps_over_2s": gaps_over_2s,
        "gaps_over_5s": gaps_over_5s,
        "gaps_over_10s": gaps_over_10s,
        "mean_time_interval": (
            float(time_differences.mean())
            if len(time_differences) else np.nan
        ),
        "median_time_interval": (
            float(time_differences.median())
            if len(time_differences) else np.nan
        ),
        "max_time_interval": (
            float(time_differences.max())
            if len(time_differences) else np.nan
        )
    }

    return result


# ============================================================
# 5. PROCESS A COMPLETE PARTITION
# ============================================================

def process_partition(partition_dir, partition_name):
    """Load and validate every trip in a partition."""

    trip_dirs = get_trip_directories(partition_dir)

    results = []

    print("\n" + "=" * 65)
    print(f"PROCESSING PARTITION: {partition_name}")
    print("Number of trips:", len(trip_dirs))
    print("=" * 65)

    for index, trip_dir in enumerate(trip_dirs, start=1):

        try:
            trip = load_trip(trip_dir)
            validation = validate_trip(trip)
            results.append(validation)

            print(
                f"[{index}/{len(trip_dirs)}] "
                f"Trip {trip['trip_id']} processed"
            )

        except Exception as error:
            print(
                f"Error processing {trip_dir.name}: {error}"
            )

    summary = pd.DataFrame(results)

    output_file = (
        PROCESSED_DIR / f"{partition_name}_summary.csv"
    )

    summary.to_csv(output_file, index=False)

    print("\nSummary saved to:", output_file)

    return summary


# ============================================================
# 6. TIMESTAMP GAP ANALYSIS
# ============================================================

def analyse_timestamp_gaps(partition_dir, partition_name):
    """Analyse the distribution of time gaps in every trip."""

    results = []

    trip_dirs = get_trip_directories(partition_dir)

    print("\nAnalysing timestamp gaps...")

    for trip_dir in trip_dirs:

        trip_id = trip_dir.name

        try:
            track = load_file(
                trip_dir / f"{trip_id}.track",
                SCHEMAS["track"]
            )

            differences = (
                track["timestamp"].diff().dropna()
            )

            results.append({
                "trip_id": trip_id,
                "observations": len(track),
                "mean_gap": (
                    differences.mean()
                    if len(differences) else np.nan
                ),
                "median_gap": (
                    differences.median()
                    if len(differences) else np.nan
                ),
                "max_gap": (
                    differences.max()
                    if len(differences) else np.nan
                ),
                "gaps_over_2s": int(
                    (differences > 2).sum()
                ),
                "gaps_over_5s": int(
                    (differences > 5).sum()
                ),
                "gaps_over_10s": int(
                    (differences > 10).sum()
                )
            })

        except Exception as error:
            print(f"Error in trip {trip_id}: {error}")

    result_df = pd.DataFrame(results)

    output_file = (
        PROCESSED_DIR
        / f"{partition_name}_timestamp_analysis.csv"
    )

    result_df.to_csv(output_file, index=False)

    print("\nTimestamp analysis saved:", output_file)

    if not result_df.empty:
        print("\nTrips with the largest timestamp gaps:")
        print(
            result_df.sort_values(
                "max_gap",
                ascending=False
            ).head(10).to_string(index=False)
        )

    return result_df


# ============================================================
# 7. ROAD NETWORK ANALYSIS
# ============================================================

def analyse_road_network(trip):
    """Analyse the directed road graph for one trip."""

    nodes = trip["nodes"]
    arcs = trip["arcs"]

    graph = nx.DiGraph()

    # Add all road nodes
    graph.add_nodes_from(range(len(nodes)))

    # Add directed road connections
    valid_arcs = arcs[
        arcs["from_node_id"].between(0, len(nodes) - 1)
        & arcs["to_node_id"].between(0, len(nodes) - 1)
    ]

    edges = list(
        zip(
            valid_arcs["from_node_id"].astype(int),
            valid_arcs["to_node_id"].astype(int)
        )
    )

    graph.add_edges_from(edges)

    # Weakly connected components
    components = list(
        nx.weakly_connected_components(graph)
    )

    largest_component_size = (
        max(map(len, components))
        if components else 0
    )

    # Isolated nodes
    isolated_nodes = list(nx.isolates(graph))

    # Degree statistics
    in_degrees = [
        degree for _, degree in graph.in_degree()
    ]

    out_degrees = [
        degree for _, degree in graph.out_degree()
    ]

    statistics = {
        "trip_id": trip["trip_id"],
        "number_of_nodes": graph.number_of_nodes(),
        "number_of_edges": graph.number_of_edges(),
        "weakly_connected_components": len(components),
        "largest_component_size": largest_component_size,
        "isolated_nodes": len(isolated_nodes),
        "average_in_degree": (
            np.mean(in_degrees) if in_degrees else 0
        ),
        "average_out_degree": (
            np.mean(out_degrees) if out_degrees else 0
        )
    }

    print("\n" + "=" * 65)
    print("ROAD NETWORK ANALYSIS")
    print("=" * 65)

    for key, value in statistics.items():
        print(f"{key}: {value}")

    return graph, statistics


# ============================================================
# 8. GPS AND REFERENCE ROUTE VISUALIZATION
# ============================================================

def visualize_trip(trip):
    """Create a Folium map showing GPS points and reference route."""

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]

    # Centre the map on the GPS observations
    center_latitude = track["latitude"].mean()
    center_longitude = track["longitude"].mean()

    map_object = folium.Map(
        location=[
            center_latitude,
            center_longitude
        ],
        zoom_start=14,
        control_scale=True
    )

    # Raw GPS coordinates
    gps_points = list(
        zip(
            track["latitude"],
            track["longitude"]
        )
    )

    folium.PolyLine(
        gps_points,
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="Raw GPS trajectory"
    ).add_to(map_object)

    # Individual GPS markers
    if len(gps_points) > 0:
        folium.CircleMarker(
            location=gps_points[0],
            radius=5,
            color="blue",
            fill=True,
            popup="GPS Start"
        ).add_to(map_object)

        folium.CircleMarker(
            location=gps_points[-1],
            radius=5,
            color="red",
            fill=True,
            popup="GPS End"
        ).add_to(map_object)

    # Reconstruct the reference route using route arc IDs.
    # Arc IDs refer to row indices in the arcs file.
    route_points = []
    invalid_route_ids = 0

    for arc_id in route["arc_id"]:

        if pd.isna(arc_id):
            invalid_route_ids += 1
            continue

        arc_id = int(arc_id)

        if arc_id < 0 or arc_id >= len(arcs):
            invalid_route_ids += 1
            continue

        from_node = int(
            arcs.iloc[arc_id]["from_node_id"]
        )

        to_node = int(
            arcs.iloc[arc_id]["to_node_id"]
        )

        if (
            from_node < 0
            or from_node >= len(nodes)
            or to_node < 0
            or to_node >= len(nodes)
        ):
            invalid_route_ids += 1
            continue

        # Store the start node of each route arc
        node = nodes.iloc[from_node]

        route_points.append(
            (node["latitude"], node["longitude"])
        )

    # Add the final destination node
    if len(route) > 0 and invalid_route_ids == 0:

        last_arc_id = int(route["arc_id"].iloc[-1])

        if 0 <= last_arc_id < len(arcs):

            last_to_node = int(
                arcs.iloc[last_arc_id]["to_node_id"]
            )

            if 0 <= last_to_node < len(nodes):

                final_node = nodes.iloc[last_to_node]

                route_points.append(
                    (
                        final_node["latitude"],
                        final_node["longitude"]
                    )
                )

    if len(route_points) > 1:

        folium.PolyLine(
            route_points,
            color="green",
            weight=5,
            opacity=0.8,
            tooltip="Reference route"
        ).add_to(map_object)

    folium.LayerControl().add_to(map_object)

    output_file = (
        OUTPUT_DIR / f"trip_{trip['trip_id']}.html"
    )

    map_object.save(str(output_file))

    print("\nMap saved to:", output_file)
    print("GPS points:", len(gps_points))
    print("Reference route points:", len(route_points))
    print("Invalid route IDs:", invalid_route_ids)

    return map_object


# ============================================================
# 9. PARTITION-LEVEL SUMMARY
# ============================================================

def print_partition_summary(summary, partition_name):
    """Print descriptive statistics for a partition."""

    if summary.empty:
        print(f"No data available for {partition_name}")
        return

    columns = [
        "gps_points",
        "road_nodes",
        "road_arcs",
        "route_arcs",
        "gaps_over_2s",
        "gaps_over_5s",
        "gaps_over_10s",
        "median_time_interval",
        "max_time_interval"
    ]

    print("\n" + "=" * 65)
    print(f"{partition_name.upper()} DESCRIPTIVE STATISTICS")
    print("=" * 65)

    print(
        summary[columns].describe().round(3)
    )


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():

    print("\nPATHSNAP - DATA PROCESSING")
    print("=" * 65)

    # --------------------------------------------------------
    # A. Load and inspect one training trip
    # --------------------------------------------------------

    train_trip_dirs = get_trip_directories(TRAIN_DIR)

    if not train_trip_dirs:
        raise RuntimeError("No training trips found.")

    first_trip = load_trip(train_trip_dirs[0])

    inspect_trip(first_trip)

    first_trip_validation = validate_trip(first_trip)

    print("\nFIRST TRIP VALIDATION")
    print("-" * 40)

    for key, value in first_trip_validation.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # B. Validate all training trips
    # --------------------------------------------------------

    train_summary = process_partition(
        TRAIN_DIR,
        "train"
    )

    print_partition_summary(
        train_summary,
        "Training"
    )

    # --------------------------------------------------------
    # C. Validate all student-testing trips
    # --------------------------------------------------------

    student_summary = process_partition(
        STUDENT_TEST_DIR,
        "student_test"
    )

    print_partition_summary(
        student_summary,
        "Student testing"
    )

    # --------------------------------------------------------
    # D. Analyse timestamp gaps
    # --------------------------------------------------------

    train_timestamp_analysis = analyse_timestamp_gaps(
        TRAIN_DIR,
        "train"
    )

    student_timestamp_analysis = analyse_timestamp_gaps(
        STUDENT_TEST_DIR,
        "student_test"
    )

    # --------------------------------------------------------
    # E. Analyse road network of first training trip
    # --------------------------------------------------------

    graph, graph_statistics = analyse_road_network(
        first_trip
    )

    graph_output = (
        PROCESSED_DIR / "first_trip_graph_statistics.csv"
    )

    pd.DataFrame([graph_statistics]).to_csv(
        graph_output,
        index=False
    )

    print("\nGraph statistics saved:", graph_output)

    # --------------------------------------------------------
    # F. Visualize first training trip
    # --------------------------------------------------------

    visualize_trip(first_trip)

    # --------------------------------------------------------
    # G. Final completion summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("DATA PROCESSING COMPLETED")
    print("=" * 65)

    print("Training trips processed:", len(train_summary))
    print("Student-testing trips processed:", len(student_summary))

    print("\nGenerated files:")
    print("1. data/processed/train_summary.csv")
    print("2. data/processed/student_test_summary.csv")
    print("3. data/processed/train_timestamp_analysis.csv")
    print("4. data/processed/student_test_timestamp_analysis.csv")
    print("5. data/processed/first_trip_graph_statistics.csv")
    print("6. outputs/trip_<trip_id>.html")

    print("\nNo evaluator-testing data was accessed.")
    print("No observations were removed or modified.")


if __name__ == "__main__":
    main()
