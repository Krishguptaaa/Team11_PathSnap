
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

EARTH_RADIUS_M = 6_371_000

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

    candidates = pd.read_csv(
        PROCESSED_DIR / f"trip_{trip_id}_candidates.csv"
    )

    return track, nodes, arcs, candidates


# ============================================================
# 3. GEOGRAPHIC DISTANCE
# ============================================================

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate geographic distance in metres."""

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return float(
        2 * EARTH_RADIUS_M * np.arcsin(
            np.sqrt(np.clip(a, 0, 1))
        )
    )


# ============================================================
# 4. BEARING CALCULATION
# ============================================================

def calculate_bearing(lon1, lat1, lon2, lat2):
    """
    Calculate compass bearing from one coordinate
    to another, in degrees clockwise from north.
    """

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    bearing = np.degrees(
        np.arctan2(x, y)
    )

    return float((bearing + 360) % 360)


def calculate_heading_difference(
    gps_bearing,
    road_bearing
):
    """Calculate the smallest directional difference."""

    difference = abs(gps_bearing - road_bearing)

    return min(difference, 360 - difference)


# ============================================================
# 5. LOCAL POINT-TO-SEGMENT PROJECTION
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
    Project a GPS point onto a road segment in a local
    metric coordinate system.

    Returns:
        distance_m: distance to projected point
        projection_fraction: position along segment [0, 1]
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

    length_squared = dx * dx + dy * dy

    if length_squared == 0:

        distance = np.sqrt(
            start_x ** 2 + start_y ** 2
        )

        return float(distance), 0.0

    fraction = -(
        start_x * dx + start_y * dy
    ) / length_squared

    fraction = float(
        np.clip(fraction, 0, 1)
    )

    projected_x = start_x + fraction * dx
    projected_y = start_y + fraction * dy

    distance = np.sqrt(
        projected_x ** 2 + projected_y ** 2
    )

    return float(distance), fraction


# ============================================================
# 6. GPS TEMPORAL FEATURES
# ============================================================

def calculate_gps_features(track):

    gps = track.copy()

    gps["previous_longitude"] = (
        gps["longitude"].shift(1)
    )

    gps["previous_latitude"] = (
        gps["latitude"].shift(1)
    )

    gps["previous_timestamp"] = (
        gps["timestamp"].shift(1)
    )

    gps["time_gap_s"] = (
        gps["timestamp"] - gps["previous_timestamp"]
    )

    gps["gps_displacement_m"] = np.nan
    gps["gps_bearing"] = np.nan
    gps["gps_speed_mps"] = np.nan

    for i in range(1, len(gps)):

        lon1 = gps.loc[i - 1, "longitude"]
        lat1 = gps.loc[i - 1, "latitude"]

        lon2 = gps.loc[i, "longitude"]
        lat2 = gps.loc[i, "latitude"]

        time_gap = gps.loc[i, "time_gap_s"]

        displacement = haversine_distance(
            lon1, lat1, lon2, lat2
        )

        gps.loc[i, "gps_displacement_m"] = displacement

        if displacement > 0:

            gps.loc[i, "gps_bearing"] = calculate_bearing(
                lon1, lat1, lon2, lat2
            )

        if pd.notna(time_gap) and time_gap > 0:

            gps.loc[i, "gps_speed_mps"] = (
                displacement / time_gap
            )

    # Keep the first observation's missing temporal features
    # as NaN for now. We will handle them consistently later.

    return gps


# ============================================================
# 7. ROAD SEGMENT FEATURES
# ============================================================

def calculate_segment_features(
    gps_row,
    candidate,
    nodes,
    arcs
):

    arc_id = int(candidate["arc_id"])

    arc = arcs.iloc[arc_id]

    from_node = int(arc["from_node_id"])
    to_node = int(arc["to_node_id"])

    start = nodes.iloc[from_node]
    end = nodes.iloc[to_node]

    start_lon = float(start["longitude"])
    start_lat = float(start["latitude"])

    end_lon = float(end["longitude"])
    end_lat = float(end["latitude"])

    # Road segment length
    segment_length = haversine_distance(
        start_lon,
        start_lat,
        end_lon,
        end_lat
    )

    # Road direction
    segment_bearing = calculate_bearing(
        start_lon,
        start_lat,
        end_lon,
        end_lat
    )

    # GPS-to-road heading difference
    gps_bearing = gps_row["gps_bearing"]

    if pd.notna(gps_bearing):
        heading_difference = calculate_heading_difference(
            gps_bearing,
            segment_bearing
        )
    else:
        heading_difference = np.nan

    # Projection of GPS point onto road segment
    projection_distance, projection_fraction = (
        project_point_onto_segment(
            gps_row["longitude"],
            gps_row["latitude"],
            start_lon,
            start_lat,
            end_lon,
            end_lat
        )
    )

    return {
        "segment_length_m": segment_length,
        "segment_bearing": segment_bearing,
        "heading_difference": heading_difference,
        "projection_distance_m": projection_distance,
        "projection_fraction": projection_fraction
    }


# ============================================================
# 8. BUILD FEATURE TABLE
# ============================================================

def extract_features(
    trip_id,
    track,
    nodes,
    arcs,
    candidates
):

    gps = calculate_gps_features(track)

    feature_records = []

    grouped_candidates = candidates.groupby(
        "gps_index",
        sort=True
    )

    print("\nExtracting candidate features...")

    for gps_index, candidate_group in grouped_candidates:

        gps_row = gps.iloc[int(gps_index)]

        for _, candidate in candidate_group.iterrows():

            segment_features = calculate_segment_features(
                gps_row,
                candidate,
                nodes,
                arcs
            )

            record = {
                "trip_id": trip_id,
                "gps_index": int(gps_index),
                "timestamp": gps_row["timestamp"],
                "candidate_rank": int(
                    candidate["candidate_rank"]
                ),
                "arc_id": int(candidate["arc_id"]),
                "distance_m": float(
                    candidate["distance_m"]
                ),
                "gps_speed_mps": gps_row["gps_speed_mps"],
                "gps_displacement_m": gps_row[
                    "gps_displacement_m"
                ],
                "time_gap_s": gps_row["time_gap_s"],
                "gps_bearing": gps_row["gps_bearing"],
                **segment_features
            }

            feature_records.append(record)

        if (int(gps_index) + 1) % 100 == 0:
            print(
                f"Processed GPS observation "
                f"{int(gps_index) + 1}/{len(track)}"
            )

    feature_df = pd.DataFrame(feature_records)

    return feature_df


# ============================================================
# 9. FEATURE INSPECTION
# ============================================================

def inspect_features(feature_df):

    print("\n" + "=" * 65)
    print("FEATURE EXTRACTION RESULTS")
    print("=" * 65)

    print("\nShape:", feature_df.shape)

    print("\nColumns:")
    print(feature_df.columns.tolist())

    print("\nFirst five records:")
    print(feature_df.head().to_string(index=False))

    numerical_features = [
        "distance_m",
        "segment_length_m",
        "segment_bearing",
        "heading_difference",
        "projection_distance_m",
        "projection_fraction",
        "gps_speed_mps",
        "gps_displacement_m",
        "time_gap_s"
    ]

    print("\nFeature descriptive statistics:")
    print(
        feature_df[numerical_features]
        .describe()
        .round(3)
    )

    print("\nMissing values:")
    print(feature_df[numerical_features].isnull().sum())


# ============================================================
# 10. SAVE FEATURE TABLE
# ============================================================

def save_features(feature_df, trip_id):

    output_file = (
        PROCESSED_DIR /
        f"trip_{trip_id}_features.csv"
    )

    feature_df.to_csv(
        output_file,
        index=False
    )

    print("\nFeature table saved to:")
    print(output_file)


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PATHSNAP - FEATURE EXTRACTION")
    print("=" * 65)

    track, nodes, arcs, candidates = load_trip_data(
        TRIP_ID
    )

    feature_df = extract_features(
        TRIP_ID,
        track,
        nodes,
        arcs,
        candidates
    )

    inspect_features(feature_df)

    save_features(
        feature_df,
        TRIP_ID
    )

    print("\nFeature extraction completed.")


if __name__ == "__main__":
    main()