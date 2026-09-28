import streamlit as st
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ------------------------------------------------------------
# EV Charging Recommendation Prototype
# Original implementation for academic demonstration.
# ------------------------------------------------------------

st.set_page_config(
    page_title="EV Charging Recommender",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ EV Charging Station Recommendation")
st.caption("Personalized station ranking using driver history, charging preference and location.")

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_table(base_name):
    """Load a CSV or Excel file with the requested base name."""
    for extension in (".csv", ".xlsx", ".xls"):
        file_path = f"{base_name}{extension}"
        try:
            if extension == ".csv":
                return pd.read_csv(file_path)
            return pd.read_excel(file_path)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"Could not find {base_name}.csv, {base_name}.xlsx or {base_name}.xls"
    )


@st.cache_data
def load_data():
    station_data = load_table("stations")
    driver_data = load_table("drivers")
    interaction_data = load_table("interactions")

    return station_data, driver_data, interaction_data


try:
    stations, drivers, interactions = load_data()
except Exception as error:
    st.error("The application could not load its data files.")
    st.write(str(error))
    st.stop()


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def distance_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two coordinates."""
    earth_radius = 6371.0

    lat1, lat2 = radians(float(lat1)), radians(float(lat2))
    delta_lat = radians(float(lat2) - float(lat1))
    delta_lon = radians(float(lon2) - float(lon1))

    value = (
        sin(delta_lat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
    )

    return 2 * earth_radius * atan2(sqrt(value), sqrt(1 - value))


def preference_score(driver_id, station_id):
    """Convert historical station usage into a normalized preference value."""
    history = interactions[
        (interactions["driver_id"] == driver_id)
        & (interactions["station_id"] == station_id)
    ]

    if history.empty:
        return 0.0

    usage = float(history["usage_count"].sum())
    return min(usage / 5.0, 1.0)


def recommend(driver_id, charging_type, number_of_results):
    """Generate and rank candidate charging stations."""
    driver = drivers.loc[drivers["driver_id"] == driver_id].iloc[0]

    preferred_cluster = driver["preferred_cluster"]
    cluster_stations = stations[
        stations["cluster"] == preferred_cluster
    ]

    # Use the preferred cluster's centre as the reference location.
    reference_lat = cluster_stations["latitude"].mean()
    reference_lon = cluster_stations["longitude"].mean()

    candidates = []

    for _, station in stations.iterrows():
        distance = distance_km(
            reference_lat,
            reference_lon,
            station["latitude"],
            station["longitude"]
        )

        historical_preference = preference_score(
            driver_id,
            station["station_id"]
        )

        cluster_match = (
            1.0 if station["cluster"] == preferred_cluster else 0.0
        )

        charging_match = (
            1.0 if station["charging_type"] == charging_type else 0.0
        )

        # Smooth distance contribution.
        proximity = np.exp(-distance / 2.0)

        # Weighted ranking model.
        score = (
            0.50 * historical_preference
            + 0.25 * cluster_match
            + 0.15 * charging_match
            + 0.10 * proximity
        )

        explanation = []

        if charging_match:
            explanation.append("preferred charging type")

        if cluster_match:
            explanation.append("preferred area")

        if historical_preference >= 0.60:
            explanation.append("strong previous usage")
        elif historical_preference > 0:
            explanation.append("previously used")

        if distance <= 2:
            explanation.append("near the reference location")

        if not explanation:
            explanation.append("matches the ranking criteria")

        candidates.append({
            "station_id": station["station_id"],
            "station_name": station["station_name"],
            "provider": station["provider"],
            "charging_type": station["charging_type"],
            "cluster": station["cluster"],
            "distance_km": distance,
            "preference": historical_preference,
            "score": score,
            "reason": " • ".join(explanation)
        })

    result = pd.DataFrame(candidates)
    result = result.sort_values(
        by="score",
        ascending=False
    ).head(number_of_results)

    result = result.reset_index(drop=True)
    result.index = result.index + 1

    return result


# ------------------------------------------------------------
# User controls
# ------------------------------------------------------------

st.sidebar.header("⚙️ Recommendation Settings")

driver_options = drivers["driver_id"].tolist()

selected_driver = st.sidebar.selectbox(
    "Select Driver",
    driver_options
)

selected_driver_row = drivers[
    drivers["driver_id"] == selected_driver
].iloc[0]

context_options = ["Home", "Workplace", "Shopping"]
default_context = (
    selected_driver_row["context"]
    if selected_driver_row["context"] in context_options
    else context_options[0]
)

selected_context = st.sidebar.selectbox(
    "Current Context",
    context_options,
    index=context_options.index(default_context)
)

charging_options = ["FAST", "SLOW"]
default_charging = (
    selected_driver_row["charging_preference"]
    if selected_driver_row["charging_preference"] in charging_options
    else charging_options[0]
)

selected_charging = st.sidebar.selectbox(
    "Charging Type",
    charging_options,
    index=charging_options.index(default_charging)
)

result_count = st.sidebar.slider(
    "Number of Recommendations",
    min_value=3,
    max_value=10,
    value=5
)

generate = st.sidebar.button(
    "🔎 Find Charging Stations",
    use_container_width=True
)


# ------------------------------------------------------------
# Recommendation output
# ------------------------------------------------------------

if generate or "recommendations" not in st.session_state:
    st.session_state["recommendations"] = recommend(
        selected_driver,
        selected_charging,
        result_count
    )

recommendations = st.session_state["recommendations"]

st.subheader("Selected Preferences")

a, b, c, d = st.columns(4)

a.metric("Driver", selected_driver)
b.metric("Context", selected_context)
c.metric("Charging", selected_charging)
d.metric("Results", result_count)

st.divider()

st.subheader("Recommended Charging Stations")

table = recommendations[
    [
        "station_id",
        "station_name",
        "provider",
        "charging_type",
        "cluster",
        "distance_km",
        "preference",
        "score",
        "reason"
    ]
].copy()

table["distance_km"] = table["distance_km"].round(2)
table["preference"] = table["preference"].round(2)
table["score"] = table["score"].round(3)

table.columns = [
    "Station ID",
    "Station",
    "Provider",
    "Charging",
    "Area",
    "Distance (km)",
    "Preference",
    "Score",
    "Why recommended"
]

st.dataframe(
    table,
    use_container_width=True,
    hide_index=False
)

st.divider()

st.subheader("Recommendation Details")

for rank, (_, station) in enumerate(recommendations.iterrows(), start=1):
    with st.expander(
        f"#{rank}  {station['station_id']} — {station['station_name']}"
    ):
        left, right = st.columns(2)

        left.write(f"**Provider:** {station['provider']}")
        left.write(f"**Charging type:** {station['charging_type']}")
        left.write(f"**Area:** {station['cluster']}")

        right.write(f"**Distance:** {station['distance_km']:.2f} km")
        right.write(f"**Preference score:** {station['preference']:.2f}")
        right.write(f"**Final score:** {station['score']:.3f}")

        st.info(
            f"Reason: {station['reason']}"
        )

st.caption("EV charging recommendation prototype")
