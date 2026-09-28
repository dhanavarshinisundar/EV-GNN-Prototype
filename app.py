
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="EV GNN Charging Recommendation", page_icon="⚡", layout="wide")

st.title("⚡ EV Charging Station Recommendation")
st.caption("Graph-based preference + spatial clustering + charging-type refinement")

stations = pd.read_csv("stations.csv")
drivers = pd.read_csv("drivers.csv")
interactions = pd.read_csv("interactions.csv")

# ---------- Helpers ----------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2-lat1)
    dlambda = radians(lon2-lon1)
    a = sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dlambda/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

def learned_preference(driver_id, station_id):
    x = interactions[
        (interactions.driver_id == driver_id) &
        (interactions.station_id == station_id)
    ]["usage_count"]
    # Prototype "GNN-like" learned score: normalized historical interaction signal.
    # This intentionally simulates the paper's learned preference stage rather than
    # claiming to reproduce the paper's trained GNN.
    if len(x) == 0:
        return 0.0
    return min(float(x.iloc[0]) / 5.0, 1.0)

def rank_stations(driver_id, context, charging_pref, top_n):
    d = drivers[drivers.driver_id == driver_id].iloc[0]
    # Context can alter the preferred cluster in this student prototype.
    preferred_cluster = d.preferred_cluster

    # A representative driver position is the centroid of the preferred cluster.
    cluster_st = stations[stations.cluster == preferred_cluster]
    center_lat = cluster_st.latitude.mean()
    center_lon = cluster_st.longitude.mean()

    rows = []
    for _, s in stations.iterrows():
        distance = haversine_km(center_lat, center_lon, s.latitude, s.longitude)
        pref = learned_preference(driver_id, s.station_id)
        cluster_match = 1.0 if s.cluster == preferred_cluster else 0.0
        type_match = 1.0 if s.charging_type == charging_pref else 0.0
        # Distance score decays smoothly; all components are shown for explainability.
        distance_score = np.exp(-distance / 2.0)

        # Context-aware refinement score.
        final_score = (
            0.50 * pref +
            0.25 * cluster_match +
            0.15 * type_match +
            0.10 * distance_score
        )

        reason = []
        if type_match: reason.append("same charging type")
        if cluster_match: reason.append("same/nearby cluster")
        if pref >= 0.6: reason.append("high learned preference")
        elif pref > 0: reason.append("previously used")
        if distance <= 2: reason.append("shorter distance")
        if not reason: reason.append("candidate from graph ranking")

        rows.append([
            s.station_id, s.station_name, s.provider, s.charging_type, s.cluster,
            s.latitude, s.longitude, distance, pref, cluster_match,
            type_match, distance_score, final_score, " • ".join(reason)
        ])

    out = pd.DataFrame(rows, columns=[
        "station_id","station_name","provider","charging_type","cluster",
        "latitude","longitude","distance_km","learned_preference",
        "cluster_match","type_match","distance_score","final_score","reason"
    ])
    return out.sort_values("final_score", ascending=False).head(top_n), out, (center_lat, center_lon)

# ---------- Sidebar ----------
st.sidebar.header("🔧 Prototype Input")
driver_id = st.sidebar.selectbox("Driver", drivers.driver_id.tolist(), index=3)
driver_row = drivers[drivers.driver_id == driver_id].iloc[0]

context = st.sidebar.selectbox(
    "Current context",
    ["Home", "Workplace", "Shopping"],
    index=["Home","Workplace","Shopping"].index(driver_row.context)
)
charging_pref = st.sidebar.selectbox(
    "Charging preference",
    ["FAST", "SLOW"],
    index=["FAST","SLOW"].index(driver_row.charging_preference)
)
top_n = st.sidebar.slider("Top-N recommendations", 3, 10, 5)

st.sidebar.info(
    "This is a student prototype inspired by the PPT. "
    "The learned-preference stage is simulated from historical usage; "
    "it is not a reproduction of the paper's trained GNN."
)

# ---------- Recommendation ----------
rec, all_scores, center = rank_stations(driver_id, context, charging_pref, top_n)

st.subheader("1️⃣ Input")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Driver", driver_id)
c2.metric("Context", context)
c3.metric("Charging", charging_pref)
c4.metric("Top-N", top_n)

st.subheader("2️⃣ Recommended Stations")
display = rec[[
    "station_id","station_name","provider","charging_type","cluster",
    "distance_km","learned_preference","final_score","reason"
]].copy()
display["distance_km"] = display["distance_km"].round(2)
display["learned_preference"] = display["learned_preference"].round(2)
display["final_score"] = display["final_score"].round(3)
display.index = range(1, len(display)+1)
st.dataframe(display, use_container_width=True)

st.subheader("3️⃣ Why was the station recommended?")
for i, (_, r) in enumerate(rec.iterrows(), start=1):
    with st.expander(f"#{i}  {r.station_id} — {r.station_name}"):
        st.write(f"**Final score:** {r.final_score:.3f}")
        st.write(f"**Distance:** {r.distance_km:.2f} km")
        st.write(f"**Learned preference:** {r.learned_preference:.2f}")
        st.write(f"**Cluster match:** {'Yes' if r.cluster_match else 'No'}")
        st.write(f"**Charging-type match:** {'Yes' if r.type_match else 'No'}")
        st.write(f"**Reason:** {r.reason}")

# ---------- Map-like plot ----------
st.subheader("4️⃣ Spatial Cluster View")
plot_df = stations.copy()
plot_df["selected"] = plot_df.station_id.isin(rec.station_id)
plot_df["size"] = np.where(plot_df.selected, 18, 8)
fig = px.scatter_map(
    plot_df,
    lat="latitude", lon="longitude",
    color="cluster",
    size="size",
    hover_name="station_name",
    hover_data=["station_id","charging_type","provider","cluster","selected"],
    zoom=10,
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Graph view ----------
st.subheader("5️⃣ Driver–Station Interaction Graph")
edge_df = interactions[interactions.driver_id == driver_id].copy()
if edge_df.empty:
    st.warning("No historical interactions for this driver.")
else:
    edge_df = edge_df.merge(stations[["station_id","station_name"]], on="station_id", how="left")
    st.write("Historical interactions used as the prototype's graph signal:")
    st.dataframe(edge_df[["driver_id","station_id","station_name","usage_count"]], use_container_width=True)

# ---------- Pipeline ----------
st.subheader("6️⃣ Mathematical / Algorithmic Pipeline")
st.markdown(
    "**Interaction Graph G** → **Learned Preference Score** → **Top-N Candidates** → "
    "**Spatial Cluster Filter** → **Charging-Type Refinement** → **Final Ranking**"
)

st.success(
    "Presentation message: the system does not simply choose the nearest charger; "
    "it combines learned driver–station preference with location and charging context."
)

st.caption("Prototype data are synthetic and intended for demonstration.")
