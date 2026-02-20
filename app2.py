import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
import pickle
import sklearn

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    with open("model/clf.pkl", 'rb') as f:
        rf = pickle.load(f)
    return rf

model = load_model()

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    path = "data/preprocessed_taxi"

    if os.path.exists(path):
        df = pd.read_parquet(path)

        # Ensure required columns exist
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df["pickup_datetime"]).dt.date
        if "hour" not in df.columns:
            df["hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour

    else:
        df = pd.read_parquet("hf://datasets/TaherMAfini/taxi_dataset/data/test-00000-of-00001.parquet")

        # Ensure required columns exist
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df["pickup_datetime"]).dt.date
        if "hour" not in df.columns:
            df["hour"] = pd.to_datetime(df["pickup_datetime"]).dt.hour

        df["zone_id"] = np.random.choice(
            ["zone_A", "zone_B", "zone_C", "zone_D"],
            size=len(df)
        )

        os.makedirs("data", exist_ok=True)
        df.to_parquet(path)

    return df


# ==========================================================
# METRIC CALCULATIONS
# ==========================================================
def compute_zone_metrics(df):
    agg = (
        df.groupby(["zone_id", "hour"])
        .size()
        .reset_index(name="trip_count")
    )

    zone_stats = (
        agg.groupby("zone_id")["trip_count"]
        .agg(["mean", "std", "var"])
        .reset_index()
        .rename(columns={
            "mean": "avg_trips",
            "std": "std_trips",
            "var": "variance"
        })
    )

    # Hotspot persistence
    threshold = agg["trip_count"].quantile(0.9)
    agg["is_peak"] = agg["trip_count"] >= threshold

    persistence = (
        agg.groupby("zone_id")["is_peak"]
        .mean()
        .reset_index(name="hotspot_persistence")
    )

    zone_stats = zone_stats.merge(persistence, on="zone_id")

    return agg, zone_stats


# ==========================================================
# APP CONFIG
# ==========================================================
st.set_page_config(
    page_title="NYC Taxi Hotspot Dashboard",
    layout="wide"
)

st.title("🚕 NYC Taxi Demand Hotspots")

df = load_data()
agg, zone_stats = compute_zone_metrics(df)

tab1, tab2 = st.tabs(["🔥 Hotspot Prediction", "📊 Operational Insights"])

# ==========================================================
# TAB 1 — HOTSPOT PREDICTION
# ==========================================================
def create_input(selected_long, selected_lat, selected_date, selected_hour):
    feature_input = pd.DataFrame([{
        "pickup_longitude": selected_long,
        "pickup_latitude": selected_lat,
        "year": selected_date.year,
        "month_sin": np.sin(2 * np.pi * selected_date.month / 12),
        "month_cos": np.cos(2 * np.pi * selected_date.month / 12),
        "day_sin": np.sin(2 * np.pi * selected_date.day / 31),
        "day_cos": np.cos(2 * np.pi * selected_date.day / 31),
        "weekday_sin": np.sin(2 * np.pi * selected_date.weekday() / 7),
        "weekday_cos": np.cos(2 * np.pi * selected_date.weekday() / 7),
        "dayofyear_sin": np.sin(2 * np.pi * selected_date.timetuple().tm_yday / 366),
        "dayofyear_cos": np.cos(2 * np.pi * selected_date.timetuple().tm_yday / 366),
        "hour_sin": np.sin(2 * np.pi * selected_hour / 24),
        "hour_cos": np.cos(2 * np.pi * selected_hour / 24)
    }])
    return feature_input
with tab1:
    st.subheader("Hotspot Quality Prediction")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_date = st.date_input(
            "Select Date",
            value=df["date"].min()
        )

    with col2:
        selected_hour = st.slider(
            "Select Hour",
            0, 23, 12
        )

    with col3:
        selected_lat = st.number_input(
            "Select Latitude",
            value=40.75,
            format="%.5f"
        )

    with col4:
        selected_long = st.number_input(
            "Select Longitude",
            value=-73.98,
            format="%.5f"
        )

    
    feature_input = create_input(selected_long, selected_lat, selected_date, selected_hour)
    # Hotspot score
    hotspot_score = model.predict_proba(feature_input)[0][1]

    st.metric(
        label="🔥 Hotspot Probability",
        value=round(hotspot_score, 2)
    )

    # Heatmap on real NYC map
    view_state = pdk.ViewState(
        latitude=selected_lat,
        longitude=selected_long,
        zoom=11,
        pitch=45,
    )

    layer = pdk.Layer(
        "HeatmapLayer",
        data=feature_input,
        get_position=["pickup_longitude", "pickup_latitude"],
        intensity=hotspot_score,
        radiusPixels=80,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="light"
    )

    st.pydeck_chart(deck)


# ==========================================================
# TAB 2 — INTERACTIVE DATA ANALYSIS
# ==========================================================
with tab2:
    st.subheader("Interactive Data Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        date_range = st.date_input(
            "Select Date Range",
            [df["date"].min(), df["date"].max()]
        )

    with col2:
        hour_range = st.slider(
            "Select Hour Range",
            0, 23,
            (0, 23)
        )

    with col3:
        selected_zones = st.multiselect(
            "Select Zones",
            options=sorted(df["zone_id"].unique()),
            default=sorted(df["zone_id"].unique())
        )

    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = df["date"].min()
        end_date = df["date"].max()

    filtered_df = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date) &
        (df["hour"] >= hour_range[0]) &
        (df["hour"] <= hour_range[1]) &
        (df["zone_id"].isin(selected_zones))
    ]

    st.markdown("### Aggregated Results")

    agg_option = st.selectbox(
        "Aggregate by:",
        [
            "Date",
            "Hour",
            "Zone",
            "Date and Hour",
            "Zone and Hour"
        ]
    )

    if filtered_df.empty:
        st.warning("No data for selected filters.")
    else:
        if agg_option == "Date":
            result = (
                filtered_df.groupby("date")
                .size()
                .reset_index(name="trip_count")
            )
            st.line_chart(result.set_index("date"))

        elif agg_option == "Hour":
            result = (
                filtered_df.groupby("hour")
                .size()
                .reset_index(name="trip_count")
            )
            st.bar_chart(result.set_index("hour"))

        elif agg_option == "Zone":
            result = (
                filtered_df.groupby("zone_id")
                .size()
                .reset_index(name="trip_count")
                .sort_values("trip_count", ascending=False)
            )
            st.bar_chart(result.set_index("zone_id"))

        elif agg_option == "Date and Hour":
            result = (
                filtered_df.groupby(["date", "hour"])
                .size()
                .reset_index(name="trip_count")
            )

        elif agg_option == "Zone and Hour":
            result = (
                filtered_df.groupby(["zone_id", "hour"])
                .size()
                .reset_index(name="trip_count")
            )

        st.markdown("### Data Table")
        st.dataframe(result)

    # Operational insights
    st.markdown("## Operational Insights")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Top High-Variance Zones")
        top_var = zone_stats.sort_values(
            "variance", ascending=False
        ).head(10)
        st.dataframe(top_var)

    with colB:
        st.subheader("Top Persistent Hotspots")
        top_persist = zone_stats.sort_values(
            "hotspot_persistence", ascending=False
        ).head(10)
        st.dataframe(top_persist)
