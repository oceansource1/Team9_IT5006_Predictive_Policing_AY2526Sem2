import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
from pathlib import Path


st.set_page_config(
    page_title="Predictive Policing Demo",
    layout="wide"
)

st.title("Predictive Policing – Weekly Theft Risk Demo")
st.caption("IT5006 Phase 3 Deployment Demo")


BASE_DIR = Path(__file__).resolve().parent

TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "train_table_test_2025.parquet"
GRID_META_PATH = BASE_DIR / "data" / "processed" / "grid_metadata.parquet"

LOGREG_PATH = BASE_DIR / "models" / "logreg_grid.pkl"
RF_PATH = BASE_DIR / "models" / "rf_grid.pkl"
XGB_PATH = BASE_DIR / "models" / "xgb_grid_pipeline.pkl"


FEATURES = [
    "cell_id",
    "year",
    "month",
    "weekofyear",
    "dayofweek",
    "lag_1",
    "lag_2",
    "lag_4",
    "roll_mean_4",
    "roll_sum_4",
    "roll_mean_8",
    "roll_sum_8",
    "roll_std_8",
]

META_COLS = ["cell_id", "time_bin", "y"]

MODEL_PATHS = {
    "Logistic Regression": LOGREG_PATH,
    "Random Forest": RF_PATH,
    "XGBoost": XGB_PATH,
}


@st.cache_data
def load_test_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test data file not found: {path}")

    df = pd.read_parquet(path).copy()

    required_cols = META_COLS + FEATURES
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in test data: {missing}")

    df["time_bin"] = pd.to_datetime(df["time_bin"], errors="coerce")
    df = df.dropna(subset=["time_bin"]).copy()
    df["cell_id"] = df["cell_id"].astype(str)

    return df.sort_values(["cell_id", "time_bin"]).reset_index(drop=True)


@st.cache_resource
def load_models():
    models = {}
    missing_files = []

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            missing_files.append(str(model_path))
        else:
            models[model_name] = joblib.load(model_path)

    if missing_files:
        raise FileNotFoundError(
            "The following model files are missing:\n" + "\n".join(missing_files)
        )

    return models


@st.cache_data
def load_grid_meta(path: Path):
    if not path.exists():
        return None

    grid_df = pd.read_parquet(path).copy()

    possible_required_sets = [
        ["cell_id", "center_lat", "center_lon"],
        ["cell_id", "lat", "lon"],
        ["cell_id", "lat_min", "lat_max", "lon_min", "lon_max"],
        ["cell_id", "cell_lat_min", "cell_lat_max", "cell_lon_min", "cell_lon_max"],
    ]

    ok = any(all(col in grid_df.columns for col in cols) for cols in possible_required_sets)
    if not ok:
        return None

    grid_df["cell_id"] = grid_df["cell_id"].astype(str)
    return grid_df


def get_prediction_output(model, X: pd.DataFrame, threshold: float = 0.5):
    prob = float(model.predict_proba(X)[0, 1])
    pred = 1 if prob >= threshold else 0
    return prob, pred


def predict_for_row(model, row_df: pd.DataFrame, threshold: float = 0.5):
    X = row_df[FEATURES].copy()
    return get_prediction_output(model, X, threshold=threshold)


def risk_label(prob: float) -> str:
    if prob >= 0.80:
        return "Very High Risk"
    elif prob >= 0.60:
        return "High Risk"
    elif prob >= 0.35:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_available_weeks_for_cell(df: pd.DataFrame, cell_id: str):
    sub = df[df["cell_id"].astype(str) == str(cell_id)].copy()
    return sorted(sub["time_bin"].dropna().unique())


def get_row_by_cell_and_week(df: pd.DataFrame, cell_id: str, target_week: pd.Timestamp):
    sub = df[
        (df["cell_id"].astype(str) == str(cell_id)) &
        (df["time_bin"] == pd.to_datetime(target_week))
    ].copy()

    if sub.empty:
        return None
    return sub.iloc[[0]].copy()


def get_future_rows(df: pd.DataFrame, cell_id: str, start_week: pd.Timestamp, horizon: int):
    sub = df[
        (df["cell_id"].astype(str) == str(cell_id)) &
        (df["time_bin"] >= pd.to_datetime(start_week))
    ].copy()

    sub = sub.sort_values("time_bin").head(horizon).copy()
    return sub


def run_forecast_table(model, df: pd.DataFrame, cell_id: str, start_week: pd.Timestamp, horizon: int, threshold: float = 0.5):
    future_rows = get_future_rows(df, cell_id, start_week, horizon)
    if future_rows.empty:
        return pd.DataFrame()

    probs = []
    preds = []
    risk_levels = []

    for _, r in future_rows.iterrows():
        row_df = pd.DataFrame([r])
        prob, pred = predict_for_row(model, row_df, threshold=threshold)
        probs.append(prob)
        preds.append(pred)
        risk_levels.append(risk_label(prob))

    out = future_rows[["cell_id", "time_bin", "y"]].copy()
    out["predicted_probability"] = probs
    out["predicted_class"] = preds
    out["risk_level"] = risk_levels

    return out.reset_index(drop=True)


def find_nearest_cell_from_grid(lat, lon, grid_df):
    if grid_df is None or grid_df.empty:
        return None

    cols = set(grid_df.columns)

    if {"cell_id", "center_lat", "center_lon"}.issubset(cols):
        tmp = grid_df.copy()
        tmp["dist2"] = (tmp["center_lat"] - lat) ** 2 + (tmp["center_lon"] - lon) ** 2
        return str(tmp.sort_values("dist2").iloc[0]["cell_id"])

    if {"cell_id", "lat", "lon"}.issubset(cols):
        tmp = grid_df.copy()
        tmp["dist2"] = (tmp["lat"] - lat) ** 2 + (tmp["lon"] - lon) ** 2
        return str(tmp.sort_values("dist2").iloc[0]["cell_id"])

    if {"cell_id", "lat_min", "lat_max", "lon_min", "lon_max"}.issubset(cols):
        matched = grid_df[
            (grid_df["lat_min"] <= lat) &
            (grid_df["lat_max"] >= lat) &
            (grid_df["lon_min"] <= lon) &
            (grid_df["lon_max"] >= lon)
        ]
        if not matched.empty:
            return str(matched.iloc[0]["cell_id"])

    if {"cell_id", "cell_lat_min", "cell_lat_max", "cell_lon_min", "cell_lon_max"}.issubset(cols):
        matched = grid_df[
            (grid_df["cell_lat_min"] <= lat) &
            (grid_df["cell_lat_max"] >= lat) &
            (grid_df["cell_lon_min"] <= lon) &
            (grid_df["cell_lon_max"] >= lon)
        ]
        if not matched.empty:
            return str(matched.iloc[0]["cell_id"])

        tmp = grid_df.copy()
        tmp["center_lat"] = (tmp["cell_lat_min"] + tmp["cell_lat_max"]) / 2
        tmp["center_lon"] = (tmp["cell_lon_min"] + tmp["cell_lon_max"]) / 2
        tmp["dist2"] = (tmp["center_lat"] - lat) ** 2 + (tmp["center_lon"] - lon) ** 2
        return str(tmp.sort_values("dist2").iloc[0]["cell_id"])

    return None


def get_cell_polygon_from_grid(grid_df: pd.DataFrame, cell_id: str):
    sub = grid_df[grid_df["cell_id"].astype(str) == str(cell_id)].copy()
    if sub.empty:
        return None

    r = sub.iloc[0]
    cols = set(sub.columns)

    if {"cell_lat_min", "cell_lat_max", "cell_lon_min", "cell_lon_max"}.issubset(cols):
        lat_min = float(r["cell_lat_min"])
        lat_max = float(r["cell_lat_max"])
        lon_min = float(r["cell_lon_min"])
        lon_max = float(r["cell_lon_max"])
    elif {"lat_min", "lat_max", "lon_min", "lon_max"}.issubset(cols):
        lat_min = float(r["lat_min"])
        lat_max = float(r["lat_max"])
        lon_min = float(r["lon_min"])
        lon_max = float(r["lon_max"])
    elif {"center_lat", "center_lon"}.issubset(cols):
        center_lat = float(r["center_lat"])
        center_lon = float(r["center_lon"])
        delta = 0.0045
        lat_min, lat_max = center_lat - delta, center_lat + delta
        lon_min, lon_max = center_lon - delta, center_lon + delta
    else:
        return None

    polygon = [
        [lon_min, lat_min],
        [lon_min, lat_max],
        [lon_max, lat_max],
        [lon_max, lat_min],
    ]

    if "center_lat" in r.index and "center_lon" in r.index:
        center_lat = float(r["center_lat"])
        center_lon = float(r["center_lon"])
    else:
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2

    return {
        "cell_id": str(cell_id),
        "polygon": polygon,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


def get_neighbor_cells_from_grid(grid_df: pd.DataFrame, cell_id: str, radius: int = 1):
    if grid_df is None or grid_df.empty:
        return pd.DataFrame()

    if not {"cell_id", "grid_row", "grid_col"}.issubset(grid_df.columns):
        return pd.DataFrame()

    sub = grid_df[grid_df["cell_id"].astype(str) == str(cell_id)].copy()
    if sub.empty:
        return pd.DataFrame()

    row = int(sub.iloc[0]["grid_row"])
    col = int(sub.iloc[0]["grid_col"])

    neighbors = grid_df[
        (grid_df["grid_row"].between(row - radius, row + radius)) &
        (grid_df["grid_col"].between(col - radius, col + radius))
    ].copy()

    return neighbors


def infer_zoom_from_bounds(lat_min, lat_max, lon_min, lon_max):
    span = max(lat_max - lat_min, lon_max - lon_min)
    if span < 0.01:
        return 14.5
    if span < 0.03:
        return 13.5
    if span < 0.08:
        return 12.5
    return 11.5


def render_selected_cell_map(grid_df: pd.DataFrame, cell_id: str, input_lat=None, input_lon=None):
    if grid_df is None:
        st.info("Map is unavailable because grid metadata is not loaded.")
        return

    cell_info = get_cell_polygon_from_grid(grid_df, cell_id)
    if cell_info is None:
        st.info("Map is unavailable because the selected cell geometry could not be found.")
        return

    neighbor_polygons = []
    neighbors = get_neighbor_cells_from_grid(grid_df, cell_id, radius=1)

    if not neighbors.empty:
        for _, r in neighbors.iterrows():
            neighbor_cell_id = str(r["cell_id"])
            info = get_cell_polygon_from_grid(grid_df, neighbor_cell_id)
            if info is not None:
                neighbor_polygons.append({
                    "cell_id": neighbor_cell_id,
                    "polygon": info["polygon"],
                    "is_selected": 1 if neighbor_cell_id == str(cell_id) else 0
                })
    else:
        neighbor_polygons.append({
            "cell_id": cell_info["cell_id"],
            "polygon": cell_info["polygon"],
            "is_selected": 1
        })

    polygon_df = pd.DataFrame(neighbor_polygons)

    center_df = pd.DataFrame([{
        "label": "Selected Cell Center",
        "lat": cell_info["center_lat"],
        "lon": cell_info["center_lon"]
    }])

    layers = []

    non_selected = polygon_df[polygon_df["is_selected"] == 0]
    if not non_selected.empty:
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=non_selected,
                get_polygon="polygon",
                get_fill_color=[180, 180, 180, 40],
                get_line_color=[120, 120, 120, 140],
                line_width_min_pixels=1,
                pickable=True,
                stroked=True,
                filled=True,
            )
        )

    selected = polygon_df[polygon_df["is_selected"] == 1]
    layers.append(
        pdk.Layer(
            "PolygonLayer",
            data=selected,
            get_polygon="polygon",
            get_fill_color=[255, 99, 71, 110],
            get_line_color=[220, 20, 60, 255],
            line_width_min_pixels=3,
            pickable=True,
            stroked=True,
            filled=True,
        )
    )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=center_df,
            get_position="[lon, lat]",
            get_radius=90,
            get_fill_color=[220, 20, 60, 255],
            pickable=True,
        )
    )

    if input_lat is not None and input_lon is not None:
        input_df = pd.DataFrame([{
            "label": "Input Location",
            "lat": float(input_lat),
            "lon": float(input_lon)
        }])

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=input_df,
                get_position="[lon, lat]",
                get_radius=90,
                get_fill_color=[30, 144, 255, 255],
                pickable=True,
            )
        )

    zoom = infer_zoom_from_bounds(
        cell_info["lat_min"],
        cell_info["lat_max"],
        cell_info["lon_min"],
        cell_info["lon_max"]
    )

    view_state = pdk.ViewState(
        latitude=cell_info["center_lat"],
        longitude=cell_info["center_lon"],
        zoom=zoom,
        pitch=0
    )

    tooltip = {
        "html": "<b>Cell:</b> {cell_id}",
        "style": {"color": "white"}
    }

    deck = pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=view_state,
        layers=layers,
        tooltip=tooltip,
    )

    st.pydeck_chart(deck, use_container_width=True)

    st.caption(
        f"Selected cell: {cell_info['cell_id']} | "
        f"Lat range: [{cell_info['lat_min']:.4f}, {cell_info['lat_max']:.4f}] | "
        f"Lon range: [{cell_info['lon_min']:.4f}, {cell_info['lon_max']:.4f}]"
    )



try:
    df = load_test_data(TEST_DATA_PATH)
    models = load_models()
    grid_meta = load_grid_meta(GRID_META_PATH)
except Exception as e:
    st.error("App failed to load.")
    st.exception(e)
    st.stop()


st.sidebar.header("Global Controls")

selected_model_name = st.sidebar.selectbox(
    "Select model",
    list(models.keys()),
    index=2
)

threshold = st.sidebar.slider(
    "Risk classification threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

forecast_horizon = st.sidebar.slider(
    "Number of future weeks to display",
    min_value=1,
    max_value=8,
    value=4,
    step=1
)

show_feature_table = st.sidebar.checkbox("Show feature snapshot", value=True)
show_history_table = st.sidebar.checkbox("Show recent history", value=True)
show_dataset_preview = st.sidebar.checkbox("Show dataset preview", value=False)

model = models[selected_model_name]


st.subheader("Project Overview")
st.write(

)


st.subheader("Input Panel")

input_mode = st.radio(
    "Choose input mode",
    ["By Area / Cell", "By Latitude / Longitude"],
    horizontal=True
)

selected_cell = None
input_lat = None
input_lon = None

if input_mode == "By Area / Cell":
    cell_options = sorted(df["cell_id"].astype(str).unique().tolist())
    selected_cell = st.selectbox("Select Area / Cell ID", cell_options)

else:
    if grid_meta is None:
        st.info(
            "Coordinate-based lookup is not enabled because grid metadata is not available. "
            "Please switch to 'By Area / Cell'."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            input_lat = st.number_input(
                "Input Latitude",
                value=41.8781,
                format="%.6f"
            )
        with c2:
            input_lon = st.number_input(
                "Input Longitude",
                value=-87.6298,
                format="%.6f"
            )

        selected_cell = find_nearest_cell_from_grid(input_lat, input_lon, grid_meta)

        if selected_cell is not None:
            st.success(f"Mapped to nearest cell: {selected_cell}")
        else:
            st.error("Unable to map the input coordinates to a valid cell.")


if selected_cell is None:
    st.stop()

available_weeks = get_available_weeks_for_cell(df, selected_cell)

if not available_weeks:
    st.warning("No weekly records available for the selected cell.")
    st.stop()

selected_week = st.selectbox(
    "Select Week",
    available_weeks,
    format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d")
)

row = get_row_by_cell_and_week(df, selected_cell, selected_week)

if row is None:
    st.warning("No matched record found for the selected cell and week.")
    st.stop()


st.subheader("Current Week Prediction")

current_prob, current_pred = predict_for_row(model, row, threshold=threshold)
actual_y = int(row["y"].iloc[0])

info1, info2, info3, info4 = st.columns(4)
with info1:
    st.info(f"**Cell ID:** {selected_cell}")
with info2:
    st.info(f"**Selected Week:** {pd.to_datetime(selected_week).strftime('%Y-%m-%d')}")
with info3:
    st.info(f"**Model:** {selected_model_name}")
with info4:
    st.info(f"**Actual Label:** {actual_y}")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Risk Probability", f"{current_prob:.4f}")
with m2:
    st.metric("Predicted Class", str(current_pred))
with m3:
    st.metric("Risk Level", risk_label(current_prob))

summary_df = pd.DataFrame(
    {
        "Model": [selected_model_name],
        "Cell_ID": [selected_cell],
        "Week": [pd.to_datetime(selected_week)],
        "Probability_of_y=1": [current_prob],
        "Predicted_Class": [current_pred],
        "Actual_y": [actual_y],
        "Risk_Level": [risk_label(current_prob)],
    }
)
st.dataframe(summary_df, use_container_width=True)


st.subheader("Selected Cell Map")

if input_mode == "By Latitude / Longitude" and grid_meta is not None:
    render_selected_cell_map(
        grid_df=grid_meta,
        cell_id=selected_cell,
        input_lat=input_lat,
        input_lon=input_lon
    )
else:
    render_selected_cell_map(
        grid_df=grid_meta,
        cell_id=selected_cell
    )


st.subheader("Future Risk Outlook")

forecast_df = run_forecast_table(
    model=model,
    df=df,
    cell_id=selected_cell,
    start_week=selected_week,
    horizon=forecast_horizon,
    threshold=threshold
)

if forecast_df.empty:
    st.warning("No future weekly records available from the selected week.")
else:
    top1, top2, top3 = st.columns(3)
    with top1:
        st.info(f"**Forecast Start Week:** {pd.to_datetime(selected_week).strftime('%Y-%m-%d')}")
    with top2:
        st.info(f"**Forecast Horizon:** {len(forecast_df)} week(s)")
    with top3:
        st.info(f"**Selected Cell:** {selected_cell}")

    st.markdown("#### Future Weekly Risk Table")
    st.dataframe(forecast_df, use_container_width=True)

    st.markdown("#### Future Risk Trend")
    chart_df = forecast_df.copy().set_index("time_bin")[["predicted_probability"]]
    st.line_chart(chart_df)


if show_feature_table:
    st.subheader("Feature Snapshot")
    display_cols = list(dict.fromkeys(META_COLS + FEATURES))
    st.dataframe(row[display_cols], use_container_width=True)


if show_history_table:
    st.subheader("Recent Historical Labels")
    history_preview = (
        df[
            (df["cell_id"].astype(str) == str(selected_cell)) &
            (df["time_bin"] < pd.to_datetime(selected_week))
        ][["cell_id", "time_bin", "y"]]
        .sort_values("time_bin", ascending=False)
        .head(12)
        .sort_values("time_bin")
    )
    st.dataframe(history_preview, use_container_width=True)


if show_dataset_preview:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)


st.markdown("---")
st.caption(
    "This application is a proof-of-concept deployment for the IT5006 project. "
    "It provides a unified workflow from area selection to current-week prediction, "
    "spatial map display, and future weekly risk outlook."
)