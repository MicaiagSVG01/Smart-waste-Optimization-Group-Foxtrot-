# ==========================================
# SMART WASTE OPTIMIZATION SYSTEM
# STREAMLIT APPLICATION
# ==========================================

from urllib.parse import quote_plus

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Waste Optimization",
    page_icon="♻️",
    layout="wide"
)


# ==========================================
# CONSTANTS / DATA
# ==========================================

LANDFILL_LOCATIONS = {
    "Olusosun": (6.6018, 3.3579),
    "Abule-Egba": (6.6480, 3.2745),
    "Somolu": (6.5380, 3.3840),
    "Badagry": (6.4167, 2.8833),
    "Epe": (6.5841, 3.9836),
    "Ikorodu": (6.6194, 3.5105)
}

DENSITY = {
    "Olusosun": 15000,
    "Abule-Egba": 12000,
    "Somolu": 13000,
    "Badagry": 8000,
    "Epe": 7000,
    "Ikorodu": 11000
}

LANDFILL_ENCODING = {
    "Abule-Egba": 0,
    "Badagry": 1,
    "Epe": 2,
    "Ikorodu": 3,
    "Olusosun": 4,
    "Somolu": 5
}

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

FESTIVE_MONTHS = [
    "March",
    "May",
    "December"
]


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("random_forest.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "The model file 'random_forest.pkl' was not found. "
        "Please make sure it is in the same GitHub folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()


# ==========================================
# SESSION STATE
# ==========================================

if "latest_prediction" not in st.session_state:
    st.session_state["latest_prediction"] = None

if "target_landfill" not in st.session_state:
    st.session_state["target_landfill"] = None

if "last_google_maps_url" not in st.session_state:
    st.session_state["last_google_maps_url"] = None

if "last_current_location" not in st.session_state:
    st.session_state["last_current_location"] = None


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_demand_level(predicted_tonnage):
    """
    Converts predicted waste tonnage into a simple demand category.
    """

    if predicted_tonnage < 10000:
        return "Low Demand"
    elif predicted_tonnage < 25000:
        return "Moderate Demand"
    else:
        return "High Demand"


def create_google_maps_url(current_location, target_landfill):
    """
    Creates a Google Maps direction URL.
    If current_location is empty, Google Maps can use the user's device location.
    """

    target_lat, target_lon = LANDFILL_LOCATIONS[target_landfill]
    destination = f"{target_lat},{target_lon}"

    if current_location.strip():
        origin = quote_plus(current_location.strip())

        google_maps_url = (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={origin}"
            f"&destination={destination}"
            "&travelmode=driving"
        )

    else:
        google_maps_url = (
            "https://www.google.com/maps/dir/?api=1"
            f"&destination={destination}"
            "&travelmode=driving"
        )

    return google_maps_url


def display_target_map(target_landfill):
    """
    Displays the selected landfill on an interactive map.
    """

    lat, lon = LANDFILL_LOCATIONS[target_landfill]

    target_map = folium.Map(
        location=[lat, lon],
        zoom_start=12
    )

    folium.Marker(
        location=[lat, lon],
        popup=f"{target_landfill} - Target Landfill",
        tooltip=f"{target_landfill} - Target Landfill"
    ).add_to(target_map)

    st_folium(
        target_map,
        width=1100,
        height=500
    )


# ==========================================
# APP HEADER
# ==========================================

st.title("♻️ Smart Waste Optimization System")

st.markdown(
    """
    This system predicts waste tonnage and helps users get directions to selected Lagos landfill sites.
    """
)


# ==========================================
# SIDEBAR
# ==========================================

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Waste Prediction",
        "Route Optimization",
        "Analytics"
    ]
)

st.sidebar.markdown("---")

if st.session_state["latest_prediction"] is not None:
    latest = st.session_state["latest_prediction"]

    st.sidebar.success("Latest Prediction Available")
    st.sidebar.write(f"**Target Landfill:** {latest['landfill']}")
    st.sidebar.write(f"**Month:** {latest['month']}")
    st.sidebar.write(f"**Trips:** {latest['trips']:,}")
    st.sidebar.write(f"**Tonnage:** {latest['prediction']:,.2f} tonnes")
    st.sidebar.write(f"**Demand Level:** {latest['demand_level']}")
else:
    st.sidebar.info("Run a waste prediction first.")


# ==========================================
# WASTE PREDICTION PAGE
# ==========================================

if page == "Waste Prediction":

    st.header("Waste Volume Prediction")

    st.write(
        """
        This page predicts the expected waste tonnage for a selected landfill.
        The selected landfill becomes the target destination on the Route Optimization page.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        landfill = st.selectbox(
            "Target Landfill Site",
            list(LANDFILL_LOCATIONS.keys())
        )

        year = st.number_input(
            "Year",
            min_value=2025,
            max_value=2035,
            value=2025
        )

    with col2:
        month = st.selectbox(
            "Month",
            list(MONTH_MAP.keys())
        )

        trips = st.number_input(
            "Expected Trips",
            min_value=1,
            value=1000
        )

    month_number = MONTH_MAP[month]
    quarter = ((month_number - 1) // 3) + 1
    festive_period = 1 if month in FESTIVE_MONTHS else 0
    population_density = DENSITY[landfill]
    landfill_encoded = LANDFILL_ENCODING[landfill]

    st.markdown("### Input Features Used by the Model")

    feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)

    feature_col1.metric("Month Number", month_number)
    feature_col2.metric("Quarter", quarter)
    feature_col3.metric("Festive Period", "Yes" if festive_period == 1 else "No")
    feature_col4.metric("Population Density", f"{population_density:,}")

    if st.button("Predict Waste Tonnage", use_container_width=True):

        input_data = pd.DataFrame({
            "Year": [year],
            "Month_Number": [month_number],
            "Quarter": [quarter],
            "Festive_Period": [festive_period],
            "Population_Density": [population_density],
            "Landfill_Encoded": [landfill_encoded],
            "Trips": [trips]
        })

        prediction = model.predict(input_data)[0]
        demand_level = get_demand_level(prediction)

        st.session_state["latest_prediction"] = {
            "landfill": landfill,
            "year": year,
            "month": month,
            "month_number": month_number,
            "quarter": quarter,
            "festive_period": festive_period,
            "population_density": population_density,
            "landfill_encoded": landfill_encoded,
            "trips": trips,
            "prediction": prediction,
            "demand_level": demand_level
        }

        st.session_state["target_landfill"] = landfill
        st.session_state["last_google_maps_url"] = None
        st.session_state["last_current_location"] = None

        st.success(f"Predicted Waste Tonnage: {prediction:,.2f} tonnes")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric("Target Landfill", landfill)
        result_col2.metric("Predicted Tonnage", f"{prediction:,.2f}")
        result_col3.metric("Demand Level", demand_level)

        st.info(
            """
            Prediction saved. Go to the Route Optimization page to get directions to this target landfill.
            """
        )


# ==========================================
# ROUTE OPTIMIZATION PAGE
# ==========================================

elif page == "Route Optimization":

    st.header("🚛 Route Optimization and Directions")

    st.write(
        """
        This page explains why the selected landfill is the destination and provides Google Maps directions to get there.
        """
    )

    if st.session_state["latest_prediction"] is None:
        st.warning(
            """
            No prediction has been generated yet. Please go to the Waste Prediction page first
            and run a waste prediction.
            """
        )
        st.stop()

    latest = st.session_state["latest_prediction"]
    target_landfill = latest["landfill"]
    target_lat, target_lon = LANDFILL_LOCATIONS[target_landfill]

    st.markdown("### Why this landfill?")

    st.info(
        f"""
        You are going to **{target_landfill}** because this is the landfill selected during waste prediction.
        The system predicted **{latest['prediction']:,.2f} tonnes** of waste for this site and classified it as
        **{latest['demand_level']}**.
        """
    )

    st.markdown("### Prediction Used for Route Planning")

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    info_col1.metric("Target Landfill", target_landfill)
    info_col2.metric("Month", latest["month"])
    info_col3.metric("Expected Trips", f"{latest['trips']:,}")
    info_col4.metric("Predicted Tonnage", f"{latest['prediction']:,.2f}")

    st.write(f"**Demand Level:** {latest['demand_level']}")
    st.write(f"**Target Coordinates:** {target_lat}, {target_lon}")

    st.markdown("---")

    st.markdown("### Get Driving Directions")

    st.write(
        """
        Enter your current location below. The app will open Google Maps directions from that location
        to the selected target landfill. If you leave the box empty, Google Maps can use your current
        device location after you grant permission.
        """
    )

    current_location = st.text_input(
        "Enter your current location",
        placeholder="Example: Ikeja, Lagos or LAWMA office, Lagos"
    )

    google_maps_url = create_google_maps_url(
        current_location=current_location,
        target_landfill=target_landfill
    )

    st.session_state["last_google_maps_url"] = google_maps_url
    st.session_state["last_current_location"] = (
        current_location.strip() if current_location.strip() else "Device current location"
    )

    if current_location.strip():
        st.link_button(
            f"Open Google Maps Directions from {current_location} to {target_landfill}",
            google_maps_url,
            use_container_width=True
        )
    else:
        st.link_button(
            f"Use My Current Location in Google Maps to {target_landfill}",
            google_maps_url,
            use_container_width=True
        )

    st.markdown("### Target Landfill Map")

    st.write(
        """
        The map below shows the selected target landfill location. Full turn-by-turn driving directions
        are handled by Google Maps using the button above.
        """
    )

    display_target_map(target_landfill)


# ==========================================
# ANALYTICS PAGE
# ==========================================

elif page == "Analytics":

    st.header("Project Analytics")

    st.info(
        """
        This section summarizes the model performance, latest prediction, and route direction status.
        """
    )

    st.subheader("Model Performance Metrics")

    mae = 2211.95
    rmse = 6459.47
    r2 = 0.98

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric("MAE", f"{mae:,.2f}")
    metric_col2.metric("RMSE", f"{rmse:,.2f}")
    metric_col3.metric("R² Score", f"{r2:.2f}")

    st.markdown("---")

    st.subheader("Latest Prediction Summary")

    if st.session_state["latest_prediction"] is not None:
        latest = st.session_state["latest_prediction"]

        pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)

        pred_col1.metric("Target Landfill", latest["landfill"])
        pred_col2.metric("Month", latest["month"])
        pred_col3.metric("Trips", f"{latest['trips']:,}")
        pred_col4.metric("Predicted Tonnage", f"{latest['prediction']:,.2f}")

        st.write(f"**Demand Level:** {latest['demand_level']}")
        st.write(
            f"**Festive Period:** {'Yes' if latest['festive_period'] == 1 else 'No'}"
        )

    else:
        st.warning("No prediction has been generated yet.")

    st.markdown("---")

    st.subheader("Route Direction Summary")

    if st.session_state["latest_prediction"] is not None:
        target = st.session_state["latest_prediction"]["landfill"]

        st.write(f"**Target Destination:** {target}")

        if st.session_state["last_current_location"] is not None:
            st.write(f"**Starting Location:** {st.session_state['last_current_location']}")
        else:
            st.write("**Starting Location:** Not entered yet")

        if st.session_state["last_google_maps_url"] is not None:
            st.link_button(
                "Open Last Generated Google Maps Direction",
                st.session_state["last_google_maps_url"],
                use_container_width=True
            )
        else:
            st.info(
                """
                No direction link has been generated yet. Go to the Route Optimization page to generate directions.
                """
            )

    else:
        st.warning("No target landfill is available yet.")

    st.markdown("---")

    st.subheader("System Interpretation")

    st.write(
        """
        The system first predicts expected waste tonnage for a selected target landfill.
        The Route Optimization page then explains why that landfill is the destination and provides a
        Google Maps direction link from the user's current location to the selected landfill.
        This keeps the app simple, practical, and easier to interpret.
        """
    )