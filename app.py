# ==========================================
# SMART WASTE OPTIMIZATION SYSTEM
# STREAMLIT APPLICATION
# ==========================================

import math
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import joblib
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Waste Optimization",
    page_icon="♻️",
    layout="wide"
)


# ==========================================
# CONSTANTS AND DATA
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


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("random_forest.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error("The model file 'random_forest.pkl' was not found. Please upload it to the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

if "latest_prediction" not in st.session_state:
    st.session_state["latest_prediction"] = None

if "latest_route" not in st.session_state:
    st.session_state["latest_route"] = None

if "latest_route_distance" not in st.session_state:
    st.session_state["latest_route_distance"] = None

if "latest_route_mode" not in st.session_state:
    st.session_state["latest_route_mode"] = None


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_demand_level(predicted_tonnage):
    if predicted_tonnage < 10000:
        return "Low Demand"
    elif predicted_tonnage < 25000:
        return "Moderate Demand"
    else:
        return "High Demand"


def haversine_distance_km(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    radius = 6371

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c


def build_distance_matrix(locations):
    distance_matrix = []

    for loc1 in locations:
        row = []

        for loc2 in locations:
            distance_km = haversine_distance_km(
                LANDFILL_LOCATIONS[loc1],
                LANDFILL_LOCATIONS[loc2]
            )

            distance_meters = int(distance_km * 1000)
            row.append(distance_meters)

        distance_matrix.append(row)

    return distance_matrix


def calculate_route_distance(route):
    total_distance = 0

    for i in range(len(route) - 1):
        current_location = route[i]
        next_location = route[i + 1]

        total_distance += haversine_distance_km(
            LANDFILL_LOCATIONS[current_location],
            LANDFILL_LOCATIONS[next_location]
        )

    return total_distance


def select_demand_based_locations(start_location, predicted_tonnage):
    other_locations = [
        location for location in LANDFILL_LOCATIONS.keys()
        if location != start_location
    ]

    sorted_locations = sorted(
        other_locations,
        key=lambda loc: haversine_distance_km(
            LANDFILL_LOCATIONS[start_location],
            LANDFILL_LOCATIONS[loc]
        )
    )

    demand_level = get_demand_level(predicted_tonnage)

    if demand_level == "Low Demand":
        number_of_extra_sites = 3
    elif demand_level == "Moderate Demand":
        number_of_extra_sites = 4
    else:
        number_of_extra_sites = len(sorted_locations)

    selected_locations = [start_location] + sorted_locations[:number_of_extra_sites]

    return selected_locations


def solve_route(locations, depot_index=0):
    distance_matrix = build_distance_matrix(locations)

    data = {
        "distance_matrix": distance_matrix,
        "num_vehicles": 1,
        "depot": depot_index
    }

    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]),
        data["num_vehicles"],
        data["depot"]
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return None

    index = routing.Start(0)
    route = []

    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(locations[node_index])
        index = solution.Value(routing.NextVar(index))

    route.append(locations[manager.IndexToNode(index)])

    return route


def display_route_map(route):
    route_map = folium.Map(
        location=[6.55, 3.40],
        zoom_start=10
    )

    route_coordinates = []

    for index, location in enumerate(route):
        lat, lon = LANDFILL_LOCATIONS[location]

        route_coordinates.append([lat, lon])

        popup_text = f"{index + 1}. {location}"

        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            tooltip=popup_text
        ).add_to(route_map)

    folium.PolyLine(
        route_coordinates,
        weight=5,
        opacity=0.8
    ).add_to(route_map)

    st_folium(
        route_map,
        width=1100,
        height=600
    )


# ==========================================
# APP HEADER
# ==========================================

st.title("♻️ Smart Waste Optimization System")

st.markdown(
    """
    This system predicts waste tonnage and supports waste collection route optimization across Lagos landfill sites.
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

    st.sidebar.write(f"**Landfill:** {latest['landfill']}")
    st.sidebar.write(f"**Month:** {latest['month']}")
    st.sidebar.write(f"**Trips:** {latest['trips']}")
    st.sidebar.write(f"**Tonnage:** {latest['prediction']:,.2f} tonnes")
else:
    st.sidebar.info("Run a waste prediction first.")


# ==========================================
# WASTE PREDICTION PAGE
# ==========================================

if page == "Waste Prediction":

    st.header("Waste Volume Prediction")

    st.write(
        """
        Select a landfill site and operational details. The system will predict the expected waste tonnage
        and save the result for route optimization.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        landfill = st.selectbox(
            "Landfill Site",
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
    festive_period = 1 if month in ["December", "January"] else 0
    population_density = DENSITY[landfill]
    landfill_encoded = LANDFILL_ENCODING[landfill]

    st.markdown("### Engineered Input Features")

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

        st.success(f"Predicted Waste Tonnage: {prediction:,.2f} tonnes")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric("Selected Landfill", landfill)
        result_col2.metric("Predicted Tonnage", f"{prediction:,.2f}")
        result_col3.metric("Demand Level", demand_level)

        st.info(
            """
            Prediction saved. Go to the Route Optimization page to generate a route based on this prediction.
            """
        )


# ==========================================
# ROUTE OPTIMIZATION PAGE
# ==========================================

elif page == "Route Optimization":

    st.header("🚛 Route Optimization")

    st.write(
        """
        This module uses Google OR-Tools to generate an optimized waste collection route.
        The route now uses the latest waste prediction as the starting point and can adjust coverage based on demand.
        """
    )

    if st.session_state["latest_prediction"] is None:
        st.warning(
            """
            No prediction has been generated yet. Please go to the Waste Prediction page first and run a prediction.
            """
        )
        st.stop()

    latest = st.session_state["latest_prediction"]

    st.markdown("### Prediction Used for Route Optimization")

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    info_col1.metric("Starting Landfill", latest["landfill"])
    info_col2.metric("Month", latest["month"])
    info_col3.metric("Expected Trips", f"{latest['trips']:,}")
    info_col4.metric("Predicted Tonnage", f"{latest['prediction']:,.2f}")

    st.markdown("---")

    route_mode = st.selectbox(
        "Route Planning Mode",
        [
            "Demand-Based Route",
            "All Landfill Route"
        ]
    )

    start_location = latest["landfill"]

    if route_mode == "Demand-Based Route":
        locations = select_demand_based_locations(
            start_location,
            latest["prediction"]
        )

        st.info(
            f"""
            Demand-based routing is active. The system classified this case as **{latest['demand_level']}** 
            and selected the most relevant landfill locations starting from **{start_location}**.
            """
        )

    else:
        other_locations = [
            location for location in LANDFILL_LOCATIONS.keys()
            if location != start_location
        ]

        locations = [start_location] + other_locations

        st.info(
            f"""
            All-landfill routing is active. The route starts from **{start_location}** and covers all landfill sites.
            """
        )

    route = solve_route(
        locations=locations,
        depot_index=0
    )

    if route is None:
        st.error("No route solution found.")
        st.stop()

    route_distance = calculate_route_distance(route)

    st.session_state["latest_route"] = route
    st.session_state["latest_route_distance"] = route_distance
    st.session_state["latest_route_mode"] = route_mode

    st.success("Optimized Route Generated")

    route_col1, route_col2, route_col3 = st.columns(3)

    route_col1.metric("Route Mode", route_mode)
    route_col2.metric("Number of Stops", len(route) - 1)
    route_col3.metric("Estimated Route Distance", f"{route_distance:.2f} km")

    st.markdown("### Route Order")

    for index, stop in enumerate(route):
        st.write(f"**{index + 1}.** {stop}")

    st.subheader("🗺️ Optimized Route Map")

    display_route_map(route)


# ==========================================
# ANALYTICS PAGE
# ==========================================

elif page == "Analytics":

    st.header("Project Analytics")

    st.info(
        """
        Model performance metrics were obtained during Random Forest evaluation.
        """
    )

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

        pred_col1.metric("Landfill", latest["landfill"])
        pred_col2.metric("Month", latest["month"])
        pred_col3.metric("Trips", f"{latest['trips']:,}")
        pred_col4.metric("Predicted Tonnage", f"{latest['prediction']:,.2f}")

        st.write(f"**Demand Level:** {latest['demand_level']}")

    else:
        st.warning("No prediction has been generated yet.")

    st.markdown("---")

    st.subheader("Impact Analysis")

    if st.session_state["latest_route_distance"] is not None:
        optimized_distance = st.session_state["latest_route_distance"]

        traditional_distance = optimized_distance * 1.58

        distance_saved = traditional_distance - optimized_distance

        fuel_saved = distance_saved * 0.15

        st.write(f"Route Mode: **{st.session_state['latest_route_mode']}**")
        st.write(f"Estimated Traditional Distance: **{traditional_distance:.2f} km**")
        st.write(f"Optimized Route Distance: **{optimized_distance:.2f} km**")
        st.write(f"Distance Saved: **{distance_saved:.2f} km**")
        st.write(f"Estimated Fuel Saved: **{fuel_saved:.2f} litres**")

    else:
        traditional_distance = 150
        optimized_distance = 95

        distance_saved = traditional_distance - optimized_distance
        fuel_saved = distance_saved * 0.15

        st.write(f"Estimated Traditional Distance: **{traditional_distance} km**")
        st.write(f"Estimated Optimized Distance: **{optimized_distance} km**")
        st.write(f"Distance Saved: **{distance_saved} km**")
        st.write(f"Estimated Fuel Saved: **{fuel_saved:.2f} litres**")

    st.markdown("---")

    st.subheader("System Interpretation")

    st.write(
        """
        The system combines waste forecasting and route optimization. The prediction module estimates expected
        waste tonnage, while the route optimization module uses the selected landfill as the operational starting
        point. The analytics dashboard summarizes model performance, prediction results, and route impact.
        """
    )