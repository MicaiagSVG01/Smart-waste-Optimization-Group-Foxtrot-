# Smart-waste-Optimization-Group-Foxtrot-

## Project Title

**Smart Waste Optimization System for Lagos Landfill Operations**

## Project Overview

The **Smart Waste Optimization System** is a machine learning-driven web application designed to support smarter waste management planning in Lagos State, Nigeria.

The system predicts expected waste tonnage for selected disposal landfill sites and provides direction support to help users navigate to the selected disposal landfill. It was developed to move waste management from manual guesswork to data-driven decision-making.

The project uses real-world Lagos waste accumulation data obtained from **LAWMA**, covering waste records from **2020 to 2024**.

## Problem Statement

Lagos generates a large amount of municipal waste daily due to rapid urbanization, population growth, increased commercial activity, and rising consumption patterns.

Traditional waste management systems often depend on manual planning, fixed schedules, and reactive responses. This can lead to:

* Delayed waste collection
* Overflowing waste sites
* Increased landfill pressure
* High fuel and operational costs
* Poor resource allocation
* Environmental pollution
* Public health risks

This project addresses these challenges by using machine learning to predict expected waste tonnage and support better waste planning.

## Aim of the Project

The aim of this project is to develop a smart waste optimization system that predicts landfill waste accumulation and supports improved decision-making for waste management operations in Lagos State.

## Key Objectives

* Use real-world Lagos waste accumulation data from LAWMA
* Preprocess and prepare waste data for machine learning
* Engineer useful features for prediction
* Train a Random Forest Regression model to predict waste tonnage
* Evaluate the model using standard regression metrics
* Build an interactive Streamlit web application
* Provide Google Maps direction support to selected disposal landfill sites
* Present prediction results in a simple and user-friendly format

## Dataset

The dataset used for this project was obtained from **LAWMA** and contains real-world Lagos waste accumulation records from **2020 to 2024**.

The dataset includes information used to understand waste patterns across selected landfill/disposal sites in Lagos.

## Machine Learning Approach

This project uses a **Random Forest Regressor** model.

Random Forest is a **supervised machine learning algorithm**. In this project, it was used for **supervised regression** because the model predicts a continuous numerical value: **waste tonnage**.

The model learns from historical waste records where the actual waste tonnage is already known. It then uses that learning to predict expected waste tonnage for new inputs.

## Features Used by the Model

The model uses the following input features:

* **Year**
  Represents the year of the waste record or prediction.

* **Month Number**
  Converts month names into numerical values. For example, January is 1, February is 2, and December is 12.

* **Quarter**
  Groups months into quarters of the year:

  * Quarter 1: January to March
  * Quarter 2: April to June
  * Quarter 3: July to September
  * Quarter 4: October to December

* **Festive Period**
  Identifies whether the selected month falls within selected festive months. In this prototype, March, May, and December are treated as festive months.

* **Population Density**
  Represents population pressure around landfill catchment areas. In this prototype, estimated proxy values were used. In a real-world deployment, official demographic data should be used.

* **Landfill Encoded Value**
  Converts landfill names into numerical values so the machine learning model can process them.

* **Expected Trips**
  Represents the expected number of waste collection trips.

## Target Variable

The target variable is:

**Waste Tonnage**

This is the value the model predicts.

## Model Evaluation

The model was evaluated using standard regression metrics:

### MAE — Mean Absolute Error

MAE shows the average difference between the actual waste tonnage and the predicted waste tonnage.
A lower MAE means the model’s predictions are closer to the actual values.

### RMSE — Root Mean Squared Error

RMSE measures prediction error and gives more attention to larger errors.
A lower RMSE means the model is making fewer large mistakes.

### R² Score — R-squared Score

R² score, also called the coefficient of determination, shows how well the model explains the pattern in the data.
The closer the value is to 1, the better the model performance.

### Model Performance Results

* **MAE:** 2211.95
* **RMSE:** 6459.47
* **R² Score:** 0.98

These results show that the Random Forest model performed strongly in capturing the waste accumulation pattern in the dataset.

## Application Features

The application was built using **Streamlit** and contains three main modules:

### 1. Waste Prediction

This module allows users to select:

* Disposal landfill
* Year
* Month
* Expected trips

The app processes these inputs and predicts the expected waste tonnage using the trained Random Forest model.

### 2. Route Optimization and Directions

This module provides direction support to the selected disposal landfill.

The app stores coordinates for each disposal landfill. When the user enters a current location and selects a disposal landfill, the app creates a Google Maps direction URL.

The Google Maps link uses:

* User’s current location as the origin
* Selected disposal landfill coordinates as the destination
* Driving mode as the travel mode

When the user clicks the button, Google Maps opens and provides live driving directions.

### 3. Analytics Dashboard

The Analytics page displays:

* Model performance metrics
* Selected disposal landfill
* Month
* Expected trips
* Predicted waste tonnage
* Demand level
* Festive period status

## Demand Level Classification

The predicted waste tonnage is classified into three demand levels:

* **Low Demand:** Below 10,000 tonnes
* **Moderate Demand:** 10,000 to 25,000 tonnes
* **High Demand:** Above 25,000 tonnes

This makes the result easier for users to understand.

## Google Maps Integration

The app does not replace Google Maps. Instead, it generates a Google Maps direction link from the code.

The app uses stored landfill coordinates and combines them with the user’s current location to create a Google Maps URL.

Example logic:

```python
https://www.google.com/maps/dir/?api=1&origin=current_location&destination=landfill_coordinates&travelmode=driving
```

Streamlit then displays this URL as a clickable button. When clicked, Google Maps opens with the driving directions.

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn
* Joblib
* Folium
* Streamlit-Folium
* Google Maps Direction Link
* Random Forest Regressor

## Project Structure

```text
Smart-waste-Optimization-Group-Foxtrot-
│
├── app.py
├── random_forest.pkl
├── requirements.txt
├── README.md
└── dataset / data files
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/MicaiagSVG01/Smart-waste-Optimization-Group-Foxtrot-.git
```

### 2. Navigate into the project folder

```bash
cd Smart-waste-Optimization-Group-Foxtrot-
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

## Requirements

Your `requirements.txt` file should include:

```text
streamlit
pandas
joblib
scikit-learn
folium
streamlit-folium
```

## Deployment

The application can be deployed on **Streamlit Community Cloud**.

Live App:
https://ypazxarn6vfxjprmtlpcxt.streamlit.app/

## Limitations

This system is a prototype and has some limitations:

* The model depends on the quality and completeness of available data
* Population density values were estimated proxy values
* The app does not fully integrate live traffic or road closure data directly
* Google Maps handles live navigation externally
* More real-time operational data would improve system performance
* Users may need training before using the system in real waste management operations

## Future Enhancements

Future improvements may include:

* Integration with live LAWMA waste data streams
* GPS tracking of waste collection trucks
* IoT-enabled smart bins
* Real-time traffic and road condition integration
* Mobile app deployment
* Use of official population density data
* Dynamic route re-optimization
* Improved analytics dashboard

## Key Takeaway

**Data enables prediction. Prediction enables optimization. Optimization improves cities.**

This project demonstrates how real-world waste data and machine learning can support smarter waste management planning in Lagos.

## Team

**Group Foxtrot**
Tech Crush Africa
June 2026

## Acknowledgment

We acknowledge LAWMA for the real-world Lagos waste accumulation data used as the foundation for this project.
