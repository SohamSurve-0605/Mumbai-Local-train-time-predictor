# 🚆 Mumbai Local Train Travel Time Predictor

An end-to-end Machine Learning Regression project comparing Linear Regression, Decision Tree, and KNN models to predict travel times between stations on the Western and Central lines of the Mumbai Local network.

![App Preview](dashboard_preview.png)

## 📌 Project Overview
This project utilizes a 30,000-row synthetic dataset simulating real-world Mumbai local train journeys. It analyzes the impact of distance, number of stops, train type (Fast/Slow), peak hours, and monsoon weather on travel time. The goal is to provide a data-driven prediction of expected travel time to help commuters plan their journeys.

## 🛠️ Tech Stack
*   **Python:** Core programming language.
*   **Pandas & NumPy:** Data manipulation and feature engineering.
*   **Scikit-Learn:** Pipeline, ColumnTransformer, OneHotEncoder, StandardScaler, GridSearchCV.
*   **Streamlit:** Interactive web UI for local deployment.

## 📊 Model Performance
Three regression models were trained, tuned via GridSearchCV, and evaluated using MAE and R² Score.

| Model | MAE (Mins) | R² Score | Notes |
| :--- | :--- | :--- | :--- |
| **Decision Tree** | **3.20** | **0.9918** | **Best overall performance (Winner)** |
| Linear Regression | 7.94 | 0.9639 | Good baseline, but missed non-linear delays |
| KNN | 7.92 | 0.9631 | Suffered from curse of dimensionality |

*Decision Tree was chosen as the best model due to its high R² score, proving its superior ability to capture the deterministic, rule-based delays of peak hours and monsoon weather.*

## 🚀 How to Run Locally
1. Clone this repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt