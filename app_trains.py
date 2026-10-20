import streamlit as st
import pandas as pd
import pickle

# Set page config
st.set_page_config(page_title="Mumbai Local Travel Time Predictor", page_icon="🚂")

# 1. Load the 3 trained models
try:
    lr_model = pickle.load(open('linear_regression_model.pkl', 'rb'))
    dt_model = pickle.load(open('decision_tree_model.pkl', 'rb'))
    knn_model = pickle.load(open('knn_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please make sure the 3 .pkl files are in the same folder as app_trains.py")
    st.stop()

models = {
    'Linear Regression': lr_model,
    'Decision Tree': dt_model,
    'KNN': knn_model
}

# 2. Combined Stations & Distances (Western + Central Line)
all_stations = {
    # Western Line
    'Churchgate': 0, 'Marine Lines': 1.5, 'Charni Road': 2.5, 'Grant Road': 3.5,
    'Mumbai Central': 4.5, 'Mahalaxmi': 5.5, 'Lower Parel': 6.5, 'Prabhadevi': 7.5,
    'Dadar': 8.5, 'Matunga Road': 9.5, 'Mahim Jn': 10.5, 'Bandra': 12.0,
    'Khar Road': 13.5, 'Santacruz': 14.5, 'Vile Parle': 15.5, 'Andheri': 17.0,
    'Jogeshwari': 18.5, 'Ram Mandir': 19.5, 'Goregaon': 20.5, 'Malad': 22.0,
    'Kandivali': 23.5, 'Borivali': 25.5, 'Dahisar': 27.0, 'Mira Road': 29.0,
    'Bhayandar': 31.0, 'Naigaon': 33.5, 'Vasai Road': 35.5, 'Nallasopara': 38.0,
    'Virar': 40.0,
    # Central Line
    'CSMT': 0, 'Masjid': 1, 'Sandhurst Road': 2, 'Byculla': 4, 'Chinchpokli': 5,
    'Currey Road': 6.5, 'Parel': 8, 'Matunga': 11, 'Sion': 13, 'Kurla': 15,
    'Vidyavihar': 18, 'Ghatkopar': 21, 'Vikhroli': 23, 'Kanjurmarg': 25,
    'Bhandup': 26, 'Nahur': 28, 'Mulund': 32, 'Thane': 34, 'Kalwa': 36,
    'Mumbra': 40, 'Diva': 43, 'Kopar': 47, 'Dombivli': 48, 'Thakurli': 50,
    'Kalyan': 54, 'Vithalwadi': 56, 'Ulhasnagar': 58, 'Ambernath': 61,
    'Badlapur': 67, 'Vangani': 74, 'Shelu': 78, 'Neral': 83, 'Bhivpuri Road': 88,
    'Karjat': 100, 'Shahad': 56, 'Ambivli': 59, 'Titwala': 64, 'Khadavli': 71,
    'Vasind': 77, 'Asangaon': 83, 'Atgaon': 91, 'Thansit': 96, 'Khardi': 101,
    'Umbermali': 109, 'Kasara': 120
}
stations = list(all_stations.keys())

# 3. UI Setup
st.title('🚂 Mumbai Local Train Travel Time Predictor')
st.markdown('Enter your journey details to predict the expected travel time!')

col1, col2 = st.columns(2)

with col1:
    dep_station = st.selectbox('Departure Station', sorted(stations), index=stations.index('Andheri'))
    train_type = st.selectbox('Train Type', ['Fast', 'Slow'])
    time_of_day = st.slider('Time of Day (24hr)', 4, 23, 9)
    season = st.selectbox('Season', ['Summer', 'Monsoon', 'Winter'])

with col2:
    # Filter arrival stations to be geographically after departure
    possible_arrivals = [s for s in stations if all_stations[s] > all_stations[dep_station]]
    if not possible_arrivals:
        st.error("No stations available after this departure station on this line.")
        st.stop()
    arr_station = st.selectbox('Arrival Station', sorted(possible_arrivals))
    day_of_week = st.selectbox('Day of Week', ['Weekday', 'Weekend'])
    weather = st.selectbox('Weather', ['Clear', 'Cloudy', 'Raining'])

# 4. Auto-calculate distance and stops
distance = all_stations[arr_station] - all_stations[dep_station]
num_stops = max(1, int(distance / 2.5))

# 5. Prediction Logic
if st.button('Predict Travel Time'):
    input_df = pd.DataFrame({
        'departure_station': [dep_station],
        'arrival_station': [arr_station],
        'distance_km': [round(distance, 1)],
        'num_stops': [num_stops],
        'train_type': [train_type],
        'time_of_day': [time_of_day],
        'day_of_week': [day_of_week],
        'season': [season],
        'weather': [weather]
    })

    st.subheader("Prediction Results:")
    st.write(f"📏 **Distance:** {round(distance, 1)} km | 🛑 **Stops:** {num_stops}")
    st.markdown("---")

    for name, model in models.items():
        pred = model.predict(input_df)[0]
        st.info(f"**{name}:** {round(pred, 1)} minutes")