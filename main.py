import pickle
from matplotlib.colors import ListedColormap
import pandas as pd
from datetime import datetime
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import polyline
from helper import DF_DUTIES

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

import pandas as pd
import streamlit as st

def load_data():
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

    if st.session_state['uploaded_file'] is None:
        uploaded_file = st.file_uploader("Please upload an Excel file (.xlsx)", type="xlsx")

        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file  

    if st.session_state['uploaded_file'] is not None:
        df = pd.read_excel(st.session_state['uploaded_file'])

        # Fixing the polyline column by removing the 'b' prefix and converting back to bytes
        if 'polyline' in df.columns:
            def fix_polyline(polyline_str):
                if isinstance(polyline_str, str) and polyline_str.startswith("b'"):
                    # Remove the "b'" prefix and the trailing "'"
                    polyline_str = polyline_str[2:-1]
                return polyline_str

            df['polyline'] = df['polyline'].apply(fix_polyline)

        return df
    else:
        st.warning("You must upload an Excel file to proceed.")
        st.stop()

df = load_data()
   
        
df = load_data()

# with open("trips_ALL.pickle", "rb") as file:
#     trips = pickle.load(file)
# df = pd.DataFrame(trips)



def extract_reg_plate(mobile_device):
    return mobile_device.split("_")[-1]

df['reg_plate'] = df['mobile_device'].apply(extract_reg_plate)
df['start_datetime'] = pd.to_datetime(df['start_datetime'])
df_duties = DF_DUTIES.copy()
df_duties["DUTY"] = df_duties.apply(
    lambda row: f"SPARE-{row['REGISTRATION']}" if "SPARE" in row["DUTY"] else row["DUTY"], axis=1
)

df_duties["reg_plate"] = df_duties["REGISTRATION"].str.replace(" ", "")

reg_plate_label_dict = {
    'reg_plate' : [],
    'label' : [],
}

for idx,row in df_duties.iterrows():
    reg_plate_label_dict["reg_plate"].append(row['reg_plate'])
    reg_plate_label_dict["label"].append("Van" + " #" + row['VAN'] + f" ({row['reg_plate']})")
    
df_convert = pd.DataFrame(reg_plate_label_dict)

def display_route(polyline_str):
    if isinstance(polyline_str, bytes):
        polyline_str = polyline_str.decode('utf-8')
        
    decoded_coords = polyline.decode(polyline_str)
    
    swapped_coords = [(coord[1], coord[0]) for coord in decoded_coords]

    m = folium.Map(location=swapped_coords[0], zoom_start=13)

    folium.PolyLine(swapped_coords, color="blue", weight=2.5, opacity=1).add_to(m)

    return m


def vehicle_analyzer():
    st.write("### Vehicle Analyzer")
    
    # Set default vehicle label for Vehicle #1
    default_vehicle = df_convert[df_convert['label'].str.contains('Van #1')]['label'].iloc[0]
    
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        selected_vehicle = st.selectbox(
            'Select a vehicle:', 
            df_convert['label'].unique(),
            index=df_convert['label'].tolist().index(default_vehicle),  # Set Vehicle #1 as default
            help="Choose a vehicle from the list"
        )

    with col2:
        min_distance = st.number_input('Min distance (in km):', min_value=0.0, value=0.0, step=0.1)

    with col3:
        min_duration = st.number_input('Min duration (in seconds):', min_value=0, value=0, step=100)

    col4, col5, col6 = st.columns(3)
    
    with col4:
        filter_arrived_depot = st.checkbox("Only arrived to depot", value=True)
    
    with col5:
        filter_distance_gt_5km = st.checkbox("Only trips > 5 km", value=True)

    with col6:
        filter_duration_gt_15min = st.checkbox("Only trips > 15 min", value=True)

    reg_plate_value = df_convert[df_convert["label"] == selected_vehicle]['reg_plate'].iloc[0]
    filtered_trips = df[df['reg_plate'] == reg_plate_value]
    
    if filter_arrived_depot:
        filtered_trips = filtered_trips[filtered_trips['arrived_to_depot']]
    
    if filter_distance_gt_5km:
        filtered_trips = filtered_trips[filtered_trips["distance_in_km"].gt(5)]
    
    if filter_duration_gt_15min:
        filtered_trips = filtered_trips[filtered_trips['duration_in_seconds'].gt(15 * 60)]

    filtered_trips = filtered_trips[
        (filtered_trips['distance_in_km'] >= min_distance) & 
        (filtered_trips['duration_in_seconds'] >= min_duration)
    ]

    if not filtered_trips.empty:
        mean_distance = filtered_trips['distance_in_km'].mean()
        median_distance = filtered_trips['distance_in_km'].median()
        min_distance_val = filtered_trips['distance_in_km'].min()
        max_distance_val = filtered_trips['distance_in_km'].max()
        perc_95_distance = np.percentile(filtered_trips['distance_in_km'], 95)
    else:
        mean_distance = median_distance = min_distance_val = max_distance_val = perc_95_distance = 0
        
    st.write("### Trip Distance Statistics:")
    col4, col5, col6, col7, col8 = st.columns(5)
    col4.metric("Mean", f"{mean_distance:.1f} km")
    col5.metric("Min", f"{min_distance_val:.1f} km")
    col6.metric("Median", f"{median_distance:.1f} km")
    col7.metric("95th Percentile", f"{perc_95_distance:.1f} km")
    col8.metric("Max", f"{max_distance_val:.1f} km")
    
    st.write("### All routes for the selected vehicle:")
    st.dataframe(data=filtered_trips[['trip_id', 'start_datetime', 'end_datetime', 'distance_in_km', 'duration_in_seconds']], width=1000)

    filtered_trips["date"] = filtered_trips["start_datetime"].dt.date
    aggregated_data = filtered_trips.groupby("date").agg(
        total_distance=("distance_in_km", "sum"),
        total_duration=("duration_in_seconds", "sum")
    ).reset_index()
    
    st.write("### Aggregated Distance and Duration by Day:")
    st.dataframe(aggregated_data, width=1000)
        
    selected_trip_id = st.selectbox(
        "Select a trip to view its route on the map:",
        filtered_trips['trip_id'],
        format_func=lambda x: f"Trip ID {x}"
    )
    
    if selected_trip_id:
        trip_polyline = filtered_trips[filtered_trips['trip_id'] == selected_trip_id]['polyline'].values[0]

        st.write("Trip Route on Map:")
        route_map = display_route(trip_polyline)
        st_folium(route_map, width=700, height=500)
        
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    filtered_trips['week_year'] = filtered_trips['start_datetime'].dt.strftime('%Y-%U')
    filtered_trips['day_of_week'] = filtered_trips['start_datetime'].dt.day_name()
    
    filtered_trips['day_of_week'] = pd.Categorical(filtered_trips['day_of_week'], categories=day_order, ordered=True)
    heatmap_data = filtered_trips.groupby(['week_year', 'day_of_week']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = ListedColormap(['lightgray', '#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027'])
    sns.heatmap(heatmap_data, cmap=cmap, annot=True, fmt="d", linewidths=.5, ax=ax, )
    st.pyplot(fig)


def vehicles_on_sunday():
    st.write('### Number of trips on Sundays')
    
    df['is_sunday'] = df['start_datetime'].dt.day_name() == 'Sunday'

    sunday_trips = df[df['is_sunday']].groupby('reg_plate').size().reset_index()
    
    sunday_trips.columns = ['reg_plate', 'sunday_trip_count']

    sunday_trips = sunday_trips.sort_values(by='sunday_trip_count', ascending=False)

    sunday_trips = sunday_trips.reset_index(drop=True)

    st.dataframe(sunday_trips, width=600)

def duty_analyzer():
    st.write("### Duty Analyzer")

    df_merged = df.merge(df_duties, on='reg_plate', how='inner')
    aggregated_df = df_merged.groupby("DUTY").agg(
        total_distance=("distance_in_km", "sum"),
        total_duration=("duration_in_seconds", "sum"),
        num_of_routes =("reg_plate", "count"),
    ).reset_index()
    st.dataframe(aggregated_df, width=800)
    

def week_analyzer():
    st.write("### Week Analyzer")
    
    # Select a week from the year
    selected_week = st.text_input("Enter a week e.g., 2024-10", "2024-10")
    
    try:
        year, week = selected_week.split('-')
        week = int(week)
        year = int(year)
    except ValueError:
        st.error("Please enter the week in the correct format (YYYY-WW).")
        return

    df['week_year'] = df['start_datetime'].dt.strftime('%Y-%U')
    df_filtered = df[df['week_year'] == f"{year}-{week:02d}"]

    if df_filtered.empty:
        st.write(f"No trips found for the selected week: {selected_week}")
        return
    
    df_filtered['day_of_week'] = df_filtered['start_datetime'].dt.day_name()
    df_pivot = df_filtered.pivot_table(
        index='reg_plate', 
        columns='day_of_week', 
        values='distance_in_km', 
        aggfunc='sum', 
        fill_value=0
    )
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_pivot = df_pivot.reindex(columns=day_order)

    st.write("### Total Distance Covered by Each Vehicle per Day of the Week")
    st.dataframe(df_pivot, width=800)

    df_heatmap = df_filtered.pivot_table(
        index='reg_plate',
        columns='day_of_week',
        values='trip_id',
        aggfunc='count',
        fill_value=0
    )
    
    df_heatmap = df_heatmap.reindex(columns=day_order)

    fig, ax = plt.subplots(figsize=(20, 24))
    cmap = ListedColormap(['lightgray', '#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027'])
    sns.heatmap(df_heatmap, cmap=cmap, annot=True, fmt="d", linewidths=.5, ax=ax)
    
    st.write("### Heatmap of Number of Routes per Car per Day of the Week")
    st.pyplot(fig)

    


st.sidebar.title("Vehicle Dashboard")
menu = st.sidebar.radio("Select a Menu", ["Vehicles on Sunday", "Vehicle Analyzer", "Duty Analyzer", "Week Analyzer"])


if menu == "Vehicles on Sunday":
    vehicles_on_sunday()  
elif menu == "Vehicle Analyzer":
    vehicle_analyzer() 
elif menu == "Duty Analyzer":
    duty_analyzer()
elif menu == "Week Analyzer":
    week_analyzer()