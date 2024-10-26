import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and the DataFrame used for training
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Get user input
company = st.selectbox('Brand', df['Company'].unique())
type_name = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop (in kg)', min_value=0.0)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])
screen_size = st.slider('Screen size (in inches)', 10.0, 18.0, 15.6)
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

# Predict Price Button
if st.button('Predict Price'):
    # Convert categorical inputs to numerical
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Extracting resolution
    X_res, Y_res = map(int, resolution.split('x'))
    
    # Calculate PPI (pixels per inch)
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create a DataFrame for the query
    query_df = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'IPS': [ips],  # Use the correct casing
        'PPI': [ppi],  # Use the correct casing
        'Cpu brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        'os': [os]
    })

    # Print the query DataFrame for debugging
    st.write("Query DataFrame:", query_df)

    # Ensure the column order matches the training DataFrame
    query_df = query_df[['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'IPS', 'PPI', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']]

    # Make predictions using the model pipeline
    predicted_price = np.exp(pipe.predict(query_df)[0])  # Assuming log transformation was used

    # Display the predicted price
    st.title(f"The predicted price of this configuration is ₹{int(predicted_price)}")


