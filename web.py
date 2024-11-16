import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and the DataFrame
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# App title
st.title("Laptop Price Predictor")

# Input fields
company = st.selectbox('Brand', df['Company'].unique())
type_name = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', sorted(df['Ram'].unique()))
weight = st.number_input('Weight of the Laptop (in kg)', min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Panel', ['No', 'Yes'])
screen_size = st.number_input('Screen Size (in Inches)', min_value=10.0, max_value=20.0, step=0.1)
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '3840x2160', '1600x900', '3200x1800', '2560x1440', '2560x1600'
])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.number_input('HDD (in GB)', min_value=0, max_value=2048, step=1)
ssd = st.number_input('SSD (in GB)', min_value=0, max_value=2048, step=1)
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

# Feature engineering
x_res, y_res = map(int, resolution.split('x'))
ppi = ((x_res**2 + y_res**2)**0.5) / screen_size
touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

# Create input DataFrame
query_df = pd.DataFrame({
    'Company': [company],
    'TypeName': [type_name],
    'Ram': [ram],
    'Weight': [weight],
    'Touchscreen': [touchscreen],
    'Ips': [ips],
    'ppi': [ppi],
    'Cpu brand': [cpu],
    'HDD': [hdd],
    'SSD': [ssd],
    'Gpu brand': [gpu],
    'os': [os]
})

# Debug: Display input values
st.write("Inputs passed to the model:")
st.write(query_df)

# Prediction button
if st.button('Predict Price'):
    try:
        raw_prediction = pipe.predict(query_df)[0]
        st.write(f"The predicted price of this configuration is: {raw_prediction}")

        # Safeguard for unrealistic predictions
        if raw_prediction > 5000:  # Log price > 20 is unreasonably high
            st.error("The predicted price might differ from the current market value.")
        else:
            predicted_price = np.exp(raw_prediction)
            st.success(f"The predicted price of this configuration is: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

