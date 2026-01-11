import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Page Config
st.set_page_config(page_title="Tesla Stock Prediction", layout="wide")

st.title("🚗 Tesla (TSLA) Stock Price Analysis")

# 2. Try to Load Data
try:
    df = pd.read_csv('TSLA.csv')
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading TSLA.csv: {e}")
    st.stop()

# 3. Sidebar Settings
st.sidebar.header("Forecast Settings")
day_choice = st.sidebar.selectbox("Days to Predict Ahead", [1, 5, 10])
train_button = st.sidebar.button("Run Models")

# 4. Preparation Logic
data = df['Adj Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_seq(data, window, step):
    X, y = [], []
    for i in range(window, len(data) - step + 1):
        X.append(data[i-window:i, 0])
        y.append(data[i + step - 1, 0])
    return np.array(X), np.array(y)

# 5. Training and Results
if train_button:
    st.info(f"Training models for a {day_choice}-day forecast... please wait.")
    
    # Prepare Data
    win = 60
    X, y = create_seq(scaled_data, win, day_choice)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # LSTM Model
    model = Sequential([
        LSTM(32, input_shape=(win, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Predict
    preds = model.predict(X_test)
    preds_prices = scaler.inverse_transform(preds)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual_prices, color='black', label='Actual')
    ax.plot(preds_prices, color='blue', label='LSTM Prediction')
    ax.set_title(f"Tesla {day_choice}-Day Forecast")
    ax.legend()
    st.pyplot(fig)

    # --- EVALUATION SECTION (Now correctly indented) ---
    # We use actual_prices and preds_prices which were defined above
    mse_val = mean_squared_error(actual_prices, preds_prices)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(actual_prices, preds_prices)

    st.subheader("📊 Model Evaluation Metrics")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(label="RMSE (Avg Error)", value=f"${rmse_val:.2f}")
        st.caption("Lower is better. Shows avg deviation in Dollars.")

    with m2:
        st.metric(label="MAE", value=f"${mae_val:.2f}")
        st.caption("Mean Absolute Error - average typical error.")

    with m3:
        st.metric(label="MSE", value=f"{mse_val:.2f}")
        st.caption("Mean Squared Error - punishes large outliers.")
    
    st.balloons()
else:
    st.warning("Click 'Run Models' in the sidebar to begin.")