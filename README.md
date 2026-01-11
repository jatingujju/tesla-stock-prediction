# tesla-stock-prediction
# 🚗 Tesla Stock Price Prediction Dashboard

An end-to-end deep learning application that predicts the future "Adjusted Close" price of Tesla (TSLA) using historical data.

## 🌟 Overview
This project compares two Recurrent Neural Network (RNN) architectures to determine which is more effective at capturing the high volatility of Tesla's stock:
1. **SimpleRNN**: A baseline recurrent model.
2. **LSTM (Long Short-Term Memory)**: A more advanced model designed to solve the vanishing gradient problem.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Deep Learning**: TensorFlow, Keras
- **Web App**: Streamlit
- **Data Handling**: Pandas, NumPy, Scikit-Learn
- **Visualization**: Matplotlib

## 📊 Features
- **Time-Series Forecasting**: Choose between 1, 5, or 10-day prediction windows.
- **Data Normalization**: Uses `MinMaxScaler` to optimize neural network training.
- **Evaluation Metrics**: Displays real-time **RMSE**, **MAE**, and **MSE** to verify model accuracy.

## 🚀 How to Run Locally
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 📈 Evaluation
The project includes automated performance tracking. Key metrics used for the final report:
- **Root Mean Squared Error (RMSE)**: Measures the average magnitude of the error.
- **Mean Absolute Error (MAE)**: Measures the average typical deviation from the actual price.
