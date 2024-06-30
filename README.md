# LSTM Stock Predictor

This Stock Predictor, developed by WeiEn Weng, is a comprehensive tool for predicting stock prices using various machine learning models. It integrates data fetching, preprocessing, and predictions with a user-friendly interface built using PyQt5.

## Features

- **Data Fetching**: Automatically fetches stock data from Yahoo Finance.
- **Preprocessing**: Preprocesses data including technical indicators and custom candlestick patterns.
- **Model Predictions**:
  - LSTM (Long Short-Term Memory)
  - ARIMA (AutoRegressive Integrated Moving Average)
  - Random Forest
  - Technical Analysis
- **Google Sheets Integration**: Saves and loads data to and from Google Sheets.
- **Visualization**: Interactive charts with Matplotlib embedded in PyQt5.
- **Metrics Calculation**: Evaluates models using MSE, RMSE, MAE, MAPE, and standard deviation of errors.
- **Concurrent Execution**: Utilizes ThreadPoolExecutor for running models concurrently.

## Prerequisites

- Python 3.7+
- Required Libraries: Install using `pip install -r requirements.txt`
  - PyQt5
  - NumPy
  - Pandas
  - Torch
  - scikit-learn
  - yfinance
  - ta (Technical Analysis Library)
  - statsmodels
  - gspread
  - google-auth

## Setup

1. **Clone the repository**:
    \```bash
    git clone https://github.com/weienweng/LSTM-Stock-Predictor.git
    cd LSTM-Stock-Predictor
    \```

2. **Install dependencies**:
    \```bash
    pip install -r requirements.txt
    \```

3. **Google Sheets API Setup**:
    - Create a Google Cloud project.
    - Enable the Google Sheets API.
    - Create service account credentials.
    - Download the credentials JSON file and place it in the project directory.

4. **Configuration**:
    - Update `SERVICE_ACCOUNT_FILE` and `SPREADSHEET_ID` in the script with your credentials file path and spreadsheet ID.

## Usage

1. **Run the application**:
    \```bash
    python app.py
    \```

2. **User Interface**:
    - **Search Bar**: Enter a stock symbol to fetch data.
    - **Date Range**: Select start and end dates for the data range.
    - **Prediction Period**: Choose the period for future predictions (e.g., One Month).
    - **Buttons**: 
        - `Start Prediction`: Begins the prediction process.
        - `Pause`: Pauses the ongoing prediction.
        - `Recalculate All`: Recalculates predictions for all stocks in Google Sheets.
        - `Import Existing Data`: Imports existing data from Google Sheets.

3. **Progress and Status**:
    - **Progress Bar**: Shows the progress of the prediction process.
    - **ETA**: Estimated time remaining for the prediction process.
    - **Status Text**: Displays status updates and logs.

4. **Tabs**:
    - **Charts**: Interactive charts for combined and individual model predictions.
    - **Metrics**: Displays evaluation metrics for the models.
    - **Data**: Shows the data used for predictions.
    - **Financial Data**: Displays financial metrics of the stock.

## Exporting Charts

- Each chart tab has an export button that allows you to save the chart as a PNG file.

## Error Handling

- The application includes logging for debugging and error tracking. Errors are displayed in the status text area and logged to the console.

## Acknowledgements

- This application leverages various open-source libraries and APIs, including PyQt5, Torch, Yahoo Finance API, and Google Sheets API.

---

This README provides an overview of the Stock Predictor, its features, setup instructions, and usage guidelines. For detailed implementation and code understanding, refer to the source code files in the repository.
