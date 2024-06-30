import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QPushButton,
                             QTabWidget, QTableWidget, QTableWidgetItem, QLineEdit, QCompleter, QProgressBar,
                             QDateEdit, QLabel, QListWidget, QMessageBox, QComboBox, QSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QStringListModel, QObject, QTimer, QDate
from PyQt5.QtGui import QMovie
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import yfinance as yf
from datetime import datetime, timedelta
import logging
import gspread
from google.oauth2.service_account import Credentials
import matplotlib.pyplot as plt
import matplotlib
import traceback
from matplotlib.dates import DateFormatter
import ta
from statsmodels.tsa.arima.model import ARIMA
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Use the Qt5Agg backend for Matplotlib
matplotlib.use('Qt5Agg')

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Google Sheets setup
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'PASTE_YOUR_JSON_FILE_PATH'  # Path to your service account credentials JSON file
SPREADSHEET_ID = 'YOUR_SHEET_ID'  # Your Google Sheets ID

# Authorize Google Sheets client
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
client = gspread.authorize(creds)

# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# SignalEmitter class for PyQt signals
class SignalEmitter(QObject):
    update_chart = pyqtSignal(dict)
    update_eta = pyqtSignal(str)
    show_loading = pyqtSignal(bool)

# Thread class for running stock prediction
class StockPredictionThread(QThread):
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, window, start_date, end_date, prediction_period):
        super().__init__()
        self.window = window
        self.symbol = window.symbol
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_period = prediction_period
        self.is_paused = False
        self.start_time = None
        self.pause_time = None
        self.elapsed_time = 0
        self.partial_results = None
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Run method for the thread
    def run(self):
        try:
            self.start_time = datetime.now()
            self.update_signal.emit(f"Starting to process data for {self.symbol}...")
            
            # Load data from Google Sheets
            sheet_data = self.window.load_from_google_sheets(self.start_date, self.end_date)
            
            # Check if we can use previously calculated data
            if sheet_data and sheet_data['dates'][-1] == datetime.now().date() - timedelta(days=1):
                self.update_signal.emit("Using data calculated yesterday.")
                self.window.signal_emitter.update_chart.emit(sheet_data)
                self.finished_signal.emit(sheet_data)
                return
            
            self.update_signal.emit(f"Fetching new data for {self.symbol}...")
            
            # Fetch new stock data
            stock_data = self.fetch_stock_data(self.symbol, self.start_date, self.end_date)
            
            if stock_data.empty:
                self.update_signal.emit("Unable to retrieve stock data. Terminating process.")
                return
            
            start_date = stock_data.index[0].strftime('%Y-%m-%d')
            end_date = stock_data.index[-1].strftime('%Y-%m-%d')
            self.update_signal.emit(f"Data range: from {start_date} to {end_date}")
            
            self.update_signal.emit("Preprocessing data...")
            processed_data, scaler, original_close = self.preprocess_data(stock_data)
            
            if processed_data is None or len(processed_data) == 0:
                self.update_signal.emit("No usable data after preprocessing. Terminating process.")
                return
            
            # Run all models concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                futures.append(executor.submit(self.run_lstm_model, processed_data, scaler, original_close))
                futures.append(executor.submit(self.run_arima_model, original_close))
                futures.append(executor.submit(self.run_random_forest_model, processed_data, scaler, original_close))
                futures.append(executor.submit(self.run_technical_analysis, stock_data))

                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        self.update_signal.emit(f"Error in model: {str(e)}")
                        logging.error(f"Error in model: {str(e)}")
                        logging.error(traceback.format_exc())

            if not results:
                self.update_signal.emit("All models failed. Terminating process.")
                return

            # Combine results from all models
            combined_results = self.combine_results(results, stock_data)

            # Fetch financial data for the stock
            financial_data = self.get_financial_data(self.symbol)
            combined_results['financial_data'] = financial_data
            combined_results['stock_data'] = stock_data

            # Save results to Google Sheets and update the UI
            self.save_to_google_sheets(combined_results)
            self.window.signal_emitter.update_chart.emit(combined_results)
            self.finished_signal.emit(combined_results)
    
        except Exception as e:
            logging.error(f"Error occurred during prediction process: {str(e)}")
            logging.error(traceback.format_exc())
            self.update_signal.emit(f"Error occurred during prediction process: {str(e)}")

    # Fetch stock data using yfinance
    def fetch_stock_data(self, symbol, start_date, end_date):
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            logging.warning(f"Unable to retrieve data for symbol: {symbol}")
        return df

    # Preprocess the stock data
    def preprocess_data(self, stock_data):
        if len(stock_data) < 30:  # Ensure we have enough data
            logging.warning("Insufficient data for preprocessing")
            return None, None, None

        data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        original_close = data['Close'].copy()
        
        # Add technical indicators
        data['SMA_5'] = ta.trend.sma_indicator(data['Close'], window=5)
        data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        data['MACD'] = ta.trend.macd_diff(data['Close'])
        bb_indicator = ta.volatility.BollingerBands(data['Close'])
        data['BB_upper'] = bb_indicator.bollinger_hband()
        data['BB_lower'] = bb_indicator.bollinger_lband()
        
        # Add custom candlestick patterns
        data['Doji'] = self.detect_doji(data)
        data['Hammer'] = self.detect_hammer(data)
        data['ShootingStar'] = self.detect_shooting_star(data)
        
        # Additional indicators
        data['EMA_10'] = ta.trend.ema_indicator(data['Close'], window=10)
        data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
        data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        
        # Add more advanced indicators
        data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
        data['CCI'] = ta.trend.cci(data['High'], data['Low'], data['Close'])
        data['STOCH'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
        data['TRIX'] = ta.trend.trix(data['Close'])
        data['WILLR'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'])
        
        data = data.dropna()
        
        if data.empty:
            logging.warning("Data preprocessing resulted in an empty dataset.")
            return None, None, None

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        return scaled_data, scaler, original_close

    # Detect Doji candlestick pattern
    def detect_doji(self, data):
        return (abs(data['Open'] - data['Close']) / (data['High'] - data['Low']) < 0.1).astype(float)

    # Detect Hammer candlestick pattern
    def detect_hammer(self, data):
        return ((data['Low'] - data['Open']) / (data['High'] - data['Low']) > 0.6).astype(float)

    # Detect Shooting Star candlestick pattern
    def detect_shooting_star(self, data):
        return ((data['High'] - data['Close']) / (data['High'] - data['Low']) > 0.6).astype(float)

    # Prepare sequences for LSTM input
    def prepare_sequences(self, data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length, 3])  # Predict closing price
        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y)).view(-1, 1)
        return X, y

    # Train the model
    def train_model(self, model, train_loader, val_loader, num_epochs, learning_rate, fold, total_epochs, current_epoch):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        for epoch in range(num_epochs):
            if self.is_paused:
                return train_losses, val_losses
    
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            if val_loader:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        y_pred = model(X_batch)
                        loss = criterion(y_pred, y_batch)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
            
            self.update_signal.emit(f'Fold {fold}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}' +
                                    (f', Val Loss: {val_loss:.6f}' if val_loader else ''))
            self.progress_signal.emit(int((current_epoch + epoch + 1) / total_epochs * 100))
            
            # Calculate and emit ETA
            elapsed_time = time.time() - start_time
            time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = total_epochs - (current_epoch + epoch + 1)
            eta_seconds = time_per_epoch * remaining_epochs
            eta = str(timedelta(seconds=int(eta_seconds)))
            self.window.signal_emitter.update_eta.emit(f"ETA: {eta}")
        
        return train_losses, val_losses if val_loader else None

    # Predict future prices using the trained model
    def predict_future(self, model, scaled_data, scaler, days=7):
        model.eval()
        last_sequence = torch.FloatTensor(scaled_data[-60:]).unsqueeze(0).to(self.device)
        future_predictions = []

        for _ in range(days):
            with torch.no_grad():
                prediction = model(last_sequence).item()
            
            future_predictions.append(prediction)
            
            new_row = np.zeros((1, scaled_data.shape[1]))
            new_row[0, 3] = prediction  # Set closing price
            last_sequence = torch.cat((last_sequence[:, 1:, :], torch.FloatTensor(new_row).unsqueeze(1).to(self.device)), dim=1)

        future_predictions = np.array(future_predictions).reshape(-1, 1)
        return future_predictions

    # Get financial data from Yahoo Finance
    def get_financial_data(self, symbol):
        stock = yf.Ticker(symbol)
        
        financial_data = {
            'Market Cap': stock.info.get('marketCap', 'N/A'),
            'Forward P/E': stock.info.get('forwardPE', 'N/A'),
            'Trailing P/E': stock.info.get('trailingPE', 'N/A'),
            'Dividend Yield': stock.info.get('dividendYield', 'N/A'),
            'Beta': stock.info.get('beta', 'N/A'),
            '52 Week High': stock.info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': stock.info.get('fiftyTwoWeekLow', 'N/A'),
            'EPS': stock.info.get('trailingEps', 'N/A'),
            'Revenue': stock.info.get('totalRevenue', 'N/A'),
            'Profit Margin': stock.info.get('profitMargins', 'N/A')
        }
        
        # Ensure all values are JSON serializable
        for key, value in financial_data.items():
            if isinstance(value, np.float32):
                financial_data[key] = float(value)
            elif value is None or value == 'N/A':
                financial_data[key] = 'N/A'
            elif isinstance(value, (int, float)):
                financial_data[key] = round(value, 4)
        
        return financial_data

    # Save results to Google Sheets
    def save_to_google_sheets(self, results):
        try:
            spreadsheet = client.open_by_key(SPREADSHEET_ID)
            
            # Create or update worksheets for each model and combined results
            worksheets = {
                'COM': f"{self.symbol}_COM",
                'LSTM': f"{self.symbol}_LSTM",
                'ARIMA': f"{self.symbol}_ARIMA",
                'RF': f"{self.symbol}_RF",
                'TA': f"{self.symbol}_TA"
            }
            
            for model, sheet_name in worksheets.items():
                try:
                    worksheet = spreadsheet.worksheet(sheet_name)
                except gspread.WorksheetNotFound:
                    worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="1000", cols="20")
                
                # Prepare data based on model type
                if model == 'COM':
                    data = self.prepare_combined_data(results)
                else:
                    model_results = next((r for r in results['individual_models'] if r['model'] == model), None)
                    if model_results:
                        data = self.prepare_model_data(model_results)
                    else:
                        continue
                
                # Update worksheet
                worksheet.clear()
                worksheet.update('A1', data)

            self.update_signal.emit(f"Data saved to Google Sheets for stock symbol: {self.symbol}")
        except Exception as e:
            logging.error(f"Error saving to Google Sheets: {str(e)}")
            logging.error(traceback.format_exc())
            self.update_signal.emit(f"Error saving to Google Sheets: {str(e)}")

    # Prepare combined results data for Google Sheets
    def prepare_combined_data(self, results):
        data = [
            ['Date', 'Actual Price', 'Predicted Price'],
            *[[date.strftime('%Y-%m-%d'), actual, pred] for date, actual, pred in zip(results['dates'], results['actual'], results['predictions'])],
            ['Future Predictions', '', ''],
            *[[date.strftime('%Y-%m-%d'), pred, ''] for date, pred in zip(results['future_dates'], results['future_predictions'])],
            ['', '', ''],
            ['Metric', 'Value', ''],
            ['MSE', results.get('mse', 'N/A'), ''],
            ['RMSE', results.get('rmse', 'N/A'), ''],
            ['MAE', results.get('mae', 'N/A'), ''],
            ['MAPE', results.get('mape', 'N/A'), ''],
            ['Standard Deviation', results.get('std_dev', 'N/A'), ''],
            ['Model Type', 'Ensemble (LSTM, ARIMA, Random Forest, Technical Analysis)', '']
        ]
        return data

    # Prepare individual model results data for Google Sheets
    def prepare_model_data(self, model_results):
        data = [
            ['Date', 'Actual Price', 'Predicted Price'],
            *[[date.strftime('%Y-%m-%d'), actual, pred] for date, actual, pred in zip(model_results['dates'], model_results['actual'], model_results['predictions'])],
            ['Future Predictions', '', ''],
            *[[date.strftime('%Y-%m-%d'), pred, ''] for date, pred in zip(model_results['future_dates'], model_results['future_predictions'])],
            ['', '', ''],
            ['Metric', 'Value', ''],
            ['MSE', model_results.get('mse', 'N/A'), ''],
            ['RMSE', model_results.get('rmse', 'N/A'), ''],
            ['MAE', model_results.get('mae', 'N/A'), ''],
            ['MAPE', model_results.get('mape', 'N/A'), ''],
            ['Standard Deviation', model_results.get('std_dev', 'N/A'), ''],
            ['Model Type', model_results['model'], '']
        ]
        return data

    # Run the LSTM model
    def run_lstm_model(self, processed_data, scaler, original_close):
        self.update_signal.emit("Running LSTM model...")
        sequence_length = 60
        X, y = self.prepare_sequences(processed_data, sequence_length)
        
        if len(X) < 100:
            self.update_signal.emit("Insufficient data for effective LSTM training and prediction.")
            return None
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        total_epochs = 5 * 50 + 100  # 5 CV folds * 50 epochs each + 100 epochs for final training
        current_epoch = 0
        
        for fold, (train_index, val_index) in enumerate(tscv.split(X), 1):
            if self.is_paused:
                self.update_signal.emit("LSTM prediction paused.")
                return None
            
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            input_dim = processed_data.shape[1]
            hidden_dim = 128  # Increased from 64
            num_layers = 3   # Increased from 2
            output_dim = 1
            
            model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout=0.3).to(self.device)
            
            train_losses, val_losses = self.train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001,
                                                        fold=fold, total_epochs=total_epochs, current_epoch=current_epoch)
            current_epoch += 50
            
            model.eval()
            with torch.no_grad():
                predictions = model(X_val.to(self.device)).cpu().numpy().flatten()
            
            actual = y_val.cpu().numpy().flatten()
            
            mape = mean_absolute_percentage_error(actual, predictions)
            cv_scores.append(mape)
        
        self.update_signal.emit(f"LSTM Cross-validation MAPE scores: {cv_scores}")
        self.update_signal.emit(f"LSTM Average MAPE: {np.mean(cv_scores):.4f}")
        
        # Final model training on all data
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        
        final_model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout=0.3).to(self.device)
        train_losses, _ = self.train_model(final_model, train_loader, None, num_epochs=100, learning_rate=0.001,
                                        fold="Final", total_epochs=total_epochs, current_epoch=current_epoch)
        
        # Generate predictions using the final model
        final_model.eval()
        with torch.no_grad():
            predictions = final_model(X.to(self.device)).cpu().numpy().flatten()
        
        actual = y.cpu().numpy().flatten()
        
        dates = original_close.index[-len(predictions):]
        
        future_predictions = self.predict_future(final_model, processed_data, scaler, days=self.get_prediction_days())
        future_dates = [original_close.index[-1] + timedelta(days=i+1) for i in range(self.get_prediction_days())]
        
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predictions))
        mape = mean_absolute_percentage_error(actual, predictions)
        std_dev = np.std(actual - predictions)
        
        self.update_signal.emit("LSTM model training and prediction completed.")
        
        # Use original closing prices
        actual_prices = original_close[-len(predictions):].values
        all_zeros = np.zeros((predictions.shape[0], scaler.n_features_in_))
        all_zeros[:, 3] = predictions  # 3 is the index for 'Close' column
        predicted_prices = scaler.inverse_transform(all_zeros)[:, 3]

        future_zeros = np.zeros((future_predictions.shape[0], scaler.n_features_in_))
        future_zeros[:, 3] = future_predictions.flatten()
        future_prices = scaler.inverse_transform(future_zeros)[:, 3]

        return {
            'model': 'LSTM',
            'dates': dates.tolist(),
            'actual': actual_prices.tolist(),
            'predictions': predicted_prices.tolist(),
            'future_dates': future_dates,
            'future_predictions': future_prices.tolist(),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'std_dev': float(std_dev),
            'train_losses': [float(loss) for loss in train_losses]
        }

    # Run the ARIMA model
    def run_arima_model(self, original_close):
        self.update_signal.emit("Running ARIMA model...")
        
        # Convert index to datetime if it's not already
        if not isinstance(original_close.index, pd.DatetimeIndex):
            original_close.index = pd.to_datetime(original_close.index)
        
        # Resample to ensure consistent frequency
        original_close = original_close.resample('D').last().ffill()
        
        # Fit ARIMA model
        model = ARIMA(original_close, order=(5,1,0), freq='D')
        model_fit = model.fit()
        
        # Make predictions
        predictions = model_fit.predict(start=0, end=len(original_close)-1)
        future_predictions = model_fit.forecast(steps=self.get_prediction_days())
        
        # Calculate metrics
        mse = mean_squared_error(original_close, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(original_close - predictions))
        mape = mean_absolute_percentage_error(original_close, predictions)
        std_dev = np.std(original_close - predictions)
        
        self.update_signal.emit("ARIMA model prediction completed.")
        
        return {
            'model': 'ARIMA',
            'dates': original_close.index.tolist(),
            'actual': original_close.tolist(),
            'predictions': predictions.tolist(),
            'future_dates': [original_close.index[-1] + timedelta(days=i+1) for i in range(self.get_prediction_days())],
            'future_predictions': future_predictions.tolist(),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'std_dev': float(std_dev)
        }

    # Run the Random Forest model
    def run_random_forest_model(self, processed_data, scaler, original_close):
        self.update_signal.emit("Running Random Forest model...")
        
        # Prepare data
        X = processed_data[:-1]
        y = original_close[1:].values
        
        # Ensure X and y have the same number of samples
        min_length = min(len(X), len(y))
        X = X[:min_length]
        y = y[:min_length]
        
        # Train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        predictions = rf_model.predict(X_test)
        future_data = processed_data[-self.get_prediction_days():]
        future_predictions = rf_model.predict(future_data)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - predictions))
        mape = mean_absolute_percentage_error(y_test, predictions)
        std_dev = np.std(y_test - predictions)
        
        self.update_signal.emit("Random Forest model prediction completed.")
        
        return {
            'model': 'Random Forest',
            'dates': original_close.index[train_size+1:].tolist(),
            'actual': y_test.tolist(),
            'predictions': predictions.tolist(),
            'future_dates': [original_close.index[-1] + timedelta(days=i+1) for i in range(self.get_prediction_days())],
            'future_predictions': future_predictions.tolist(),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'std_dev': float(std_dev)
        }

    # Run Technical Analysis
    def run_technical_analysis(self, stock_data):
        self.update_signal.emit("Running Technical Analysis...")
        
        # Calculate technical indicators
        stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['RSI'] = ta.momentum.rsi(stock_data['Close'], window=14)
        
        # Generate buy/sell signals
        stock_data['Signal'] = 0
        stock_data.loc[(stock_data['SMA_20'] > stock_data['SMA_50']) & (stock_data['RSI'] < 30), 'Signal'] = 1  # Buy signal
        stock_data.loc[(stock_data['SMA_20'] < stock_data['SMA_50']) & (stock_data['RSI'] > 70), 'Signal'] = -1  # Sell signal
        
        # Calculate returns
        stock_data['Returns'] = stock_data['Close'].pct_change()
        stock_data['Strategy_Returns'] = stock_data['Signal'].shift(1) * stock_data['Returns']
        
        # Calculate metrics
        cumulative_returns = (1 + stock_data['Returns']).cumprod()
        cumulative_strategy_returns = (1 + stock_data['Strategy_Returns']).cumprod()
        
        # Safely calculate Sharpe ratio
        strategy_returns_mean = stock_data['Strategy_Returns'].mean()
        strategy_returns_std = stock_data['Strategy_Returns'].std()
        if strategy_returns_std != 0:
            sharpe_ratio = np.sqrt(252) * strategy_returns_mean / strategy_returns_std
        else:
            sharpe_ratio = 0
        
        self.update_signal.emit("Technical Analysis completed.")
        
        return {
            'model': 'Technical Analysis',
            'dates': stock_data.index.tolist(),
            'actual': stock_data['Close'].tolist(),
            'predictions': stock_data['Close'].tolist(),  # Using actual prices as "predictions" for consistency
            'future_dates': [stock_data.index[-1] + timedelta(days=i+1) for i in range(self.get_prediction_days())],
            'future_predictions': [stock_data['Close'].iloc[-1]] * self.get_prediction_days(),  # Placeholder for future predictions
            'cumulative_returns': cumulative_returns.tolist(),
            'cumulative_strategy_returns': cumulative_strategy_returns.tolist(),
            'sharpe_ratio': float(sharpe_ratio),
            'mse': 0,  # Not applicable for this method
            'rmse': 0,  # Not applicable for this method
            'mae': 0,  # Not applicable for this method
            'mape': 0,  # Not applicable for this method
            'std_dev': 0  # Not applicable for this method
        }

    # Combine results from all models
    def combine_results(self, results, stock_data):
        self.update_signal.emit("Combining results from all models...")
        
        # Find the model with the shortest date range
        shortest_model = min(results, key=lambda x: len(x['dates']))
        common_dates = shortest_model['dates']
    
        # Initialize combined results
        combined_results = {
            'dates': common_dates,
            'actual': [],
            'predictions': [],
            'future_dates': results[0]['future_dates'],
            'future_predictions': [],
            'mse': 0,
            'rmse': 0,
            'mae': 0,
            'mape': 0,
            'std_dev': 0,
            'individual_models': results
        }
    
        # Combine predictions
        for date in common_dates:
            actual_values = []
            predicted_values = []
            for r in results:
                if date in r['dates']:
                    index = r['dates'].index(date)
                    if index < len(r['actual']) and index < len(r['predictions']):
                        actual_values.append(r['actual'][index])
                        predicted_values.append(r['predictions'][index])
            
            if actual_values and predicted_values:
                combined_results['actual'].append(np.mean(actual_values))
                combined_results['predictions'].append(np.mean(predicted_values))
    
        # Combine future predictions
        for date in combined_results['future_dates']:
            future_predictions = []
            for r in results:
                if date in r['future_dates']:
                    index = r['future_dates'].index(date)
                    if index < len(r['future_predictions']):
                        future_predictions.append(r['future_predictions'][index])
            if future_predictions:
                combined_results['future_predictions'].append(np.mean(future_predictions))
    
        # Calculate combined metrics
        if combined_results['actual'] and combined_results['predictions']:
            combined_results['mse'] = mean_squared_error(combined_results['actual'], combined_results['predictions'])
            combined_results['rmse'] = np.sqrt(combined_results['mse'])
            combined_results['mae'] = np.mean(np.abs(np.array(combined_results['actual']) - np.array(combined_results['predictions'])))
            combined_results['mape'] = mean_absolute_percentage_error(combined_results['actual'], combined_results['predictions'])
            combined_results['std_dev'] = np.std(np.array(combined_results['actual']) - np.array(combined_results['predictions']))
    
        self.update_signal.emit("Results combination completed.")
        return combined_results

    # Get prediction days based on selected period
    def get_prediction_days(self):
        prediction_periods = {
            'One Day': 1,
            'One Week': 7,
            'One Month': 30,
            'Three Months': 90,
            'Six Months': 180,
            'Nine Months': 270,
            'One Year': 365
        }
        return prediction_periods[self.prediction_period]

# Main window class for the PyQt application
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Prediction Application by WeiEn Weng")
        self.setGeometry(100, 100, 1200, 800)

        self.symbol = ""
        self.signal_emitter = SignalEmitter()
        self.signal_emitter.update_chart.connect(self.update_chart)
        self.signal_emitter.update_eta.connect(self.update_eta)
        self.signal_emitter.show_loading.connect(self.show_loading_indicator)

        layout = QVBoxLayout()

        # Search bar and suggestions
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Enter stock symbol...")
        self.search_bar.textChanged.connect(self.update_suggestions)
        self.search_bar.returnPressed.connect(self.search_stock)
        search_layout.addWidget(self.search_bar)

        self.suggestion_list = QListWidget()
        self.suggestion_list.itemClicked.connect(self.select_suggestion)
        search_layout.addWidget(self.suggestion_list)

        layout.addLayout(search_layout)

        # Date range selector
        self.start_date = QDateEdit()
        self.end_date = QDateEdit()
        self.reset_date_range()
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Start Date:"))
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(QLabel("End Date:"))
        date_layout.addWidget(self.end_date)
        layout.addLayout(date_layout)

        # Prediction period selector
        prediction_layout = QHBoxLayout()
        prediction_layout.addWidget(QLabel("Prediction Period:"))
        self.prediction_period = QComboBox()
        self.prediction_period.addItems(['One Day', 'One Week', 'One Month', 'Three Months', 'Six Months', 'Nine Months', 'One Year'])
        self.prediction_period.setCurrentText('One Month')  # Set default to One Month
        prediction_layout.addWidget(self.prediction_period)
        layout.addLayout(prediction_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Prediction")
        self.start_button.clicked.connect(self.start_prediction)
        button_layout.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        button_layout.addWidget(self.pause_button)

        self.recalculate_button = QPushButton("Recalculate All")
        self.recalculate_button.clicked.connect(self.recalculate_all)
        self.recalculate_button.setEnabled(False)
        button_layout.addWidget(self.recalculate_button)

        self.import_button = QPushButton("Import Existing Data")
        self.import_button.clicked.connect(self.import_existing_data)
        self.import_button.setEnabled(False)
        button_layout.addWidget(self.import_button)

        layout.addLayout(button_layout)

        # Progress bar and ETA
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        self.eta_label = QLabel("ETA: Calculating...")
        progress_layout.addWidget(self.eta_label)
        layout.addLayout(progress_layout)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Chart tabs
        self.chart_tabs = {}
        for model in ['Combined', 'LSTM', 'ARIMA', 'Random Forest', 'Technical Analysis']:
            tab = QWidget()
            tab_layout = QVBoxLayout()
            figure = Figure(figsize=(12, 6), dpi=100)
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas, self)
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas)
            
            # Add accuracy label and author name
            labels_layout = QHBoxLayout()
            accuracy_label = QLabel(f"{model} Accuracy: N/A")
            author_label = QLabel("Made by: WeiEn Weng")
            labels_layout.addWidget(accuracy_label)
            labels_layout.addStretch()  # This will push the author label to the right
            labels_layout.addWidget(author_label)
            tab_layout.addLayout(labels_layout)
            
            # Add export button for each chart
            export_button = QPushButton(f"Export {model} Chart")
            export_button.clicked.connect(lambda checked, m=model: self.export_chart(m))
            tab_layout.addWidget(export_button)
            
            # Add loading indicator
            loading_label = QLabel()
            loading_movie = QMovie("loading.gif")  # Make sure to have a loading.gif in your project directory
            loading_label.setMovie(loading_movie)
            loading_label.hide()
            tab_layout.addWidget(loading_label)
            
            tab.setLayout(tab_layout)
            self.tab_widget.addTab(tab, f"{model} Chart")
            self.chart_tabs[model] = {
                'figure': figure,
                'canvas': canvas,
                'accuracy_label': accuracy_label,
                'author_label': author_label,
                'loading_label': loading_label,
                'loading_movie': loading_movie
            }

        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout()
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)
        self.metrics_tab.setLayout(metrics_layout)
        self.tab_widget.addTab(self.metrics_tab, "Metrics")

        self.data_tab = QWidget()
        data_layout = QVBoxLayout()
        self.data_table = QTableWidget()
        data_layout.addWidget(self.data_table)
        self.data_tab.setLayout(data_layout)
        self.tab_widget.addTab(self.data_tab, "Data")

        self.financial_tab = QWidget()
        financial_layout = QVBoxLayout()
        self.financial_table = QTableWidget()
        financial_layout.addWidget(self.financial_table)
        self.financial_tab.setLayout(financial_layout)
        self.tab_widget.addTab(self.financial_tab, "Financial Data")

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.prediction_thread = None
        self.results = None

        # Top 100 market cap companies (as of a recent date)
        self.top_100_companies = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "LLY",
            "JPM", "XOM", "JNJ", "V", "PG", "MA", "HD", "CVX", "AVGO", "MRK",
            "ABBV", "PEP", "KO", "COST", "ADBE", "WMT", "BAC", "MCD", "CRM", "ACN",
            "TMO", "CSCO", "ABT", "DHR", "LIN", "CMCSA", "PFE", "NKE", "ORCL", "NFLX",
            "TXN", "AMD", "UNP", "WFC", "PM", "INTC", "VZ", "DIS", "COP", "CAT",
            "NEE", "INTU", "RTX", "HON", "QCOM", "UPS", "BMY", "LOW", "MS", "BA",
            "AMGN", "SPGI", "DE", "GE", "AMAT", "BLK", "SYK", "IBM", "AXP", "ISRG",
            "ELV", "ADI", "PLD", "SBUX", "GILD", "CVS", "GS", "MDLZ", "TJX", "SCHW",
            "LMT", "MMC", "C", "BDX", "AMT", "MO", "TMUS", "CB", "PGR", "EOG",
            "REGN", "DUK", "SO", "CI", "SLB", "VRTX", "ZTS", "AON", "BSX", "NOW", "2330.TW"
        ]

    # Reset the date range to default values
    def reset_date_range(self):
        self.start_date.setDate(QDate.currentDate().addYears(-1))
        self.end_date.setDate(QDate.currentDate())
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        self.end_date.setDisplayFormat("yyyy-MM-dd")

    # Update suggestions based on the text input
    def update_suggestions(self, text):
        if len(text) < 1:
            self.suggestion_list.clear()
            return

        suggestions = [company for company in self.top_100_companies if text.upper() in company]
        self.suggestion_list.clear()
        self.suggestion_list.addItems(suggestions[:5])  # Show top 5 suggestions

    # Select suggestion from the list
    def select_suggestion(self, item):
        self.search_bar.setText(item.text())
        self.suggestion_list.clear()
        self.search_stock()

    # Search stock based on the entered symbol
    def search_stock(self):
        self.symbol = self.search_bar.text().upper()
        if not self.symbol:
            return
    
        # Get full date range for the stock
        stock = yf.Ticker(self.symbol)
        history = stock.history(period="max")
        if not history.empty:
            self.start_date.setDate(history.index[0].date())
            self.end_date.setDate(history.index[-1].date())
            self.update_status(f"Available date range for {self.symbol}: {history.index[0].date()} to {history.index[-1].date()}")
        else:
            self.update_status(f"Unable to retrieve data for symbol: {self.symbol}")
            return
    
        # Check if data exists in Google Sheets
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()
        sheet_data = self.load_from_google_sheets(start_date, end_date)
        if sheet_data:
            self.import_button.setEnabled(True)
            self.update_status(f"Existing data found for {self.symbol} within the selected date range. You can import it or start a new prediction.")
        else:
            self.import_button.setEnabled(False)
            self.update_status(f"No existing data found for {self.symbol} within the selected date range. You can start a new prediction.")
    
        self.start_button.setEnabled(True)

    # Start the prediction process
    def start_prediction(self):
        if not self.symbol:
            self.update_status("Please enter a stock symbol.")
            return
    
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()
        prediction_period = self.prediction_period.currentText()
        
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.recalculate_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.status_text.clear()
        self.progress_bar.setValue(0)
        self.eta_label.setText("ETA: Calculating...")
        
        self.prediction_thread = StockPredictionThread(self, start_date, end_date, prediction_period)
        self.prediction_thread.update_signal.connect(self.update_status)
        self.prediction_thread.progress_signal.connect(self.update_progress)
        self.prediction_thread.finished_signal.connect(self.show_results)
        self.prediction_thread.start()

    # Toggle pause/resume of the prediction process
    def toggle_pause(self):
        if self.prediction_thread and self.prediction_thread.isRunning():
            if self.prediction_thread.is_paused:
                self.prediction_thread.is_paused = False
                self.pause_button.setText("Pause")
                self.update_status("Prediction resumed.")
            else:
                self.prediction_thread.is_paused = True
                self.pause_button.setText("Resume")
                self.update_status("Prediction paused.")

    # Recalculate all data
    def recalculate_all(self):
        self.update_status("Starting recalculation of all data...")
        self.recalculate_button.setEnabled(False)
        self.start_button.setEnabled(False)

        # Get all worksheet names (stock symbols) from Google Sheets
        try:
            spreadsheet = client.open_by_key(SPREADSHEET_ID)
            worksheets = spreadsheet.worksheets()
            symbols = list(set([ws.title.split('_')[0] for ws in worksheets]))  # Get unique symbols
        except Exception as e:
            logging.error(f"Error fetching worksheet names: {str(e)}")
            self.update_status(f"Error fetching worksheet names: {str(e)}")
            return

        for symbol in symbols:
            self.update_status(f"Recalculating for {symbol}...")
            
            # Get full date range for the stock
            stock = yf.Ticker(symbol)
            history = stock.history(period="max")
            if not history.empty:
                start_date = history.index[0].date()
                end_date = history.index[-1].date()
            else:
                self.update_status(f"Unable to retrieve data for {symbol}. Skipping...")
                continue
            
            prediction_period = self.prediction_period.currentText()
            self.prediction_thread = StockPredictionThread(self, start_date, end_date, prediction_period)
            self.prediction_thread.update_signal.connect(self.update_status)
            self.prediction_thread.progress_signal.connect(self.update_progress)
            self.prediction_thread.finished_signal.connect(self.show_results)
            self.prediction_thread.run()  # Run synchronously to avoid overlapping calculations

        self.update_status("Recalculation of all data completed.")
        self.recalculate_button.setEnabled(True)
        self.start_button.setEnabled(True)

    # Import existing data from Google Sheets
    def import_existing_data(self):
        if not self.symbol:
            self.update_status("Please enter a stock symbol.")
            return
    
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()
        sheet_data = self.load_from_google_sheets(start_date, end_date)
        if sheet_data:
            self.show_results(sheet_data)
            self.update_status(f"Existing data for {self.symbol} imported successfully.")
        else:
            self.update_status(f"No existing data found for {self.symbol}.")

    # Load data from Google Sheets
    def load_from_google_sheets(self, start_date, end_date):
        try:
            spreadsheet = client.open_by_key(SPREADSHEET_ID)
            
            results = {}
            worksheets = {
                'COM': f"{self.symbol}_COM",
                'LSTM': f"{self.symbol}_LSTM",
                'ARIMA': f"{self.symbol}_ARIMA",
                'RF': f"{self.symbol}_RF",
                'TA': f"{self.symbol}_TA"
            }
            
            for model, sheet_name in worksheets.items():
                try:
                    worksheet = spreadsheet.worksheet(sheet_name)
                    all_data = worksheet.get_all_values()
                    parsed_data = self.parse_sheet_data(all_data)
                    
                    # Filter data based on date range
                    filtered_data = {
                        'dates': [],
                        'actual': [],
                        'predictions': [],
                        'future_dates': parsed_data['future_dates'],
                        'future_predictions': parsed_data['future_predictions']
                    }
                    
                    for date, actual, pred in zip(parsed_data['dates'], parsed_data['actual'], parsed_data['predictions']):
                        if start_date <= date <= end_date:
                            filtered_data['dates'].append(date)
                            filtered_data['actual'].append(actual)
                            filtered_data['predictions'].append(pred)
                    
                    filtered_data.update({k: v for k, v in parsed_data.items() if k not in filtered_data})
                    
                    if filtered_data['dates']:  # Only add if there's data within the date range
                        results[model] = filtered_data
                except gspread.WorksheetNotFound:
                    continue
            
            if not results:
                return None
            
            # Combine all model results
            combined_results = results['COM']
            combined_results['individual_models'] = [results[model] for model in ['LSTM', 'ARIMA', 'RF', 'TA'] if model in results]
            
            return combined_results
            
        except Exception as e:
            logging.error(f"Error loading data from Google Sheets: {str(e)}")
            logging.error(traceback.format_exc())
            self.update_status(f"Error loading data from Google Sheets: {str(e)}")
            return None

    # Parse sheet data from Google Sheets
    def parse_sheet_data(self, all_data):
        dates = []
        actual = []
        predictions = []
        future_dates = []
        future_predictions = []
        metrics = {}

        future_start = -1
        metrics_start = -1

        for i, row in enumerate(all_data[1:]):  # Skip header row
            if not row:  # Skip empty rows
                continue
            if row[0] == 'Future Predictions':
                future_start = i + 1
                break
            if len(row) >= 3:  # Ensure row has at least 3 elements
                dates.append(datetime.strptime(row[0], '%Y-%m-%d').date())
                actual.append(float(row[1]))
                predictions.append(float(row[2]))

        if future_start != -1:
            for row in all_data[future_start + 1:]:
                if not row:  # Skip empty rows
                    continue
                if row[0] == '':
                    break
                if len(row) >= 2:  # Ensure row has at least 2 elements
                    future_dates.append(datetime.strptime(row[0], '%Y-%m-%d').date())
                    future_predictions.append(float(row[1]))

        metrics_start = next((i for i, row in enumerate(all_data) if row and row[0] == 'Metric'), -1)
        if metrics_start != -1:
            for row in all_data[metrics_start + 1:]:
                if not row or row[0] == '':
                    break
                if len(row) >= 2:  # Ensure row has at least 2 elements
                    metrics[row[0]] = float(row[1]) if row[0] != 'Model Type' else row[1]

        return {
            'dates': dates,
            'actual': actual,
            'predictions': predictions,
            'future_dates': future_dates,
            'future_predictions': future_predictions,
            'mse': metrics.get('MSE'),
            'rmse': metrics.get('RMSE'),
            'mae': metrics.get('MAE'),
            'mape': metrics.get('MAPE'),
            'std_dev': metrics.get('Standard Deviation'),
            'model': metrics.get('Model Type')
        }

    # Update status text
    def update_status(self, message):
        self.status_text.append(message)

    # Update progress bar
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    # Update ETA label
    def update_eta(self, eta):
        self.eta_label.setText(eta)

    # Show/hide loading indicator
    def show_loading_indicator(self, show):
        for chart_tab in self.chart_tabs.values():
            if show:
                chart_tab['loading_movie'].start()
                chart_tab['loading_label'].show()
            else:
                chart_tab['loading_movie'].stop()
                chart_tab['loading_label'].hide()

    # Update chart with results
    def update_chart(self, results):
        try:
            self.show_loading_indicator(True)
            # Update combined chart
            self.update_single_chart('Combined', results)

            # Update individual model charts
            for model_results in results['individual_models']:
                self.update_single_chart(model_results['model'], model_results)

        except Exception as e:
            logging.error(f"Error updating charts: {str(e)}")
            logging.error(traceback.format_exc())
            self.update_status(f"Error updating charts: {str(e)}")
        finally:
            self.show_loading_indicator(False)

    # Update a single chart
    def update_single_chart(self, model, data):
        fig = self.chart_tabs[model]['figure']
        canvas = self.chart_tabs[model]['canvas']
        accuracy_label = self.chart_tabs[model]['accuracy_label']
        author_label = self.chart_tabs[model]['author_label']
        
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Ensure that dates, actual, and predictions have the same length
        min_length = min(len(data['dates']), len(data['actual']), len(data['predictions']))
        dates = data['dates'][:min_length]
        actual = data['actual'][:min_length]
        predictions = data['predictions'][:min_length]
        
        ax.plot(dates, actual, label='Actual Price', color='blue')
        ax.plot(dates, predictions, label=f'{model} Prediction', color='red')
        ax.plot(data['future_dates'], data['future_predictions'], label='Future Prediction', color='green', linestyle='--')
        ax.set_title(f'{model} Stock Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        canvas.draw()
    
        # Update accuracy label
        accuracy = 100 * (1 - data.get('mape', 0))
        accuracy_label.setText(f"{model} Accuracy: {accuracy:.2f}%")
        
        # Update author label
        author_label.setText("WeiEn")

    # Show results in the UI
    def show_results(self, results):
        try:
            self.results = results

            # Update metrics
            self.update_metrics(results)

            # Update data table
            self.update_data_table(results)

            # Update financial data table
            self.update_financial_table(results)

            # Update charts
            self.update_chart(results)

        except Exception as e:
            logging.error(f"Error displaying results: {str(e)}")
            logging.error(traceback.format_exc())
            self.update_status(f"Error displaying results: {str(e)}")

        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.recalculate_button.setEnabled(True)
    
    # Update metrics tab
    def update_metrics(self, results):
        self.metrics_text.clear()
        self.metrics_text.append("Combined Model Metrics:")
        self.metrics_text.append(f"Mean Squared Error (MSE): {results.get('mse', 'N/A')}")
        self.metrics_text.append(f"Root Mean Squared Error (RMSE): {results.get('rmse', 'N/A')}")
        self.metrics_text.append(f"Mean Absolute Error (MAE): {results.get('mae', 'N/A')}")
        self.metrics_text.append(f"Mean Absolute Percentage Error (MAPE): {results.get('mape', 'N/A')}")
        self.metrics_text.append(f"Standard Deviation of Error: {results.get('std_dev', 'N/A')}")
        self.metrics_text.append("\nIndividual Model Metrics:")
        for model_results in results.get('individual_models', []):
            model = model_results['model']
            self.metrics_text.append(f"\n{model}:")
            self.metrics_text.append(f"MSE: {model_results.get('mse', 'N/A')}")
            self.metrics_text.append(f"RMSE: {model_results.get('rmse', 'N/A')}")
            self.metrics_text.append(f"MAE: {model_results.get('mae', 'N/A')}")
            self.metrics_text.append(f"MAPE: {model_results.get('mape', 'N/A')}")
            self.metrics_text.append(f"Standard Deviation: {model_results.get('std_dev', 'N/A')}")
            if 'sharpe_ratio' in model_results:
                self.metrics_text.append(f"Sharpe Ratio: {model_results['sharpe_ratio']}")
            
    # Update data table tab
    def update_data_table(self, results):
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(['Date', 'Actual Price', 'Predicted Price'])
        self.data_table.setRowCount(len(results['dates']))
        
        for i, (date, actual, predicted) in enumerate(zip(results['dates'], results['actual'], results['predictions'])):
            self.data_table.setItem(i, 0, QTableWidgetItem(date.strftime('%Y-%m-%d') if isinstance(date, datetime) else str(date)))
            self.data_table.setItem(i, 1, QTableWidgetItem(f"{actual:.2f}"))
            self.data_table.setItem(i, 2, QTableWidgetItem(f"{predicted:.2f}"))

        self.data_table.resizeColumnsToContents()

    # Update financial data table tab
    def update_financial_table(self, results):
        if 'financial_data' in results:
            self.financial_table.setColumnCount(2)
            self.financial_table.setHorizontalHeaderLabels(['Metric', 'Value'])
            self.financial_table.setRowCount(len(results['financial_data']))

            for i, (metric, value) in enumerate(results['financial_data'].items()):
                self.financial_table.setItem(i, 0, QTableWidgetItem(metric))
                self.financial_table.setItem(i, 1, QTableWidgetItem(str(value)))

            self.financial_table.resizeColumnsToContents()

    # Export chart as PNG file
    def export_chart(self, model):
        if self.results is None:
            self.update_status(f"No data available to export for {model} chart. Please run a prediction first.")
            return

        try:
            # Create 'charts' directory if it doesn't exist
            if not os.path.exists('charts'):
                os.makedirs('charts')

            # Generate file name
            symbol = self.search_bar.text().upper()
            start_date = self.results['dates'][0].strftime('%y%m%d') if isinstance(self.results['dates'][0], datetime) else self.results['dates'][0]
            end_date = self.results['dates'][-1].strftime('%y%m%d') if isinstance(self.results['dates'][-1], datetime) else self.results['dates'][-1]
            current_time = datetime.now().strftime('%y%m%d_%H%M%S')
            file_name = f'charts/{model}_{symbol}_{current_time}_{start_date}to{end_date}.png'

            # Save the chart
            self.chart_tabs[model]['figure'].savefig(file_name)

            self.update_status(f"{model} chart exported to {file_name}")
        except Exception as e:
            logging.error(f"Error exporting {model} chart: {str(e)}")
            logging.error(traceback.format_exc())
            self.update_status(f"Error exporting {model} chart: {str(e)}")

# Main entry point for the application
if __name__ == "__main__":
    try:
        # Enable MPS (Metal Performance Shaders) for M1 Macs if available
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("MPS (Metal Performance Shaders) is available. Using MPS backend.")
        else:
            device = torch.device('cpu')
            print("MPS not available. Using CPU backend.")

        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Application runtime error: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"Application runtime error: {str(e)}")
