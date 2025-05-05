import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ForecastEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        
    def prepare_data(self, df, target_column='revenue'):
        """Prepare data for forecasting"""
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Aggregate data by date if multiple products exist
        if 'product' in df.columns:
            df = df.groupby('date')[target_column].sum().reset_index()
        
        return df
    
    def moving_average(self, df, window=7, target_column='revenue'):
        """Calculate moving average forecast"""
        df = self.prepare_data(df)
        
        # Calculate historical moving average
        historical_ma = df[target_column].rolling(window=window).mean()
        
        # Calculate future moving average
        last_values = df[target_column].tail(window).values
        future_ma = []
        
        for _ in range(30):  # Forecast for 30 days
            next_value = np.mean(last_values)
            future_ma.append(next_value)
            # Update last_values using numpy array operations
            last_values = np.roll(last_values, -1)
            last_values[-1] = next_value
        
        return np.array(future_ma)
    
    def arima_forecast(self, df, target_column='revenue', forecast_periods=30):
        """Generate ARIMA forecast"""
        df = self.prepare_data(df)
        
        # Fit ARIMA model
        model = ARIMA(df[target_column], order=(5,1,0))
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=forecast_periods)
        
        # Store model for future use
        self.models['arima'] = model_fit
        
        return forecast
    
    def random_forest_forecast(self, df, target_column='revenue', forecast_periods=30):
        """Generate Random Forest forecast"""
        df = self.prepare_data(df)
        
        # Create features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day_of_month'] = df['date'].dt.day
        df['quarter'] = df['date'].dt.quarter
        
        # Add lag features
        for lag in [1, 7, 14]:
            df[f'lag_{lag}'] = df[target_column].shift(lag)
        
        # Drop NaN values
        df = df.dropna()
        
        # Prepare features and target
        features = ['day_of_week', 'month', 'year', 'day_of_month', 'quarter', 
                   'lag_1', 'lag_7', 'lag_14']
        X = df[features]
        y = df[target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Store model and scaler
        self.models['random_forest'] = model
        
        # Generate future dates
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                   periods=forecast_periods)
        
        # Prepare future features
        future_df = pd.DataFrame({
            'date': future_dates,
            'day_of_week': future_dates.dayofweek,
            'month': future_dates.month,
            'year': future_dates.year,
            'day_of_month': future_dates.day,
            'quarter': future_dates.quarter
        })
        
        # Initialize lag features with zeros
        for lag in [1, 7, 14]:
            future_df[f'lag_{lag}'] = 0
        
        # Add lag features for future predictions
        last_values = df[target_column].tail(14).values
        predictions = []
        
        for i in range(forecast_periods):
            # Update lag features
            if i == 0:
                future_df.loc[i, 'lag_1'] = last_values[-1]
                future_df.loc[i, 'lag_7'] = last_values[-7] if len(last_values) >= 7 else last_values[-1]
                future_df.loc[i, 'lag_14'] = last_values[-14] if len(last_values) >= 14 else last_values[-1]
            else:
                future_df.loc[i, 'lag_1'] = predictions[-1]
                future_df.loc[i, 'lag_7'] = predictions[-7] if len(predictions) >= 7 else predictions[-1]
                future_df.loc[i, 'lag_14'] = predictions[-14] if len(predictions) >= 14 else predictions[-1]
            
            # Scale features and predict
            future_X = future_df.loc[i, features].values.reshape(1, -1)
            future_X_scaled = self.scaler.transform(future_X)
            prediction = model.predict(future_X_scaled)[0]
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def generate_forecast(self, df, model_type='arima', forecast_periods=30, target_column='revenue'):
        """Generate forecast using specified model"""
        try:
            if model_type == 'arima':
                return self.arima_forecast(df, target_column, forecast_periods)
            elif model_type == 'random_forest':
                return self.random_forest_forecast(df, target_column, forecast_periods)
            elif model_type == 'moving_average':
                return self.moving_average(df, window=7, target_column=target_column)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            print(f"Error in {model_type} forecast: {str(e)}")
            # Fallback to moving average if other models fail
            return self.moving_average(df, window=7, target_column=target_column) 