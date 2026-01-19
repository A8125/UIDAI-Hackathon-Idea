"""
Prophet Forecasting Module
===========================
Implements time-series forecasting to predict Aadhaar saturation levels.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SaturationForecaster:
    """Forecasts Aadhaar saturation using Facebook Prophet."""
    
    def __init__(self):
        """Initialize forecaster."""
        self.models = {}
        self.forecasts = {}
        
    def prepare_prophet_data(self, timeseries_df, district_id):
        """
        Prepare data for Prophet model.
        
        Parameters:
        -----------
        timeseries_df : pd.DataFrame
            Time series data
        district_id : str
            District to prepare data for
            
        Returns:
        --------
        pd.DataFrame : Prophet-formatted data (ds, y columns)
        """
        # Filter to district
        district_data = timeseries_df[timeseries_df['district_id'] == district_id].copy()
        
        # Sort by date
        district_data = district_data.sort_values('date')
        
        # Prophet requires 'ds' and 'y' columns
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(district_data['date']).dt.tz_localize(None),
            'y': district_data['saturation_level'].astype(float)
        })
        
        return prophet_df
    
    def fit_district_model(self, prophet_df, district_id):
        """
        Fit Prophet model for a single district.
        
        Parameters:
        -----------
        prophet_df : pd.DataFrame
            Prophet-formatted data
        district_id : str
            District identifier
            
        Returns:
        --------
        Prophet : Fitted model
        """
        # Add holiday/special event effects (enrollment cycles)
        holidays = pd.DataFrame({
            'holiday': 'enrollment_season',
            'ds': pd.to_datetime([
                '2023-04-15', '2023-05-15', '2024-04-15', '2024-05-15',
                '2025-04-15', '2025-05-15', '2026-04-15', '2026-05-15'
            ]),
            'lower_window': -15,
            'upper_window': 15
        })

        # Initialize Prophet with custom parameters
        model = Prophet(
            growth='logistic',  # Saturation has an upper bound
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=holidays,
            changepoint_prior_scale=0.05,  # Flexibility for trend changes
            seasonality_prior_scale=10.0
        )
        
        # For logistic growth, we need to specify cap (maximum saturation = 1.0)
        prophet_df['cap'] = 0.99  # Maximum saturation level
        prophet_df['floor'] = 0.0  # Minimum saturation level
        
        # Fit model
        model.fit(prophet_df)
        
        self.models[district_id] = model
        
        return model
    
    def forecast_district(self, district_id, periods=12, freq='M'):
        """
        Generate forecast for a district.
        
        Parameters:
        -----------
        district_id : str
            District identifier
        periods : int
            Number of periods to forecast
        freq : str
            Frequency ('M' for monthly, 'D' for daily)
            
        Returns:
        --------
        pd.DataFrame : Forecast with confidence intervals
        """
        if district_id not in self.models:
            raise ValueError(f"No model fitted for district {district_id}")
        
        model = self.models[district_id]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        future['cap'] = 0.99
        future['floor'] = 0.0
        
        # Generate forecast
        forecast = model.predict(future)
        
        self.forecasts[district_id] = forecast
        
        return forecast
    
    def batch_forecast(self, timeseries_df, district_ids=None, periods=12):
        """
        Generate forecasts for multiple districts.
        
        Parameters:
        -----------
        timeseries_df : pd.DataFrame
            Time series data
        district_ids : list
            List of district IDs to forecast (None = all)
        periods : int
            Forecast horizon in months
            
        Returns:
        --------
        dict : Dictionary of forecasts by district
        """
        if district_ids is None:
            district_ids = timeseries_df['district_id'].unique()
        
        print(f"Generating forecasts for {len(district_ids)} districts...")
        
        forecasts = {}
        
        for i, district_id in enumerate(district_ids):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(district_ids)} districts")
            
            try:
                # Prepare data
                prophet_df = self.prepare_prophet_data(timeseries_df, district_id)
                
                # Fit model
                self.fit_district_model(prophet_df, district_id)
                
                # Forecast
                forecast = self.forecast_district(district_id, periods=periods)
                
                forecasts[district_id] = forecast
                
            except Exception as e:
                print(f"  Warning: Failed to forecast {district_id}: {str(e)}")
                continue
        
        print(f"✓ Generated forecasts for {len(forecasts)} districts")
        
        return forecasts
    
    def identify_saturation_milestones(self, forecast_df, threshold=0.99):
        """
        Identify when a district will reach saturation threshold.
        
        Parameters:
        -----------
        forecast_df : pd.DataFrame
            Prophet forecast
        threshold : float
            Saturation threshold
            
        Returns:
        --------
        dict : Milestone information
        """
        # Find first date where predicted saturation exceeds threshold
        future_data = forecast_df[forecast_df['ds'] > datetime.now()]
        
        milestone_dates = future_data[future_data['yhat'] >= threshold]
        
        if len(milestone_dates) > 0:
            milestone_date = milestone_dates.iloc[0]['ds']
            days_until = (milestone_date - datetime.now()).days
            
            return {
                'will_reach': True,
                'milestone_date': milestone_date,
                'days_until': days_until,
                'months_until': round(days_until / 30, 1),
                'current_saturation': forecast_df[forecast_df['ds'] <= datetime.now()].iloc[-1]['yhat']
            }
        else:
            return {
                'will_reach': False,
                'milestone_date': None,
                'current_saturation': forecast_df[forecast_df['ds'] <= datetime.now()].iloc[-1]['yhat'],
                'projected_saturation': forecast_df.iloc[-1]['yhat']
            }
    
    def generate_saturation_report(self, timeseries_df, districts_df, sample_size=10):
        """
        Generate comprehensive saturation forecast report.
        
        Parameters:
        -----------
        timeseries_df : pd.DataFrame
            Time series data
        districts_df : pd.DataFrame
            District metadata
        sample_size : int
            Number of districts to analyze in detail
            
        Returns:
        --------
        pd.DataFrame : Saturation milestone summary
        """
        # Select sample districts (mix of different saturation levels)
        latest_saturation = timeseries_df.groupby('district_id')['saturation_level'].last().reset_index()
        latest_saturation = latest_saturation.sort_values('saturation_level')
        
        # Sample from different quartiles
        sample_districts = []
        quartiles = [0, 0.25, 0.5, 0.75]
        for q in quartiles:
            idx = int(len(latest_saturation) * q)
            if idx < len(latest_saturation):
                sample_districts.append(latest_saturation.iloc[idx]['district_id'])
        
        # Add some random districts
        remaining = sample_size - len(sample_districts)
        if remaining > 0:
            random_sample = latest_saturation.sample(min(remaining, len(latest_saturation)))['district_id'].tolist()
            sample_districts.extend(random_sample)
        
        sample_districts = list(set(sample_districts))[:sample_size]
        
        # Generate forecasts
        forecasts = self.batch_forecast(timeseries_df, sample_districts, periods=12)
        
        # Analyze milestones
        milestones = []
        
        for district_id, forecast_df in forecasts.items():
            milestone = self.identify_saturation_milestones(forecast_df, threshold=0.99)
            
            district_info = districts_df[districts_df['district_id'] == district_id].iloc[0]
            
            milestones.append({
                'district_id': district_id,
                'district_name': district_info['district_name'],
                'state': district_info['state'],
                'district_type': district_info['district_type'],
                'current_saturation': round(milestone['current_saturation'], 4),
                'will_reach_99pct': milestone['will_reach'],
                'months_until_saturation': milestone.get('months_until', 'Not in forecast period'),
                'projected_saturation': round(forecast_df.iloc[-1]['yhat'], 4)
            })
        
        return pd.DataFrame(milestones)
    
    def visualize_forecast(self, district_id, districts_df=None, output_path='output/forecast.png'):
        """
        Visualize forecast for a district.
        
        Parameters:
        -----------
        district_id : str
            District to visualize
        districts_df : pd.DataFrame
            District metadata (optional)
        output_path : str
            Path to save visualization
        """
        if district_id not in self.forecasts:
            raise ValueError(f"No forecast available for {district_id}")
        
        model = self.models[district_id]
        forecast = self.forecasts[district_id]
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Forecast with confidence intervals
        model.plot(forecast, ax=axes[0])
        axes[0].axhline(y=0.99, color='red', linestyle='--', linewidth=2, label='99% Saturation Target')
        axes[0].axvline(x=datetime.now(), color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Today')
        axes[0].set_xlabel('Date', fontsize=12)
        axes[0].set_ylabel('Saturation Level', fontsize=12)
        
        title = f'Aadhaar Saturation Forecast: {district_id}'
        if districts_df is not None:
            district_info = districts_df[districts_df['district_id'] == district_id]
            if len(district_info) > 0:
                title = f'Saturation Forecast: {district_info.iloc[0]["district_name"]}'
        
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.0])
        
        # Plot 2: Components
        model.plot_components(forecast, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Forecast visualization saved to {output_path}")


if __name__ == "__main__":
    print("Prophet Forecasting Module - Saturation Prediction")
    print("Use this module to predict when districts will reach Aadhaar saturation")
