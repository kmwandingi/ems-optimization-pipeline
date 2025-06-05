import numpy as np
import pandas as pd
import logging
import datetime
from typing import Union

# [FC] -- Full updated PVAgent code with forced datetime conversion
class PVAgent:
    """
    Encapsulates all PV-related data loading and usage.
    Supports both measured PV profile data and PV forecast data.
    """
    def __init__(self, profile_data: pd.DataFrame = None, forecast_data: pd.DataFrame = None,
                 profile_cols: list = None, forecast_cols: list = None):
        import pandas as pd
        import numpy as np
        import logging

        # =========== Handle PV profile_data ===========
        if profile_data is not None:
            self.profile_data = profile_data.copy()
            if profile_cols is None:
                profile_cols = [c for c in self.profile_data.columns
                                if 'pv' in c.lower() and c != 'price_per_kwh']
            if len(profile_cols) == 0:
                logging.info("No PV profile columns found. Using all zeros for pv_profile.")
                self.pv_profile = np.zeros(len(self.profile_data))
            else:
                logging.info(f"PVAgent: Found profile columns: {profile_cols}")
                self.profile_data['pv_summed'] = self.profile_data[profile_cols].sum(axis=1)
                # Convert to negative since generation is represented as negative load.
                self.profile_data['pv_summed'] = -self.profile_data['pv_summed']
                # print(self.profile_data)
                self.pv_profile = self.profile_data['pv_summed'].values
        else:
            self.profile_data = None
            self.pv_profile = None

        # =========== Handle PV forecast_data ===========
        if forecast_data is not None:
            # Convert to pandas if needed (e.g., from polars)
            if not isinstance(forecast_data, pd.DataFrame):
                forecast_data = forecast_data.to_pandas()

            self.forecast_data = forecast_data.copy()

            # Make sure there's a utc_timestamp column
            # If it's in the index instead, reset it
            if 'utc_timestamp' not in self.forecast_data.columns:
                if self.forecast_data.index.name == 'utc_timestamp':
                    logging.warning("Resetting index to retrieve 'utc_timestamp'.")
                    self.forecast_data.reset_index(inplace=True)

            if 'utc_timestamp' not in self.forecast_data.columns:
                raise ValueError("No 'utc_timestamp' column found in forecast_data.")

            # Force-convert to datetime so we can use .dt
            self.forecast_data['utc_timestamp'] = pd.to_datetime(
                self.forecast_data['utc_timestamp'], errors='coerce', utc=True
            )

            # Identify forecast columns if none specified
            if forecast_cols is None:
                forecast_cols = [c for c in self.forecast_data.columns if 'solar' in c.lower()]
            if len(forecast_cols) == 0:
                logging.info("No PV forecast columns found. Using zero forecast.")
                self.forecast_data['pv_forecast_summed'] = 0.0
                self.forecast_col = 'pv_forecast_summed'
            else:
                self.forecast_col = forecast_cols[0]
                logging.info(f"PVAgent: Using forecast column: {self.forecast_col}")
                self.forecast_data['pv_forecast_summed'] = self.forecast_data[forecast_cols].sum(axis=1)

        else:
            self.forecast_data = None
            self.forecast_col = None

        # print("Unique forecast dates:", self.forecast_data['utc_timestamp'].dt.date.unique())

        # =========== Log summary stats ===========
        if self.pv_profile is not None:
            logging.info("PV Profile stats: mean={:.3f}, min={:.3f}, max={:.3f}".format(
                self.pv_profile.mean(), self.pv_profile.min(), self.pv_profile.max()))

        if self.forecast_data is not None:
            fvals = self.forecast_data['pv_forecast_summed'].values
            # Handle empty arrays safely
            if len(fvals) > 0:
                logging.info("PV Forecast stats: mean={:.3f}, min={:.3f}, max={:.3f}".format(
                    fvals.mean(), fvals.min(), fvals.max()))
            else:
                logging.info("PV Forecast array is empty. Using zeros for forecasts.")

    def get_pv_profile(self, indices: np.ndarray) -> np.ndarray:
        """Returns measured PV profile (negative sign for generation) at given indices."""
        import numpy as np
        if self.pv_profile is None:
            return np.zeros(len(indices))
        # Clip indices that are out of range
        indices = [i for i in indices if (0 <= i < len(self.pv_profile))]
        return self.pv_profile[indices]

    def get_hourly_forecast_pv(self, target_date):
        """
        Return a 24-hour forecast array for the given target date.
        Negative values indicate generation.
        """
        import pandas as pd
        import numpy as np
        import logging

        # Convert string dates if needed
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date, errors='coerce').date()

        # If no forecast data available, return zeros
        if (self.forecast_data is None) or (self.forecast_col is None):
            logging.warning("No forecast data. Returning zero forecast.")
            return np.zeros(24)

        # Filter out rows matching target_date in the forecast_data
        forecast_day = self.forecast_data[self.forecast_data['utc_timestamp'].dt.date == target_date]
        if forecast_day.empty:
            logging.warning(f"No forecast data for {target_date}. Returning zeros.")
            return np.zeros(24)

        # Group by hour and average
        forecast_day = forecast_day.copy()  # Avoid SettingWithCopyWarning
        forecast_day['hour'] = forecast_day['utc_timestamp'].dt.hour
        hourly_sums = forecast_day.groupby('hour')[self.forecast_col].mean() \
                                  .reindex(range(24), fill_value=0).values

        # Scale factor assumption: typical max ~32704 → scale to negative
        scale_factor = 1.0 / 32704.0
        hourly_scaled = -hourly_sums * scale_factor
        return hourly_scaled
    # 
    def compute_hourly_error_std(self, target_date=None):
        """
        Computes the standard deviation of the forecast error (measured minus forecast)
        for each hour (0-23) using historical data, with day-specific variations.
        
        Returns an array of 24 values.
        """
        import pandas as pd
        import numpy as np
    
        if self.profile_data is None or self.forecast_data is None:
            # If we do not have both sources, return zeros.
            return np.zeros(24)
    
        # Ensure both dataframes have datetime columns
        df_actual = self.profile_data.copy()
        df_forecast = self.forecast_data.copy()
                
        # Standard error calculation (unchanged)
        if 'utc_timestamp' not in df_actual.columns:
            df_actual = df_actual.reset_index()
        if 'utc_timestamp' not in df_forecast.columns:
            df_forecast = df_forecast.reset_index()
    
        df_actual['utc_timestamp'] = pd.to_datetime(df_actual['utc_timestamp'], utc=True)
        df_forecast['utc_timestamp'] = pd.to_datetime(df_forecast['utc_timestamp'], utc=True)
    
        df_actual['date_hour'] = df_actual['utc_timestamp'].dt.floor('h')
        df_forecast['date_hour'] = df_forecast['utc_timestamp'].dt.floor('h')
        
        # Add month for filtering
        df_actual['month'] = df_actual['utc_timestamp'].dt.month
        df_forecast['month'] = df_forecast['utc_timestamp'].dt.month
    
        # Filter for similar months
        if target_date is not None:
            if isinstance(target_date, str):
                target_date = pd.to_datetime(target_date).date()
                
            target_month = target_date.month
            target_day = target_date.day
            
            # Use month filtering (similar months)
            similar_months = [(target_month-1) % 12 or 12, target_month, (target_month+1) % 12 or 12]
            df_actual = df_actual[df_actual['month'].isin(similar_months)]
            df_forecast = df_forecast[df_forecast['month'].isin(similar_months)]
    
        # Join and calculate error
        df_joined = pd.merge(df_actual, df_forecast, on='date_hour', suffixes=('_actual', '_forecast'))
        scale_factor = 1.0 / 32704.0
        df_joined['error'] = df_joined['pv_summed'] - (df_joined['pv_forecast_summed'] * scale_factor)
        df_joined['hour'] = df_joined['utc_timestamp_actual'].dt.hour
        
        # Calculate standard deviation by hour
        error_std = df_joined.groupby('hour')['error'].std()
        
        # Create base hourly STD
        hourly_std = np.array([error_std.get(h, 0.001) for h in range(24)])
        
        # SOLUTION: Add day-specific variation using the day number as a seed
        if target_date:
            day_seed = target_date.day
            np.random.seed(day_seed)
            # Each day gets a unique but deterministic variation
            day_variation = np.random.uniform(0.7, 1.3, 24)  # 30% variation up/down
            hourly_std = hourly_std * day_variation
        
        # Add minimum uncertainty floor
        hourly_std = np.maximum(hourly_std, 0.001)
        
        # Stats for debugging
        # if target_date:
        #     # print(f"Stats for {target_date}: min={hourly_std.min():.4f}, max={hourly_std.max():.4f}, mean={hourly_std.mean():.4f}")
        
        return hourly_std
