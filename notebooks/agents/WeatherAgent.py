import pandas as pd
import numpy as np
import logging

class WeatherAgent:
    """
    Returns a dictionary {colName: 24-element array} for each day,
    for all weather columns in your loaded weather DataFrame.
    e.g. { "DE_temperature": [24 floats], "DE_radiation_direct_horizontal": [...], ... }
    """

    def __init__(self, weather_df: pd.DataFrame):
        # Convert polars => pandas if needed
        if not isinstance(weather_df, pd.DataFrame):
            weather_df = weather_df.to_pandas()
        self.weather_df = weather_df.copy()

        if 'utc_timestamp' not in self.weather_df.columns:
            raise ValueError("weather_df must have 'utc_timestamp' column.")

        # Convert timestamp to a standard form
        self.weather_df['utc_timestamp'] = pd.to_datetime(self.weather_df['utc_timestamp'], utc=True)
        self.weather_df['day'] = self.weather_df['utc_timestamp'].dt.date
        self.weather_df['hour'] = self.weather_df['utc_timestamp'].dt.hour

        # Identify all weather columns (besides the time fields)
        self.weather_cols = [c for c in self.weather_df.columns
                             if c not in ['utc_timestamp','day','hour']]

    def get_all_hourly_forecasts(self, target_date) -> dict:
        """
        Return a dict of { columnName: np.array of length=24 }
        corresponding to the weather for `target_date`.
        """
        day_mask = (self.weather_df['day'] == target_date)
        day_data = self.weather_df[day_mask].copy()
        if len(day_data) < 24:
            logging.warning(f"WeatherAgent: incomplete data ({len(day_data)} rows) for {target_date}")
        day_data.sort_values("hour", inplace=True)

        forecast_dict = {}
        # Restrict to genuinely numeric columns
        numeric_cols = [
            c for c in self.weather_df.columns
            if c not in ['utc_timestamp','day','hour']
               and pd.api.types.is_numeric_dtype(self.weather_df[c])
        ]

        for col in numeric_cols:
            arr = np.zeros(24, dtype=float)
            for h in range(24):
                row = day_data[day_data['hour'] == h]
                if not row.empty:
                    val = row[col].iloc[0]
                    # If the cell itself is an array/list, extract a scalar
                    if isinstance(val, (list, np.ndarray)):
                        # e.g. take the first element (or mean, whichever makes sense)
                        scalar = float(val[0])
                    else:
                        scalar = float(val)
                    arr[h] = scalar
                else:
                    arr[h] = arr[h-1] if h > 0 else 0.0
            forecast_dict[col] = arr

        return forecast_dict

