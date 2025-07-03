from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64

# Define an entity for the device
device = Entity(
    name="device",
    join_keys=["building", "plain_device_type"],
)

# Create a dummy source
source = FileSource(
    name="device_usage_source",
    path="data/dummy_data.parquet",
    timestamp_field="date",
)

# Define feature view
device_usage_view = FeatureView(
    name="device_usage",
    entities=[device],
    ttl=timedelta(days=30),
    schema=[
        Feature(name="temperature", dtype=Float32),
        Feature(name="solar_radiation", dtype=Float32),
        Feature(name="pv_forecast", dtype=Float32),
        Feature(name="hour", dtype=Int64),
        Feature(name="day_of_week", dtype=Int64),
        Feature(name="is_weekend", dtype=Int64),
        Feature(name="rolling_24h_usage", dtype=Float32),
        Feature(name="temp_hour", dtype=Float32),
        Feature(name="rolling_7d_usage", dtype=Float32),
        Feature(name="rolling_3d_usage", dtype=Float32),
        Feature(name="rolling_14d_usage", dtype=Float32),
        Feature(name="lag_7d_usage", dtype=Float32),
        Feature(name="peak_hour_sin", dtype=Float32),
        Feature(name="peak_hour_cos", dtype=Float32),
        Feature(name="temp_month", dtype=Float32),
        Feature(name="dow_pv", dtype=Float32),
        Feature(name="is_holiday", dtype=Int64),
        Feature(name="day_cumulative_usage", dtype=Float32),
        Feature(name="prev_hour_on", dtype=Int64),
        Feature(name="hour_sin", dtype=Float32),
        Feature(name="hour_cos", dtype=Float32),
    ],
    source=source,
)
