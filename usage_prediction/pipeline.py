from zenml import pipeline
from usage_prediction.steps import (
    data_prep_step,
    feature_engineering_step,
    train_daily_step,
    train_hourly_step,
    evaluation_step,
)

@pipeline
def device_usage_pipeline():
    daily_df, hourly_df = data_prep_step()
    X_day, y_day, X_hr  = feature_engineering_step(daily_df, hourly_df)
    daily_model         = train_daily_step(X_day, y_day)
    hourly_model        = train_hourly_step(X_hr)
    evaluation_step(daily_model, hourly_model, X_day, y_day, X_hr)
