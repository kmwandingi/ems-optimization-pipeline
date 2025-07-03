import pandas as pd
file_path = 'D:/Kenneth - TU Eindhoven/Jads/Graduation Project 2024-2025/ems_project/ems-optimization-pipeline/notebooks/data/DE_KN_residential1_processed_data.parquet'
df = pd.read_parquet(file_path)
print(df['price_per_kwh'].head(50).tolist())