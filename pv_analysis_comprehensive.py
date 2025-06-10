#!/usr/bin/env python3
"""
Comprehensive PV Analysis Script for Energy Management System

This script performs in-depth PV (solar) analysis over real historical data 
for buildings with PV systems, using existing Agent classes from the codebase.

Author: Post-doctoral researcher at DeepMind
Focus: Energy systems optimization and PV forecast uncertainty analysis
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add notebooks directory to path for agent imports
sys.path.append(str(Path(__file__).parent / "notebooks"))
sys.path.append(str(Path(__file__).parent / "scripts"))

# Configuration
building_ids = [
    "DE_KN_residential1",
    "DE_KN_residential2", 
    "DE_KN_residential3",
    "DE_KN_residential4",
    "DE_KN_residential5",
    "DE_KN_residential6",
    "DE_KN_industrial3"
]
min_days = 100  # number of days to analyze per building
output_dir = "./pv_analysis_plots"

# Imports from codebase
from common import get_con

# Configure plotting style
sns.set_theme(style="whitegrid")
plt.style.use("ggplot")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_table_exists(con, table_name):
    """Check that required table exists in DuckDB"""
    result = con.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_name = ?
    """, [table_name]).fetchall()
    
    if not result:
        raise ValueError(f"Required table {table_name} not found in DuckDB")

def fetch_full_days(con, building_id):
    """Fetch list of days with complete 24-hour data"""
    query = f"""
        SELECT DISTINCT DATE(utc_timestamp) AS day, COUNT(*) AS cnt
        FROM {building_id}_processed_data
        GROUP BY DATE(utc_timestamp)
        HAVING COUNT(*) = 24
        ORDER BY day
    """
    
    df = con.execute(query).df()
    all_days = df["day"].dt.strftime("%Y-%m-%d").tolist()
    
    if len(all_days) < min_days:
        raise ValueError(f"Not enough full 24-hour days for {building_id}: found {len(all_days)} days, need ≥ {min_days}")
    
    return all_days

def detect_pv_columns(con, building_id):
    """Detect PV forecast and actual generation columns"""
    # Get table schema
    schema_result = con.execute(f"PRAGMA table_info('{building_id}_processed_data')").fetchall()
    columns = [row[1] for row in schema_result]  # Column names are in index 1
    
    # Look for PV columns
    pv_forecast_col = None
    pv_actual_cols = []
    
    # Check for forecast column
    for col in columns:
        if 'pv_forecast' in col.lower():
            pv_forecast_col = col
            break
    
    # Check for actual PV generation columns (may be multiple: roof, facade, etc.)
    for col in columns:
        if f'{building_id}_pv' in col and 'forecast' not in col.lower():
            pv_actual_cols.append(col)
        elif 'pv_generation' in col.lower() or 'pv_actual' in col.lower():
            pv_actual_cols.append(col)
    
    # Fallback options
    if not pv_forecast_col:
        # Use grid export as proxy for PV if forecast not available
        for col in columns:
            if f'{building_id}_grid_export' == col:
                pv_forecast_col = col
                logger.warning(f"Using grid export as PV forecast proxy for {building_id}")
                break
    
    if not pv_actual_cols:
        # Use grid export as proxy for actual PV if not available
        for col in columns:
            if f'{building_id}_grid_export' == col:
                pv_actual_cols = [col]
                logger.warning(f"Using grid export as PV actual proxy for {building_id}")
                break
                
    if not pv_forecast_col:
        raise ValueError(f"No PV forecast column found in {building_id}_processed_data")
    if not pv_actual_cols:
        raise ValueError(f"No PV actual columns found in {building_id}_processed_data")
    
    logger.info(f"Found PV columns for {building_id}: forecast={pv_forecast_col}, actual={pv_actual_cols}")
    return pv_forecast_col, pv_actual_cols

def fetch_day_df(con, building_id, day_str, pv_forecast_col, pv_actual_cols):
    """Fetch complete day data with PV information"""
    # Create query to sum multiple PV actual columns
    pv_actual_sum = " + ".join([f"COALESCE({col}, 0)" for col in pv_actual_cols])
    
    query = f"""
        SELECT 
            DATE_TRUNC('hour', utc_timestamp) AS timestamp_hour,
            EXTRACT(hour FROM utc_timestamp) AS hour,
            COALESCE(price_per_kwh, 0.3) AS price_per_kwh,
            COALESCE({pv_forecast_col}, 0) AS pv_forecast,
            ({pv_actual_sum}) AS pv_actual,
            *
        FROM {building_id}_processed_data
        WHERE DATE(utc_timestamp) = '{day_str}'
        ORDER BY timestamp_hour
    """
    
    df = con.execute(query).df()
    
    if df.shape[0] != 24:
        raise ValueError(f"Incomplete 24-hour data for {building_id} on {day_str}: got {df.shape[0]} hours")
    
    # Set hour as index and ensure complete 24-hour coverage
    df = df.set_index("hour")
    df = df.reindex(range(24), fill_value=0)
    
    return df

def get_device_load_sum(df_day, building_id):
    """Calculate total device load for the day"""
    # Get all device columns for this building (exclude PV, grid, price, timestamp columns)
    device_columns = [col for col in df_day.columns 
                     if building_id in col and 'pv' not in col and 'grid' not in col 
                     and 'price' not in col and 'timestamp' not in col]
    
    if not device_columns:
        logger.warning(f"No device columns found for {building_id}, using 1 kWh baseline load")
        return np.ones(24)  # 1 kWh baseline load per hour
    
    # Sum device loads (take absolute values as loads are typically positive)
    device_loads = df_day[device_columns].fillna(0).abs()
    total_load = device_loads.sum(axis=1).values
    
    # Ensure non-zero load for meaningful analysis
    total_load = np.maximum(total_load, 0.1)  # Minimum 0.1 kWh per hour
    
    return total_load

def simulate_costs_with_pv(df_day, building_id):
    """Simulate costs with PV generation"""
    pv_actual = df_day['pv_actual'].values
    total_load = get_device_load_sum(df_day, building_id)
    prices = df_day['price_per_kwh'].values
    
    # Convert PV to positive generation (if stored as negative)
    pv_generation = np.abs(pv_actual)
    
    # Calculate net grid flows
    net_import = np.maximum(0, total_load - pv_generation)
    net_export = np.maximum(0, pv_generation - total_load)
    
    # Calculate costs (assume zero feed-in tariff for simplicity)
    feed_in_price = 0.0  # EUR/kWh
    cost_hourly = net_import * prices - net_export * feed_in_price
    daily_cost = cost_hourly.sum()
    
    # Calculate cost without PV
    daily_cost_no_pv = (total_load * prices).sum()
    
    # Calculate self-consumption metrics
    pv_consumed = np.minimum(total_load, pv_generation)
    total_pv = pv_generation.sum()
    total_load_sum = total_load.sum()
    
    self_consumption_ratio = pv_consumed.sum() / total_pv if total_pv > 0 else 0
    self_generation_ratio = pv_consumed.sum() / total_load_sum if total_load_sum > 0 else 0
    
    return {
        'daily_cost': daily_cost,
        'daily_cost_no_pv': daily_cost_no_pv,
        'cost_savings': daily_cost_no_pv - daily_cost,
        'self_consumption_ratio': self_consumption_ratio,
        'self_generation_ratio': self_generation_ratio,
        'net_import': net_import,
        'net_export': net_export,
        'pv_consumed': pv_consumed,
        'total_pv_generation': total_pv,
        'total_load': total_load_sum
    }

def analyze_building(con, building_id):
    """Analyze PV performance for a single building"""
    logger.info(f"Starting analysis for {building_id}")
    
    # Setup
    table = f"{building_id}_processed_data"
    ensure_table_exists(con, table)
    all_days = fetch_full_days(con, building_id)
    days_to_use = all_days[:min_days]
    pv_forecast_col, pv_actual_cols = detect_pv_columns(con, building_id)
    
    logger.info(f"Found {len(all_days)} complete days, analyzing first {min_days}")
    logger.info(f"Using PV columns: forecast={pv_forecast_col}, actual={pv_actual_cols}")
    
    # Create output directory
    building_output_dir = os.path.join(output_dir, building_id)
    os.makedirs(building_output_dir, exist_ok=True)
    
    # Collect metrics over all days
    daily_metrics = []
    forecast_errors_all = []
    
    for i, day in enumerate(days_to_use):
        if i % 20 == 0:
            logger.info(f"Processing {building_id}, day {i+1}/{min_days}")
        
        try:
            # Fetch day data
            df_day = fetch_day_df(con, building_id, day, pv_forecast_col, pv_actual_cols)
            
            # Simulate costs and metrics
            metrics = simulate_costs_with_pv(df_day, building_id)
            
            # Calculate forecast errors
            pv_forecast = df_day['pv_forecast'].values
            pv_actual = df_day['pv_actual'].values
            forecast_errors = pv_forecast - pv_actual
            forecast_errors_all.extend(forecast_errors)
            
            # Daily aggregated metrics
            daily_pv_generation = np.abs(pv_actual).sum()
            daily_pv_forecast = np.abs(pv_forecast).sum()
            pv_forecast_rmse = np.sqrt(np.mean(forecast_errors**2))
            
            daily_metrics.append({
                'day': day,
                'daily_pv_generation': daily_pv_generation,
                'daily_pv_forecast': daily_pv_forecast,
                'pv_forecast_rmse': pv_forecast_rmse,
                'pv_forecast_error_mean': np.mean(forecast_errors),
                **metrics
            })
            
        except Exception as e:
            logger.warning(f"Failed to process day {day} for {building_id}: {e}")
            continue
    
    if not daily_metrics:
        logger.error(f"No valid data processed for {building_id}")
        return
    
    # Convert to DataFrame
    df_metrics = pd.DataFrame(daily_metrics)
    df_metrics['day'] = pd.to_datetime(df_metrics['day'])
    
    logger.info(f"Successfully processed {len(df_metrics)} days for {building_id}")
    
    # Generate plots
    generate_plots(df_metrics, forecast_errors_all, building_id, building_output_dir)
    
    # Save metrics to CSV
    df_metrics.to_csv(os.path.join(building_output_dir, 'pv_analysis_metrics.csv'), index=False)

def generate_plots(df_metrics, forecast_errors_all, building_id, output_dir):
    """Generate all analysis plots for a building"""
    
    # 1. PV forecast error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(forecast_errors_all, bins=30, kde=True)
    plt.title(f"{building_id}: PV Forecast Error Distribution")
    plt.xlabel("Forecast − Actual (kWh)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pv_forecast_error_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Self-consumption ratio over days
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(df_metrics)), df_metrics['self_consumption_ratio'], 'b-', alpha=0.7)
    plt.title(f"{building_id}: Daily Self-Consumption Ratio over {len(df_metrics)} Days")
    plt.xlabel("Day Number")
    plt.ylabel("Self-Consumption Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'self_consumption_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cost savings over days
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(df_metrics)), df_metrics['cost_savings'], 'g-', alpha=0.7)
    plt.title(f"{building_id}: Daily Cost Savings from PV over {len(df_metrics)} Days")
    plt.xlabel("Day Number")
    plt.ylabel("Cost Savings (EUR)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_savings.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Cost sensitivity to PV forecast error
    plt.figure(figsize=(10, 8))
    plt.scatter(df_metrics['pv_forecast_rmse'], df_metrics['cost_savings'], alpha=0.6)
    plt.title(f"{building_id}: Cost Savings vs PV Forecast Error")
    plt.xlabel("PV Forecast RMSE (kWh)")
    plt.ylabel("Cost Savings (EUR)")
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df_metrics['pv_forecast_rmse'], df_metrics['cost_savings'], 1)
    p = np.poly1d(z)
    plt.plot(df_metrics['pv_forecast_rmse'], p(df_metrics['pv_forecast_rmse']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_vs_pv_error.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. PV generation statistics
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df_metrics['daily_pv_generation'], 'b-', alpha=0.7, label='Actual')
    plt.plot(df_metrics['daily_pv_forecast'], 'r--', alpha=0.7, label='Forecast')
    plt.title('Daily PV Generation')
    plt.ylabel('kWh')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(df_metrics['self_consumption_ratio'], bins=20, alpha=0.7, color='green')
    plt.title('Self-Consumption Ratio Distribution')
    plt.xlabel('Self-Consumption Ratio')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 3)
    plt.hist(df_metrics['cost_savings'], bins=20, alpha=0.7, color='orange')
    plt.title('Daily Cost Savings Distribution')
    plt.xlabel('Cost Savings (EUR)')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df_metrics['daily_pv_generation'], df_metrics['self_consumption_ratio'], alpha=0.6)
    plt.title('Self-Consumption vs PV Generation')
    plt.xlabel('Daily PV Generation (kWh)')
    plt.ylabel('Self-Consumption Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pv_statistics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Generated plots for {building_id} in {output_dir}")

def main():
    """Main analysis function"""
    logger.info("Starting comprehensive PV analysis")
    logger.info(f"Analyzing buildings: {building_ids}")
    logger.info(f"Analysis period: {min_days} days per building")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to database
    con = get_con()
    
    # Summary statistics
    summary_stats = []
    
    for building_id in building_ids:
        try:
            analyze_building(con, building_id)
            
            # Load saved metrics for summary
            metrics_file = os.path.join(output_dir, building_id, 'pv_analysis_metrics.csv')
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                summary_stats.append({
                    'building_id': building_id,
                    'avg_daily_pv_generation': df['daily_pv_generation'].mean(),
                    'avg_self_consumption_ratio': df['self_consumption_ratio'].mean(),
                    'avg_cost_savings': df['cost_savings'].mean(),
                    'avg_forecast_rmse': df['pv_forecast_rmse'].mean(),
                    'total_cost_savings': df['cost_savings'].sum(),
                    'days_analyzed': len(df)
                })
                
        except Exception as e:
            logger.error(f"Failed to analyze {building_id}: {e}")
            continue
    
    # Generate summary report
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(os.path.join(output_dir, 'pv_analysis_summary.csv'), index=False)
        
        # Summary plot
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.bar(summary_df['building_id'], summary_df['avg_daily_pv_generation'])
        plt.title('Average Daily PV Generation by Building')
        plt.ylabel('kWh')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 2)
        plt.bar(summary_df['building_id'], summary_df['avg_self_consumption_ratio'])
        plt.title('Average Self-Consumption Ratio by Building')
        plt.ylabel('Ratio')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 3)
        plt.bar(summary_df['building_id'], summary_df['avg_cost_savings'])
        plt.title('Average Daily Cost Savings by Building')
        plt.ylabel('EUR')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 4)
        plt.bar(summary_df['building_id'], summary_df['avg_forecast_rmse'])
        plt.title('Average PV Forecast RMSE by Building')
        plt.ylabel('kWh')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 5)
        plt.bar(summary_df['building_id'], summary_df['total_cost_savings'])
        plt.title('Total Cost Savings by Building')
        plt.ylabel('EUR')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 6)
        plt.scatter(summary_df['avg_daily_pv_generation'], summary_df['avg_self_consumption_ratio'])
        for i, building in enumerate(summary_df['building_id']):
            plt.annotate(building, (summary_df['avg_daily_pv_generation'].iloc[i], 
                                   summary_df['avg_self_consumption_ratio'].iloc[i]))
        plt.title('Self-Consumption vs PV Generation')
        plt.xlabel('Avg Daily PV Generation (kWh)')
        plt.ylabel('Avg Self-Consumption Ratio')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pv_analysis_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Analysis complete! Summary saved to {output_dir}")
        logger.info(f"Total buildings analyzed: {len(summary_stats)}")
        logger.info(f"Average cost savings across all buildings: {summary_df['avg_cost_savings'].mean():.2f} EUR/day")
    
    con.close()

if __name__ == "__main__":
    main()