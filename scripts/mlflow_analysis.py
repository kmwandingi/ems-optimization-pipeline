#!/usr/bin/env python
"""
MLflow Analysis Utilities for EMS Optimization

This script provides simple analysis and comparison of MLflow experiments
for the EMS optimization pipelines.

Features:
- Compare Comparison Pipeline and Learning Pipeline experiments
- Analyze savings performance across different buildings
- Generate experiment comparison reports
- Export experiment data for further analysis

Usage:
    python scripts/mlflow_analysis.py
    python scripts/mlflow_analysis.py --experiment "Comparison_Pipeline"
    python scripts/mlflow_analysis.py --export results/mlflow_analysis.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add utils to path for MLflow tracker
sys.path.append(str(Path.cwd() / "utils"))

try:
    import mlflow
    from mlflow_tracker import EMS_OptimizationTracker
    MLFLOW_AVAILABLE = True
except ImportError as e:
    print(f"❌ MLflow not available: {e}")
    print("Please install MLflow: pip install mlflow")
    sys.exit(1)


def setup_mlflow():
    """Setup MLflow connection."""
    mlflow.set_tracking_uri("file:./mlflow_runs")
    print("✓ Connected to MLflow tracking")


def get_experiment_runs(experiment_name: str) -> pd.DataFrame:
    """
    Get all runs from a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        DataFrame with experiment runs
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"⚠ Experiment '{experiment_name}' not found")
            return pd.DataFrame()
        
        runs = mlflow.search_runs(experiment.experiment_id)
        print(f"✓ Found {len(runs)} runs in experiment '{experiment_name}'")
        return runs
    except Exception as e:
        print(f"❌ Error getting experiment runs: {e}")
        return pd.DataFrame()


def analyze_comparison_pipeline_experiments() -> pd.DataFrame:
    """Analyze Comparison Pipeline experiments."""
    print("\n" + "="*60)
    print("COMPARISON PIPELINE EXPERIMENT ANALYSIS")
    print("="*60)
    
    runs_df = get_experiment_runs("Comparison_Pipeline")
    
    if runs_df.empty:
        print("⚠ No Comparison Pipeline experiments found")
        return pd.DataFrame()
    
    # Extract key metrics
    analysis = {
        "run_id": [],
        "building_id": [],
        "optimization_mode": [],
        "battery_enabled": [],
        "ev_enabled": [],
        "n_days": [],
        "avg_savings_pct": [],
        "total_cost_avg": [],
        "total_days_processed": [],
        "execution_status": [],
        "start_time": []
    }
    
    for _, run in runs_df.iterrows():
        analysis["run_id"].append(run['run_id'][:8])
        
        # Extract parameters (using correct MLflow column format)
        analysis["building_id"].append(run.get('params.building_id', 'Unknown'))
        analysis["optimization_mode"].append(run.get('params.optimization_mode', 'Unknown'))
        analysis["battery_enabled"].append(run.get('params.battery_enabled', 'Unknown'))
        analysis["ev_enabled"].append(run.get('params.ev_enabled', 'Unknown'))
        # Convert string parameters to numeric
        try:
            n_days = int(run.get('params.n_days', 0) or 0)
        except (ValueError, TypeError):
            n_days = 0
        analysis["n_days"].append(n_days)
        analysis["execution_status"].append(run.get('params.execution_status', 'Unknown'))
        
        # Extract metrics (using correct MLflow column format)
        analysis["avg_savings_pct"].append(run.get('metrics.avg_savings_pct', 0.0))
        analysis["total_cost_avg"].append(run.get('metrics.total_cost_avg', 0.0))
        analysis["total_days_processed"].append(run.get('metrics.total_days_processed', 0))
        
        # Extract start time
        analysis["start_time"].append(run.get('start_time', 0))
    
    analysis_df = pd.DataFrame(analysis)
    
    if len(analysis_df) > 0:
        # Summary statistics
        print(f"Total runs: {len(analysis_df)}")
        print(f"Unique buildings: {analysis_df['building_id'].nunique()}")
        print(f"Optimization modes: {list(analysis_df['optimization_mode'].unique())}")
        
        # Best performing runs
        if 'avg_savings_pct' in analysis_df.columns and analysis_df['avg_savings_pct'].notna().any():
            best_run = analysis_df.loc[analysis_df['avg_savings_pct'].idxmax()]
            print(f"\nBest performing run:")
            print(f"  Run ID: {best_run['run_id']}")
            print(f"  Building: {best_run['building_id']}")
            print(f"  Mode: {best_run['optimization_mode']}")
            print(f"  Savings: {best_run['avg_savings_pct']:.2f}%")
            
            # Average performance
            avg_savings = analysis_df['avg_savings_pct'].mean()
            print(f"\nAverage savings across all runs: {avg_savings:.2f}%")
    
    return analysis_df


def analyze_learning_pipeline_experiments() -> pd.DataFrame:
    """Analyze Learning Pipeline experiments."""
    print("\n" + "="*60)
    print("LEARNING PIPELINE EXPERIMENT ANALYSIS")
    print("="*60)
    
    runs_df = get_experiment_runs("Learning_Pipeline")
    
    if runs_df.empty:
        print("⚠ No Learning Pipeline experiments found")
        return pd.DataFrame()
    
    # Extract key metrics
    analysis = {
        "run_id": [],
        "building_id": [],
        "optimization_mode": [],
        "n_days": [],
        "training_days_count": [],
        "devices_trained": [],
        "avg_savings_pct": [],
        "total_cumulative_savings": [],
        "execution_status": [],
        "start_time": []
    }
    
    for _, run in runs_df.iterrows():
        analysis["run_id"].append(run['run_id'][:8])
        
        # Extract parameters (using correct MLflow column format)
        analysis["building_id"].append(run.get('params.building_id', 'Unknown'))
        analysis["optimization_mode"].append(run.get('params.optimization_mode', 'Unknown'))
        # Convert string parameters to numeric
        try:
            n_days = int(run.get('params.n_days', 0) or 0)
        except (ValueError, TypeError):
            n_days = 0
        analysis["n_days"].append(n_days)
        
        try:
            training_days = int(run.get('params.training_days_count', 0) or 0)
        except (ValueError, TypeError):
            training_days = 0
        analysis["training_days_count"].append(training_days)
        
        try:
            devices_trained = int(run.get('params.devices_trained', 0) or 0)
        except (ValueError, TypeError):
            devices_trained = 0
        analysis["devices_trained"].append(devices_trained)
        analysis["execution_status"].append(run.get('params.execution_status', 'Unknown'))
        
        # Extract metrics (using correct MLflow column format)
        analysis["avg_savings_pct"].append(run.get('metrics.avg_savings_pct', 0.0))
        analysis["total_cumulative_savings"].append(run.get('metrics.total_cumulative_savings', 0.0))
        
        # Extract start time
        analysis["start_time"].append(run.get('start_time', 0))
    
    analysis_df = pd.DataFrame(analysis)
    
    if len(analysis_df) > 0:
        # Summary statistics
        print(f"Total runs: {len(analysis_df)}")
        print(f"Unique buildings: {analysis_df['building_id'].nunique()}")
        print(f"Average training days: {analysis_df['training_days_count'].mean():.1f}")
        print(f"Average devices trained: {analysis_df['devices_trained'].mean():.1f}")
        
        # Best performing runs
        if 'avg_savings_pct' in analysis_df.columns and analysis_df['avg_savings_pct'].notna().any():
            best_run = analysis_df.loc[analysis_df['avg_savings_pct'].idxmax()]
            print(f"\nBest performing run:")
            print(f"  Run ID: {best_run['run_id']}")
            print(f"  Building: {best_run['building_id']}")
            print(f"  Training days: {best_run['training_days_count']}")
            print(f"  Savings: {best_run['avg_savings_pct']:.2f}%")
            
            # Learning effectiveness
            if analysis_df['devices_trained'].sum() > 0:
                avg_savings = analysis_df['avg_savings_pct'].mean()
                print(f"\nAverage savings with learning: {avg_savings:.2f}%")
    
    return analysis_df


def compare_experiments(comparison_pipeline_df: pd.DataFrame, learning_pipeline_df: pd.DataFrame):
    """Compare Comparison Pipeline and Learning Pipeline experiments."""
    print("\n" + "="*60)
    print("PIPELINE COMPARISON")
    print("="*60)
    
    if comparison_pipeline_df.empty and learning_pipeline_df.empty:
        print("⚠ No experiments to compare")
        return
    
    # Compare performance
    if not comparison_pipeline_df.empty and 'avg_savings_pct' in comparison_pipeline_df.columns:
        comparison_avg = comparison_pipeline_df['avg_savings_pct'].mean()
        print(f"Comparison Pipeline average savings: {comparison_avg:.2f}%")
    
    if not learning_pipeline_df.empty and 'avg_savings_pct' in learning_pipeline_df.columns:
        learning_avg = learning_pipeline_df['avg_savings_pct'].mean()
        print(f"Learning Pipeline average savings: {learning_avg:.2f}%")
    
    # Compare by building (if both pipelines have data for same buildings)
    if not comparison_pipeline_df.empty and not learning_pipeline_df.empty:
        common_buildings = set(comparison_pipeline_df['building_id']) & set(learning_pipeline_df['building_id'])
        
        if common_buildings:
            print(f"\nCommon buildings analyzed: {len(common_buildings)}")
            for building in common_buildings:
                comparison_building = comparison_pipeline_df[comparison_pipeline_df['building_id'] == building]
                learning_building = learning_pipeline_df[learning_pipeline_df['building_id'] == building]
                
                if not comparison_building.empty and not learning_building.empty:
                    comparison_savings = comparison_building['avg_savings_pct'].mean()
                    learning_savings = learning_building['avg_savings_pct'].mean()
                    print(f"  {building}: Comparison={comparison_savings:.2f}%, Learning={learning_savings:.2f}%")


def create_experiment_visualization(comparison_pipeline_df: pd.DataFrame, learning_pipeline_df: pd.DataFrame, output_dir: str = "results/mlflow_analysis"):
    """Create visualizations of experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('EMS Optimization Experiment Analysis', fontsize=16)
    
    # Comparison Pipeline savings distribution
    if not comparison_pipeline_df.empty and 'avg_savings_pct' in comparison_pipeline_df.columns:
        axes[0, 0].hist(comparison_pipeline_df['avg_savings_pct'], bins=10, alpha=0.7, color='blue', label='Comparison Pipeline')
        axes[0, 0].set_title('Comparison Pipeline: Savings Distribution')
        axes[0, 0].set_xlabel('Average Savings (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Learning Pipeline savings distribution
    if not learning_pipeline_df.empty and 'avg_savings_pct' in learning_pipeline_df.columns:
        axes[0, 1].hist(learning_pipeline_df['avg_savings_pct'], bins=10, alpha=0.7, color='green', label='Learning Pipeline')
        axes[0, 1].set_title('Learning Pipeline: Savings Distribution')
        axes[0, 1].set_xlabel('Average Savings (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Building performance comparison
    if not comparison_pipeline_df.empty and not learning_pipeline_df.empty:
        common_buildings = set(comparison_pipeline_df['building_id']) & set(learning_pipeline_df['building_id'])
        if common_buildings:
            building_comparison = []
            for building in common_buildings:
                comparison_avg = comparison_pipeline_df[comparison_pipeline_df['building_id'] == building]['avg_savings_pct'].mean()
                learning_avg = learning_pipeline_df[learning_pipeline_df['building_id'] == building]['avg_savings_pct'].mean()
                building_comparison.append({'building': building, 'Comparison_Pipeline': comparison_avg, 'Learning_Pipeline': learning_avg})
            
            if building_comparison:
                comparison_df = pd.DataFrame(building_comparison)
                x = np.arange(len(comparison_df))
                width = 0.35
                
                axes[1, 0].bar(x - width/2, comparison_df['Comparison_Pipeline'], width, label='Comparison Pipeline', alpha=0.7, color='blue')
                axes[1, 0].bar(x + width/2, comparison_df['Learning_Pipeline'], width, label='Learning Pipeline', alpha=0.7, color='green')
                axes[1, 0].set_title('Savings by Building')
                axes[1, 0].set_xlabel('Building')
                axes[1, 0].set_ylabel('Average Savings (%)')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels([b.split('_')[-1] for b in comparison_df['building']], rotation=45)
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
    
    # Experiment timeline
    all_experiments = []
    if not comparison_pipeline_df.empty:
        for _, row in comparison_pipeline_df.iterrows():
            all_experiments.append({
                'timestamp': pd.to_datetime(row['start_time'], unit='ms'),
                'savings': row.get('avg_savings_pct', 0),
                'pipeline': 'Comparison'
            })
    
    if not learning_pipeline_df.empty:
        for _, row in learning_pipeline_df.iterrows():
            all_experiments.append({
                'timestamp': pd.to_datetime(row['start_time'], unit='ms'),
                'savings': row.get('avg_savings_pct', 0),
                'pipeline': 'Learning'
            })
    
    if all_experiments:
        timeline_df = pd.DataFrame(all_experiments)
        for pipeline in ['Comparison', 'Learning']:
            pipeline_data = timeline_df[timeline_df['pipeline'] == pipeline]
            if not pipeline_data.empty:
                color = 'blue' if pipeline == 'Comparison' else 'green'
                axes[1, 1].scatter(pipeline_data['timestamp'], pipeline_data['savings'], 
                                 alpha=0.7, label=f'{pipeline} Pipeline', color=color)
        
        axes[1, 1].set_title('Experiment Timeline')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Savings (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_file = f"{output_dir}/experiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved experiment visualization: {output_file}")
    return output_file


def export_experiment_data(comparison_pipeline_df: pd.DataFrame, learning_pipeline_df: pd.DataFrame, output_file: str):
    """Export experiment data to CSV."""
    
    # Combine datasets with pipeline identifier
    combined_data = []
    
    if not comparison_pipeline_df.empty:
        comparison_pipeline_df['pipeline'] = 'Comparison'
        combined_data.append(comparison_pipeline_df)
    
    if not learning_pipeline_df.empty:
        learning_pipeline_df['pipeline'] = 'Learning'
        combined_data.append(learning_pipeline_df)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        # Note: CSV creation removed to save disk space
        print(f"✓ Experiment data analysis completed (CSV creation disabled to save disk space)")
        return output_file
    else:
        print("⚠ No data to export")
        return None


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze EMS MLflow experiments")
    parser.add_argument("--experiment", type=str, help="Specific experiment to analyze")
    parser.add_argument("--export", type=str, help="Export data to CSV file")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    
    args = parser.parse_args()
    
    print("="*60)
    print("EMS MLFLOW EXPERIMENT ANALYSIS")
    print("="*60)
    
    # Setup MLflow
    setup_mlflow()
    
    if args.experiment:
        # Analyze specific experiment
        runs_df = get_experiment_runs(args.experiment)
        if not runs_df.empty:
            print(f"\n{args.experiment} Analysis:")
            print(f"Total runs: {len(runs_df)}")
            print("\nRecent runs:")
            for _, run in runs_df.head(5).iterrows():
                building_id = run.get('params.building_id', 'Unknown')
                savings_pct = run.get('metrics.avg_savings_pct', 0.0)
                print(f"  {run['run_id'][:8]}: {building_id} - {savings_pct:.2f}% savings")
    else:
        # Analyze both pipelines
        comparison_pipeline_df = analyze_comparison_pipeline_experiments()
        learning_pipeline_df = analyze_learning_pipeline_experiments()
        
        # Compare experiments
        compare_experiments(comparison_pipeline_df, learning_pipeline_df)
        
        # Export data if requested
        if args.export:
            export_experiment_data(comparison_pipeline_df, learning_pipeline_df, args.export)
        
        # Create visualizations if requested
        if args.visualize:
            create_experiment_visualization(comparison_pipeline_df, learning_pipeline_df)
    
    print("\n✅ Analysis completed")


if __name__ == "__main__":
    main()