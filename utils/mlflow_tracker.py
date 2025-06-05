#!/usr/bin/env python
"""
MLflow EMS Optimization Tracker

This module provides elegant MLflow tracking capabilities for EMS optimization experiments.
Wraps around existing pipelines without modifying their core logic.

Features:
- Local file-based tracking (no server infrastructure needed)
- Automatic parameter and metrics logging
- Artifact management for results and visualizations
- Experiment comparison and analysis
- Non-intrusive integration with existing Agent-based pipelines

Usage:
    with EMS_OptimizationTracker("Comparison_Pipeline") as tracker:
        tracker.log_params({"building_id": "DE_KN_residential1", "mode": "centralized"})
        # ... run optimization ...
        tracker.log_metrics({"total_cost": 150.0, "savings_pct": 8.5})
        tracker.log_artifacts("results/output")
"""

import mlflow
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union


class EMS_OptimizationTracker:
    """
    MLflow tracker for EMS optimization experiments.
    
    Provides elegant experiment tracking for Comparison Pipeline and Learning Pipeline
    without modifying existing Agent-based optimization logic.
    """
    
    def __init__(self, experiment_name: str = "EMS_Optimization", tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker for EMS optimization.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (defaults to local file storage)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlflow_runs"
        self.run_context = None
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking configuration."""
        # Set tracking URI to local file storage
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create experiment if it doesn't exist
        try:
            mlflow.set_experiment(self.experiment_name)
            print(f"✓ MLflow experiment: {self.experiment_name}")
            print(f"✓ Tracking URI: {self.tracking_uri}")
        except Exception as e:
            # Create experiment explicitly
            experiment_id = mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(experiment_id=experiment_id)
            print(f"✓ Created MLflow experiment: {self.experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> 'EMS_OptimizationTracker':
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run
            
        Returns:
            Self for method chaining
        """
        self.run_context = mlflow.start_run(run_name=run_name, nested=nested)
        
        # Log system information
        self.log_params({
            "mlflow_version": mlflow.__version__,
            "tracking_timestamp": datetime.now().isoformat(),
            "experiment_name": self.experiment_name
        })
        
        if run_name:
            print(f"✓ Started MLflow run: {run_name}")
        else:
            print(f"✓ Started MLflow run: {self.run_context.info.run_id[:8]}")
        
        return self
    
    def end_run(self):
        """End the current MLflow run."""
        if self.run_context:
            mlflow.end_run()
            self.run_context = None
            print("✓ Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters for the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        if not self.run_context:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        # Convert all values to strings (MLflow requirement)
        str_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(str_params)
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """
        Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        if not self.run_context:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        # Filter numeric values only
        numeric_metrics = {k: float(v) for k, v in metrics.items() 
                          if isinstance(v, (int, float)) and not isinstance(v, bool)}
        
        if step is not None:
            for key, value in numeric_metrics.items():
                mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metrics(numeric_metrics)
    
    def log_artifacts(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifacts (files/directories) for the current run.
        
        Args:
            local_path: Local path to files or directory to upload
            artifact_path: Optional artifact path within the run
        """
        if not self.run_context:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        if os.path.exists(local_path):
            if os.path.isdir(local_path):
                mlflow.log_artifacts(local_path, artifact_path)
            else:
                mlflow.log_artifact(local_path, artifact_path)
            print(f"✓ Logged artifacts from: {local_path}")
        else:
            print(f"⚠ Artifact path not found: {local_path}")
    
    def log_dict(self, dictionary: Dict[str, Any], filename: str):
        """
        Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to save
            filename: Name of the JSON file
        """
        if not self.run_context:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        # Create temporary file and log it
        temp_path = f"/tmp/mlflow_{filename}"
        with open(temp_path, 'w') as f:
            json.dump(dictionary, f, indent=2, default=str)
        
        mlflow.log_artifact(temp_path, filename)
        os.remove(temp_path)  # Clean up
        print(f"✓ Logged dictionary as: {filename}")
    
    def log_model_config(self, config: Dict[str, Any]):
        """
        Log model/optimization configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.log_dict(config, "optimization_config.json")
    
    def log_battery_config(self, battery_params: Dict[str, Any]):
        """Log battery configuration parameters."""
        battery_config = {f"battery_{k}": v for k, v in battery_params.items()}
        self.log_params(battery_config)
    
    def log_ev_config(self, ev_params: Dict[str, Any]):
        """Log EV configuration parameters."""
        ev_config = {f"ev_{k}": v for k, v in ev_params.items()}
        self.log_params(ev_config)
    
    def log_grid_config(self, grid_params: Dict[str, Any]):
        """Log grid configuration parameters."""
        grid_config = {f"grid_{k}": v for k, v in grid_params.items()}
        self.log_params(grid_config)
    
    def log_optimization_results(self, results: Dict[str, Any], day: Optional[str] = None):
        """
        Log optimization results for a specific day or overall.
        
        Args:
            results: Results dictionary with costs, savings, etc.
            day: Optional day identifier for step-based logging
        """
        if day:
            # Log as step-based metrics for daily tracking
            step = None
            if isinstance(day, str):
                try:
                    day_dt = datetime.strptime(day, '%Y-%m-%d')
                    step = (day_dt - datetime(day_dt.year, 1, 1)).days
                except:
                    pass
            
            self.log_metrics(results, step=step)
        else:
            # Log as final metrics
            self.log_metrics(results)
    
    def log_probability_learning(self, device_probs: Dict[str, Any]):
        """
        Log probability learning results from ProbabilityModelAgent.
        
        Args:
            device_probs: Device probability learning results
        """
        learning_metrics = {}
        
        for device, prob_data in device_probs.items():
            if isinstance(prob_data, dict):
                # Log JS divergence and update counts
                if 'js_prior' in prob_data:
                    learning_metrics[f"{device}_js_divergence_prior"] = prob_data['js_prior']
                if 'js_step' in prob_data:
                    learning_metrics[f"{device}_js_divergence_step"] = prob_data['js_step']
                if 'observation_count' in prob_data:
                    learning_metrics[f"{device}_observation_count"] = prob_data['observation_count']
                if 'updates_count' in prob_data:
                    learning_metrics[f"{device}_updates_count"] = prob_data['updates_count']
        
        if learning_metrics:
            self.log_metrics(learning_metrics)
            print(f"✓ Logged probability learning metrics for {len(device_probs)} devices")
    
    def log_agent_performance(self, optimization_time: float, solver_status: str, 
                             variables_count: Optional[int] = None, constraints_count: Optional[int] = None):
        """
        Log agent optimization performance metrics.
        
        Args:
            optimization_time: Time taken for optimization in seconds
            solver_status: Status from optimization solver
            variables_count: Number of optimization variables
            constraints_count: Number of optimization constraints
        """
        performance_metrics = {
            "optimization_time_seconds": optimization_time,
            "solver_status_success": 1.0 if solver_status.lower() in ['optimal', 'feasible'] else 0.0
        }
        
        performance_params = {
            "solver_status": solver_status
        }
        
        if variables_count is not None:
            performance_metrics["variables_count"] = variables_count
        if constraints_count is not None:
            performance_metrics["constraints_count"] = constraints_count
        
        self.log_metrics(performance_metrics)
        self.log_params(performance_params)
    
    def __enter__(self):
        """Context manager entry - start a run with timestamp."""
        run_name = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.start_run(run_name)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - end the run."""
        if exc_type is not None:
            # Log exception if one occurred
            self.log_params({
                "execution_status": "failed",
                "error_type": str(exc_type.__name__),
                "error_message": str(exc_val)
            })
        else:
            self.log_params({"execution_status": "success"})
        
        self.end_run()


def get_experiment_tracker(experiment_name: str) -> EMS_OptimizationTracker:
    """
    Factory function to get an experiment tracker.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Configured EMS_OptimizationTracker instance
    """
    return EMS_OptimizationTracker(experiment_name)


# Convenience functions for common tracking patterns
def track_pipeline_a_run(building_id: str, mode: str, battery_enabled: bool, ev_enabled: bool, n_days: int):
    """
    Create a configured tracker for Comparison Pipeline runs.
    
    Returns:
        Context manager for Comparison Pipeline tracking
    """
    tracker = EMS_OptimizationTracker("Comparison_Pipeline")
    run_name = f"comparison_{building_id}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    return tracker.start_run(run_name)


def track_pipeline_b_run(building_id: str, mode: str, battery_enabled: bool, ev_enabled: bool, n_days: int):
    """
    Create a configured tracker for Learning Pipeline runs.
    
    Returns:
        Context manager for Learning Pipeline tracking
    """
    tracker = EMS_OptimizationTracker("Learning_Pipeline")
    run_name = f"learning_{building_id}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    return tracker.start_run(run_name)