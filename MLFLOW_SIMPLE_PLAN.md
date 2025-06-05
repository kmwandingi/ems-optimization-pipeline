# MLflow Simple Integration Plan - EMS Optimization Tracking MVP

## 🎯 MVP Objective

Create a **minimal viable product** for tracking EMS optimization experiments using MLflow's core tracking capabilities. Focus on logging optimization parameters, results, and artifacts without complex model versioning.

## 📋 Simple Approach

### Core Concept
Treat each EMS optimization run as an **experiment** where:
- **Parameters** = Optimization inputs (building, mode, battery settings)
- **Metrics** = Optimization results (costs, savings, solver performance)  
- **Artifacts** = Generated files (schedules, plots, reports)

## 🚀 MVP Implementation Plan

### Phase 1: Basic Tracking (1 week)

#### 1.1 Setup MLflow Tracking
```python
# Simple MLflow wrapper for EMS
import mlflow
import os
from datetime import datetime

class EMS_SimpleTracker:
    def __init__(self, experiment_name="EMS_Optimization"):
        # Local file-based tracking (no server needed)
        mlflow.set_tracking_uri("file:./mlflow_runs")
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params_dict):
        mlflow.log_params(params_dict)
    
    def log_metrics(self, metrics_dict):
        mlflow.log_metrics(metrics_dict)
    
    def log_artifacts(self, folder_path):
        mlflow.log_artifacts(folder_path)
    
    def end_run(self):
        mlflow.end_run()
```

#### 1.2 Integration Points

**Pipeline A (scripts/01_run.py):**
```python
# Add at start of main()
tracker = EMS_SimpleTracker("Pipeline_A_Comparison")

with tracker.start_run(f"comparison_{building_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
    # Log input parameters
    tracker.log_params({
        "building_id": building_id,
        "optimization_mode": mode,
        "battery_enabled": battery_enabled,
        "ev_enabled": ev_enabled,
        "n_days": n_days
    })
    
    # ... existing optimization logic ...
    
    # Log results
    tracker.log_metrics({
        "total_days_processed": len(results),
        "avg_savings_pct": avg_savings,
        "total_cost": sum(costs),
        "optimization_success": 1.0
    })
    
    # Log output files
    tracker.log_artifacts("results/output")
```

**Pipeline B (scripts/02_integrated_pipeline.py):**
```python
# Add similar tracking
tracker = EMS_SimpleTracker("Pipeline_B_Learning")

with tracker.start_run(f"learning_{building_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
    # Log parameters
    tracker.log_params({
        "building_id": building_id,
        "n_days": n_days,
        "training_days": len(training_days),
        "optimization_days": len(optimization_days),
        "mode": mode
    })
    
    # Log probability learning results
    if device_probs:
        for device, prob_data in device_probs.items():
            tracker.log_metrics({
                f"{device}_js_prior": prob_data.get('js_prior', 0),
                f"{device}_updates_count": prob_data.get('observation_count', 0)
            })
    
    # Log daily optimization results
    for i, result in enumerate(results):
        tracker.log_metrics({
            f"day_{i}_cost": result['total_cost'],
            f"day_{i}_savings_pct": result['savings_pct']
        })
    
    # Log artifacts
    tracker.log_artifacts("results/visualizations")
```

#### 1.3 Key Metrics to Track

**Optimization Performance:**
```python
optimization_metrics = {
    # Cost metrics
    "total_cost": float,
    "savings_eur": float,
    "savings_pct": float,
    "cost_per_kwh": float,
    
    # Solver metrics  
    "optimization_time_seconds": float,
    "solver_status": str,  # "Optimal", "Feasible", etc.
    "variables_count": int,
    "constraints_count": int,
    
    # Energy metrics
    "total_consumption_kwh": float,
    "battery_cycles": float,
    "ev_charging_kwh": float,
    
    # Quality metrics
    "convergence_achieved": bool,
    "constraint_violations": int
}
```

**Learning Performance (Pipeline B only):**
```python
learning_metrics = {
    # Per device
    f"{device}_js_divergence": float,
    f"{device}_learning_updates": int,
    f"{device}_convergence_rate": float,
    
    # Overall
    "devices_trained": int,
    "training_days": int,
    "avg_convergence": float
}
```

#### 1.4 Artifacts to Save

**Standard Artifacts:**
- `results/output/*.csv` - Performance metrics
- `results/visualizations/*.png` - Optimization plots  
- `results/figures/*.png` - Analysis charts
- Configuration files (as JSON)

### Phase 2: Simple Analysis (1 week)

#### 2.1 Basic MLflow UI Usage
```bash
# Launch MLflow UI to view experiments
cd /Users/kennethmwandingi/ems
mlflow ui --port 5000

# Access at http://localhost:5000
```

#### 2.2 Simple Comparison Queries
```python
# scripts/mlflow_analysis.py
import mlflow
import pandas as pd

def compare_experiments():
    """Simple analysis of MLflow experiments."""
    
    # Get all runs from Pipeline A
    experiment = mlflow.get_experiment_by_name("Pipeline_A_Comparison")
    runs = mlflow.search_runs(experiment.experiment_id)
    
    # Simple comparisons
    print("=== Pipeline A Results ===")
    print(f"Best savings: {runs['metrics.avg_savings_pct'].max():.2f}%")
    print(f"Worst savings: {runs['metrics.avg_savings_pct'].min():.2f}%")
    print(f"Average cost: ${runs['metrics.total_cost'].mean():.2f}")
    
    # Best performing run
    best_run = runs.loc[runs['metrics.avg_savings_pct'].idxmax()]
    print(f"Best run: {best_run['tags.mlflow.runName']}")
    print(f"Best building: {best_run['params.building_id']}")
    print(f"Best mode: {best_run['params.optimization_mode']}")
    
    return runs

if __name__ == "__main__":
    df = compare_experiments()
    df.to_csv("mlflow_analysis.csv", index=False)
```

### Phase 3: Simple Optimization Model (1 week)

#### 3.1 Package Optimization as MLflow Model
```python
# utils/mlflow_optimization_model.py
import mlflow.pyfunc
import pandas as pd
import json

class EMS_OptimizationModel(mlflow.pyfunc.PythonModel):
    """Simple MLflow model wrapper for EMS optimization."""
    
    def load_context(self, context):
        """Load optimization configuration."""
        config_path = context.artifacts["config"]
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def predict(self, context, model_input):
        """Run optimization with new parameters."""
        # model_input: DataFrame with columns like:
        # ['building_id', 'n_days', 'battery_enabled', 'ev_enabled', 'mode']
        
        results = []
        for _, row in model_input.iterrows():
            # Extract parameters
            building_id = row['building_id'] 
            n_days = row['n_days']
            mode = row['mode']
            battery_enabled = row.get('battery_enabled', True)
            
            # Run optimization (simplified)
            # In reality, this would call the actual pipeline
            result = {
                'building_id': building_id,
                'predicted_savings_pct': 8.5,  # Placeholder
                'predicted_cost': 150.0,       # Placeholder  
                'optimization_feasible': True
            }
            results.append(result)
        
        return pd.DataFrame(results)

def save_optimization_model(config_dict, run_id=None):
    """Save EMS optimization as MLflow model."""
    
    # Save configuration
    config_path = "optimization_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Log as MLflow model
    mlflow.pyfunc.log_model(
        artifact_path="ems_optimization_model",
        python_model=EMS_OptimizationModel(),
        artifacts={"config": config_path},
        conda_env={
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.8",
                "pip",
                {"pip": ["mlflow", "pandas", "numpy", "pulp"]}
            ]
        }
    )
```

#### 3.2 Model Usage Example
```python
# Example: Load and use optimization model
model_uri = "runs:/<RUN_ID>/ems_optimization_model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# New optimization requests
new_requests = pd.DataFrame({
    'building_id': ['DE_KN_residential1', 'DE_KN_residential2'],
    'n_days': [5, 3],
    'mode': ['centralized', 'centralized_phases'],
    'battery_enabled': [True, False]
})

predictions = loaded_model.predict(new_requests)
print(predictions)
```

## 🔧 Implementation Steps

### Week 1: Basic Setup
1. **Install MLflow**: `pip install mlflow`
2. **Add EMS_SimpleTracker** to both pipelines
3. **Test tracking** with one sample run per pipeline
4. **Verify MLflow UI** shows tracked experiments

### Week 2: Full Integration  
1. **Add comprehensive parameter logging** to both pipelines
2. **Add all key metrics** (cost, savings, solver performance)
3. **Add artifact logging** for results and visualizations
4. **Test with multiple buildings** and optimization modes

### Week 3: Analysis & Model
1. **Create analysis scripts** for comparing experiments
2. **Implement simple optimization model** wrapper
3. **Test model saving and loading**
4. **Document usage patterns**

## 📊 Expected Benefits

### Development Benefits
- **Experiment Comparison**: Easily compare optimization modes and buildings
- **Performance Tracking**: Monitor optimization quality over time
- **Result Archive**: Never lose optimization results again
- **Quick Access**: MLflow UI for browsing all experiments

### Operational Benefits  
- **Reproducibility**: Rerun exact optimization configurations
- **Sharing**: Share optimization results with stakeholders
- **Monitoring**: Track system performance trends
- **Debugging**: Identify when optimization quality degrades

## 🎯 Success Criteria

### Week 1 Success
- [ ] MLflow UI shows Pipeline A and B experiments
- [ ] All key parameters logged for each run
- [ ] Basic metrics captured (cost, savings, time)

### Week 2 Success  
- [ ] 10+ experiment runs tracked successfully
- [ ] All result artifacts saved and accessible
- [ ] No impact on pipeline execution time

### Week 3 Success
- [ ] Simple analysis script working
- [ ] Optimization model can be saved and loaded
- [ ] Documentation complete for team usage

## 📁 File Structure

```
ems/
├── mlflow_runs/           # Local MLflow tracking directory
├── utils/
│   ├── mlflow_simple.py   # Simple tracker class
│   └── mlflow_models.py   # Model wrapper utilities
├── scripts/
│   ├── mlflow_analysis.py # Analysis utilities
│   └── launch_mlflow.py   # UI launcher
└── README.md              # Updated with MLflow usage
```

## 💡 Key Design Principles

1. **Simplicity First**: No complex model versioning or deployment
2. **Local Storage**: File-based tracking, no server infrastructure  
3. **Non-Intrusive**: Minimal changes to existing pipeline code
4. **Immediate Value**: Useful experiment comparison from day one
5. **Easy Extension**: Foundation for future advanced features

This MVP approach provides immediate value for experiment tracking while keeping complexity minimal and implementation straightforward.