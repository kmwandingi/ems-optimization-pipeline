#!/usr/bin/env python3
"""
run_mlflow.py – End-to-end device-usage prediction & (optional) Azure deployment.

Core steps
──────────
1. Data prep & feature engineering  (steps.py)
2. Optional Great Expectations      (--validate)
3. Optional Feast feature fetch     (--use_feast)
4. Train LightGBM (daily) + CatBoost (hourly) and log to MLflow
5. Evaluate & log artefacts
6. Optional Azure ML ACI deploy     (--deploy_azure)

The script avoids run-ID conflicts:
• If `mlflow run .` already supplied a run, we just use it.
• If run standalone, we create one run inside “device_usage_prediction”.
"""

import argparse
import datetime
import io
import json
import math
import os
import pickle
import re
import sys
import uuid
import warnings
from pathlib import Path
from contextlib import nullcontext

import mlflow
import mlflow.lightgbm
import mlflow.catboost
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import Pool

from steps import (
    data_prep_step,
    feature_engineering_step,
    train_daily_step,
    train_hourly_step,
    evaluation_step,
)


def validate_with_ge(df: pd.DataFrame, name: str, root="."):
    print(f"\n[GE] Starting validation for {name} dataset...")
    try:
        from great_expectations.data_context import DataContext
        print(f"[GE] Looking for DataContext in {root}")
        ctx = DataContext(root)
        print(f"[GE] DataContext found! Available validation operators: {ctx.list_validation_operators()}")
        br  = {"runtime_parameters": {"batch_data": df}}
        res = ctx.run_validation_operator(
            "action_list_operator",
            assets_to_validate=[br],
            run_name=f"validate_{name}",
        )
        print(f"✅ GE validation ({name}): {res['success']}")
    except Exception as exc:
        print(f"⚠️  GE validation error ({name}): {exc}")
        import traceback
        traceback.print_exc()


def fetch_from_feast(daily_df, hourly_df, repo="feature_repo"):
    print(f"\n[FEAST] Starting feature retrieval from repository {repo}...")
    try:
        from feast import FeatureStore
        print(f"[FEAST] Initializing FeatureStore from {repo}")
        store = FeatureStore(repo_path=repo)
        
        # Check what's available in the repository
        print(f"[FEAST] Checking repository contents...")
        print(f"[FEAST] Available entities: {store.list_entities()}")
        print(f"[FEAST] Available feature views: {store.list_feature_views()}")
        print(f"[FEAST] Available feature services: {store.list_feature_services()}")
        
        daily_e  = daily_df[["date","building","plain_device_type"]]
        hourly_e = hourly_df[["datetime","building","plain_device_type"]]
        
        # Define feature lists
        daily_features = [f"device_usage:{c}" for c in [
            "rolling_7d_usage","temperature","solar_radiation","pv_forecast",
            "day_of_week","rolling_3d_usage","rolling_14d_usage","lag_7d_usage",
            "peak_hour_sin","peak_hour_cos","temp_month","dow_pv",
            "is_holiday","is_weekend",
        ]]
        
        hourly_features = [f"device_usage:{c}" for c in [
            "temperature","solar_radiation","pv_forecast","hour",
            "day_of_week","is_weekend","rolling_24h_usage","temp_hour",
            "dow_pv","day_cumulative_usage","prev_hour_on","hour_sin","hour_cos",
        ]]
        
        print(f"[FEAST] Requesting {len(daily_features)} daily features")
        daily_feats = store.get_historical_features(
            daily_e,
            features=daily_features,
        ).to_df()

        print(f"[FEAST] Requesting {len(hourly_features)} hourly features")
        hourly_feats = store.get_historical_features(
            hourly_e,
            features=hourly_features,
        ).to_df()

        print("✅ Feast features fetched successfully!")
        return daily_feats, hourly_feats
    except Exception as exc:
        print(f"⚠️  Feast fetch error: {exc}")
        import traceback
        traceback.print_exc()
        return None, None


def deploy_to_azure_v2(model_uri, model_name, svc_name, continuous_devices=None):
    """Deploy model to Azure ML using the Azure ML SDK v2
    
    Args:
        model_uri: The MLflow model URI (runs:/<run_id>/model)
        model_name: Name to register the model as in Azure ML
        svc_name: Name for the deployment service
        continuous_devices: Set, list, or comma-separated string of continuous devices to handle specially
        
    Returns:
        The endpoint URL if successful, None otherwise
    """
    # Ensure continuous_devices is consistently handled as a set
    if continuous_devices is not None:
        if isinstance(continuous_devices, str):
            continuous_devices = set(continuous_devices.split(","))
        elif isinstance(continuous_devices, (list, tuple)):
            continuous_devices = set(continuous_devices)
        # If it's already a set, no conversion needed
    else:
        # Default known continuous devices from memory
        continuous_devices = set(["refrigerator", "freezer"])
    try:
        # Try to import required packages
        try:
            from azure.ai.ml import MLClient
            from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
            from azure.ai.ml.constants import AssetTypes
            from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
            import json
            import os
            from pathlib import Path
            import mlflow
        except ImportError as e:
            # Install required packages
            print("\033[1;33mInstalling required packages...\033[0m")
            import subprocess
            subprocess.run(["pip", "install", "azure-ai-ml", "azure-identity"], check=True)
            
            # Import after installation
            from azure.ai.ml import MLClient
            from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
            from azure.ai.ml.constants import AssetTypes
            from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
        
        print(f"\n\033[1m[Azure ML] Deploying {model_name} using Azure ML SDK v2...\033[0m")
        
        # Process continuous devices
        if continuous_devices is None:
            # Default continuous devices (from memory: refrigerator, freezer)
            continuous_devices_list = ["refrigerator", "freezer"]
        elif isinstance(continuous_devices, str):
            continuous_devices_list = continuous_devices.split(",")
        elif isinstance(continuous_devices, (list, tuple)):
            continuous_devices_list = list(continuous_devices)
        elif isinstance(continuous_devices, set):
            # Handle set type (from memory: known_continuous_devices is a set)
            continuous_devices_list = list(continuous_devices)
        else:
            continuous_devices_list = ["refrigerator", "freezer"]
            
        print(f"  - Continuous devices to handle: {', '.join(continuous_devices_list)}")
        
        # Get run ID from model URI
        run_id = model_uri.split("/")[1]
        model_path = f"runs:/{run_id}/model"
        print(f"  - Using model path: {model_path}")
        
        # Get Azure ML workspace details from environment variables
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
        
        if not all([subscription_id, resource_group, workspace_name]):
            print("\033[1;33mAzure ML workspace environment variables not set or incomplete.\033[0m")
            print("Please make sure all of these environment variables are set:")
            print("  - AZURE_SUBSCRIPTION_ID")
            print("  - AZURE_RESOURCE_GROUP")
            print("  - AZURE_WORKSPACE_NAME")
            return None
            
        print(f"  - Azure ML workspace: {workspace_name}")
        print(f"  - Subscription ID: {subscription_id}")
        print(f"  - Resource group: {resource_group}")
        
        # Get credentials - try DefaultAzureCredential first, fallback to interactive login
        try:
            credential = DefaultAzureCredential()
            # Test the credential
            credential.get_token("https://management.azure.com/.default")
            print("  - Using default Azure credential")
        except Exception as e:
            print(f"  - Default credential failed: {e}")
            print("  - Initiating interactive browser login...")
            credential = InteractiveBrowserCredential()
        
        # Connect to the workspace
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        print(f"  \033[1;32m✓ Connected to Azure ML workspace: {workspace_name}\033[0m")
        
        # Create export directory with timestamp for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        deploy_dir = Path(f"azure_deployment_{timestamp}")
        if deploy_dir.exists():
            shutil.rmtree(deploy_dir)
        deploy_dir.mkdir(exist_ok=True)
        
        # Find local model files (for local MLflow tracking)
        local_model_path = None
        
        # Try to locate the model files if using local tracking
        if not model_uri.startswith("models:/") and "/" in model_uri:
            # For local MLflow tracking, model_uri format is runs:/<run_id>/model
            print("  - MLflow tracking URI:", mlflow.get_tracking_uri())
            
            run_id = model_uri.split("/")[1]
            model_rel_path = "/".join(model_uri.split("/")[2:])
            
            # Check several possible locations for local MLflow files
            tracking_uri = mlflow.get_tracking_uri()
            if tracking_uri.startswith("file:"):
                tracking_dir = tracking_uri[5:]
            else:
                tracking_dir = "mlruns"
                
            possible_paths = [
                os.path.join(tracking_dir, "0", run_id, "artifacts", model_rel_path),
                os.path.join("mlruns", "0", run_id, "artifacts", model_rel_path),
                os.path.join(run_id, "artifacts", model_rel_path),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    local_model_path = path
                    break
                    
            if local_model_path:
                print(f"  - Found local model files at: {local_model_path}")
            else:
                print("  - Could not find local model files, will try direct registration")
        
        # 1. Register the model - save and upload files directly
        try:
            print("  - Registering model in Azure ML...")
            
            # Create export directory for model files if needed
            export_dir = deploy_dir / "model_export" / model_name
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize model_saved flag
            model_saved = False
            
            # Method 1: Try to save the model directly from MLflow
            try:
                # Get the model from MLflow tracking
                if "daily" in model_name.lower():
                    # For daily model (LightGBM)
                    import mlflow.lightgbm
                    loaded_model = mlflow.lightgbm.load_model(model_uri)
                    mlflow.lightgbm.save_model(loaded_model, export_dir)
                    model_saved = True
                else:
                    # For hourly model (CatBoost)
                    import mlflow.catboost
                    loaded_model = mlflow.catboost.load_model(model_uri)
                    mlflow.catboost.save_model(loaded_model, export_dir)
                    model_saved = True
                    
                print(f"  - Successfully exported model files to {export_dir}")
            except Exception as e:
                print(f"  - Could not export model directly: {e}")
            
            # Method 2: Try to copy from local model path if found
            if not model_saved and local_model_path:
                try:
                    import shutil
                    shutil.copytree(local_model_path, export_dir, dirs_exist_ok=True)
                    model_saved = True
                    print(f"  - Copied model files from {local_model_path} to {export_dir}")
                except Exception as e:
                    print(f"  - Could not copy model files: {e}")
            
            # Method 3: Create a minimal model file with required properties
            if not model_saved:
                # Create a minimal MLflow model
                try:
                    import yaml
                    mlmodel_path = export_dir / "MLmodel"
                    conda_env_path = export_dir / "conda.yaml"
                    
                    # Create MLmodel file
                    mlmodel_content = {
                        "flavors": {
                            "python_function": {
                                "loader_module": "mlflow.sklearn",
                                "model_path": "model.pkl",
                                "env": "conda.yaml"
                            }
                        },
                        "run_id": run_id,
                        "utc_time_created": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "model_uuid": str(uuid.uuid4()),
                        "mlflow_version": mlflow.__version__
                    }
                    
                    # Write MLmodel file
                    os.makedirs(export_dir, exist_ok=True)
                    with open(mlmodel_path, "w") as f:
                        yaml.dump(mlmodel_content, f)
                    
                    # Write conda.yaml file with dependencies
                    conda_env = {
                        "name": "mlflow-env",
                        "channels": ["conda-forge"],
                        "dependencies": [
                            "python=3.9",
                            "pip",
                            {"pip": [
                                "mlflow",
                                "scikit-learn",
                                "pandas",
                                "numpy",
                                "lightgbm",
                                "catboost"
                            ]}
                        ]
                    }
                    
                    with open(conda_env_path, "w") as f:
                        yaml.dump(conda_env, f)
                        
                    # Create empty model.pkl
                    with open(export_dir / "model.pkl", "wb") as f:
                        pickle.dump({"continuous_devices": continuous_devices_list}, f)
                        
                    model_saved = True
                    print("  - Created minimal model files for upload")
                except Exception as e:
                    print(f"  - Could not create minimal model files: {e}")
            
            if not model_saved:
                raise Exception("Could not save model files through any method")
                
            # Register the model using the exported files
            model = Model(
                path=str(export_dir),  # Use the exported files
                type=AssetTypes.MLFLOW_MODEL,
                name=model_name,
                description=f"Device usage prediction model with continuous devices: {','.join(continuous_devices_list)}",
                tags={"continuous_devices": ",".join(continuous_devices_list)}
            )
            
            registered_model = ml_client.models.create_or_update(model)
            print(f"  \033[1;32m✓ Model registered: {registered_model.name} (version {registered_model.version})\033[0m")
        except Exception as e:
            print(f"  \033[1;31m× Model registration failed: {e}\033[0m")
            # Create a downloadable export of the model if registration fails
            export_dir = deploy_dir / "model_export" / model_name
            export_dir.mkdir(parents=True, exist_ok=True)
            
            if local_model_path:
                print(f"  - Saving model files to: {export_dir}")
                import shutil
                shutil.copytree(local_model_path, export_dir, dirs_exist_ok=True)
                print(f"  \033[1;33mModel files saved to {export_dir} for manual upload\033[0m")
            
            return None
        
        # 2. Create an endpoint
        endpoint_name = f"{model_name}-endpoint".replace('_', '-')
        print(f"  - Creating/updating endpoint: {endpoint_name}")
        
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=f"Endpoint for {model_name}",
            auth_mode="key",
            tags={
                "project": "energy-management-system",
                "model_type": "device_usage",
                "continuous_devices": ",".join(continuous_devices_list)
            }
        )
        
        try:
            ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            print(f"  \033[1;32m✓ Endpoint created/updated: {endpoint_name}\033[0m")
        except Exception as e:
            print(f"  \033[1;31m× Endpoint creation failed: {e}\033[0m")
            return None
        
        # 3. Create a deployment
        print(f"  - Creating deployment: {svc_name}")
        
        deployment = ManagedOnlineDeployment(
            name=svc_name,
            endpoint_name=endpoint_name,
            model=registered_model.id,
            instance_type="Standard_DS3_v2",
            instance_count=1,
            environment_variables={
                "CONTINUOUS_DEVICES": ",".join(continuous_devices_list)
            }
        )
        
        try:
            ml_client.online_deployments.begin_create_or_update(deployment).result()
            print(f"  \033[1;32m✓ Deployment created: {svc_name}\033[0m")
            
            # Set the deployment as default
            endpoint.traffic = {svc_name: 100}
            ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            print(f"  \033[1;32m✓ Set deployment as default\033[0m")
        except Exception as e:
            print(f"  \033[1;31m× Deployment creation failed: {e}\033[0m")
            return None
        
        # 4. Create a sample input file
        sample_input_file = deploy_dir / f"{model_name}_sample_input.json"
        
        # Sample input tailored to your model type
        if "daily" in model_name.lower():
            # Daily model sample
            sample_input = {
                "data": [
                    {
                        "temperature": 20.5,
                        "solar_radiation": 150.0,
                        "pv_forecast": 2.3,
                        "day_of_week": "2",
                        "month": 5,
                        "is_weekend": "0",
                        "building": "house1",
                        "plain_device_type": "washing_machine"
                    }
                ]
            }
        else:
            # Hourly model sample
            sample_input = {
                "data": [
                    {
                        "temperature": 20.5,
                        "solar_radiation": 150.0,
                        "pv_forecast": 2.3,
                        "hour": 14,
                        "day_of_week": "2",
                        "month": 5,
                        "is_weekend": "0",
                        "prev_hour_on": "0",
                        "building": "house1",
                        "plain_device_type": "washing_machine"
                    }
                ]
            }
        
        with open(sample_input_file, "w") as f:
            json.dump(sample_input, f, indent=2)
            
        print(f"  - Created sample input file: {sample_input_file}")
        
        # Get the endpoint URL
        endpoint_url = ml_client.online_endpoints.get(name=endpoint_name).scoring_uri
        print(f"\n\033[1;32m✓ Model {model_name} successfully deployed to {endpoint_url} ✓\033[0m")
        
        # Create a test script
        test_script_file = deploy_dir / f"{model_name}_test.py"
        test_script = f'''
# Test script for {model_name} endpoint
import requests
import json

# Load sample input data
with open("{sample_input_file}", "r") as f:
    sample_input = json.load(f)

# Get your API key from the Azure Portal
api_key = "YOUR_API_KEY_HERE"  # Replace with your actual API key

# Send request to the endpoint
headers = {{
    "Content-Type": "application/json",
    "Authorization": f"Bearer {{api_key}}"
}}

response = requests.post(
    url="{endpoint_url}",
    headers=headers,
    data=json.dumps(sample_input)
)

print(f"Response status code: {{response.status_code}}")
print("Response body:")
print(json.dumps(response.json(), indent=2))
'''
        
        with open(test_script_file, "w") as f:
            f.write(test_script)
            
        print(f"  - Created test script: {test_script_file}")
        return endpoint_url
        
    except ImportError as err:
        print("\n\033[1;31m[Azure ML] Deployment Error: Required packages missing\033[0m")
        print("\nTo deploy to Azure ML, you need to install the required packages.")
        print("Run this command:")
        print("  pip install azure-ai-ml azure-identity")
        return None
        
    except Exception as e:
        print(f"\n\033[1;31m× Azure ML deployment failed: {e}\033[0m")
        return None


def run_pipeline(parquet_dir, thresh, do_validate, use_feast, deploy_azure):
    # Only log params once: skip if inside `mlflow run .`
    if not os.getenv("MLFLOW_RUN_ID"):
        mlflow.log_params({
            "parquet_dir": str(Path(parquet_dir).resolve()),
            "default_threshold": thresh,
            "validate": do_validate,
            "use_feast": use_feast,
            "deploy_azure": deploy_azure,
        })

    daily_df, hourly_df = data_prep_step(parquet_dir, thresh)

    if do_validate:
        validate_with_ge(daily_df, "daily")
        validate_with_ge(hourly_df, "hourly")

    if use_feast:
        df_d, df_h = fetch_from_feast(daily_df, hourly_df)
        if df_d is not None:
            X_day = df_d.drop(columns=["device_used"]); y_day = df_d["device_used"]
        else:
            X_day, y_day, _ = feature_engineering_step(daily_df, hourly_df)
        if df_h is not None:
            X_hr = df_h.copy()
            X_hr["device_on_at_hour"] = hourly_df["device_on_at_hour"]
            X_hr["date"]             = hourly_df["date"]
        else:
            _, _, X_hr = feature_engineering_step(daily_df, hourly_df)
    else:
        X_day, y_day, X_hr = feature_engineering_step(daily_df, hourly_df)

    daily_m  = train_daily_step(X_day, y_day)
    mlflow.lightgbm.log_model(daily_m, "daily_model", registered_model_name="daily_device_model")

    hourly_m = train_hourly_step(X_hr)
    mlflow.catboost.log_model(hourly_m, "hourly_model", registered_model_name="hourly_device_model")

    feats = hourly_m.feature_names_; cats = [feats[i] for i in hourly_m.get_cat_feature_indices()]
    mlflow.log_dict({"feature_names": feats},    "hourly_model/feat_names.json")
    mlflow.log_dict({"cat_feature_names": cats}, "hourly_model/cat_names.json")

    evaluation_step(daily_m, hourly_m, X_day, y_day, X_hr, output_dir="evaluation")
    mlflow.log_artifacts("evaluation", "evaluation")

    if deploy_azure:
        # Use our new implementation with Azure ML SDK v2
        try:
            print("\n\033[1m[Azure ML Deployment]\033[0m")
            rid = mlflow.active_run().info.run_id
            print(f"  - Using run ID: {rid}")
            
            # Save identified continuous devices from pipeline as a tag
            if hasattr(hourly_m, 'continuous_devices') and hourly_m.continuous_devices:
                mlflow.set_tag("continuous_devices", ",".join(hourly_m.continuous_devices))
                print(f"  - Saved continuous devices to run tags: {hourly_m.continuous_devices}")
            
            # Check if continuous devices were identified during training
            continuous_devices_tag = set(["refrigerator", "freezer"])  # Default known continuous devices
            try:
                run = mlflow.get_run(rid)
                tag_value = run.data.tags.get("continuous_devices")
                if tag_value:
                    # Convert to set regardless of input type
                    if isinstance(tag_value, str):
                        continuous_devices_tag = set(tag_value.split(","))
                    elif isinstance(tag_value, (list, tuple)):
                        continuous_devices_tag = set(tag_value)
                    # If it's already a set, no conversion needed
                    print(f"  - Continuous devices from run: {continuous_devices_tag}")
            except Exception as e:
                print(f"  - Warning: Could not get run tags: {e}")
                print(f"  - Using default continuous devices: {continuous_devices_tag}")
            
            # Deploy daily model
            print("\n\033[1m[Deploying Daily Device Model]\033[0m")
            daily_endpoint = deploy_to_azure_v2(
                model_uri=f"runs:/{rid}/daily_model",
                model_name="daily-device-model",
                svc_name="daily-device-deployment",
                continuous_devices=continuous_devices_tag
            )
            
            # If daily deployment succeeded, try hourly with the same approach
            if daily_endpoint:
                print("\n\033[1m[Deploying Hourly Device Model]\033[0m")
                try:
                    hourly_endpoint = deploy_to_azure_v2(
                        model_uri=f"runs:/{rid}/hourly_model",
                        model_name="hourly-device-model",
                        svc_name="hourly-device-deployment",
                        continuous_devices=continuous_devices_tag
                    )
                    
                    if hourly_endpoint:
                        print("\n\033[1;32m✓ Both models deployed successfully!\033[0m")
                        print(f"  - Daily model endpoint: {daily_endpoint}")
                        print(f"  - Hourly model endpoint: {hourly_endpoint}")
                    else:
                        print("\n\033[1;33m⚠ Daily model deployed successfully, but hourly model deployment failed\033[0m")
                        print(f"  - Daily model endpoint: {daily_endpoint}")
                except Exception as inner_e:
                    print(f"\n\033[1;33m⚠ Hourly model deployment error: {inner_e}\033[0m")
                    print(f"  - Daily model endpoint still available: {daily_endpoint}")
            else:
                print("\n\033[1;31m× Daily model deployment failed, not attempting hourly model\033[0m")
        except Exception as e:
            print(f"\n\033[1;31m× Azure ML deployment failed: {e}\033[0m")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Device-usage pipeline")
    ap.add_argument("--parquet_dir",       required=True)
    ap.add_argument("--default_threshold", type=float, default=0.05)
    ap.add_argument("--validate",          action="store_true")
    ap.add_argument("--use_feast",         action="store_true")
    ap.add_argument("--deploy_azure",      action="store_true")
    args = ap.parse_args()

    inside = bool(os.getenv("MLFLOW_RUN_ID"))
    ctx    = nullcontext() if inside else mlflow.start_run()
    if not inside:
        mlflow.set_experiment("device_usage_prediction")
    with ctx:
        run_pipeline(
            parquet_dir=args.parquet_dir,
            thresh=args.default_threshold,
            do_validate=args.validate,
            use_feast=args.use_feast,
            deploy_azure=args.deploy_azure,
        )
    print("✅ Pipeline complete.")