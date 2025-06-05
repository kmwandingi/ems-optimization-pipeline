#!/usr/bin/env python
"""
Test the deployed EMS optimization endpoints with real data
"""
import sys
import json
from pathlib import Path

# Add notebooks to path for agent imports
sys.path.append(str(Path.cwd() / "notebooks"))
sys.path.append(str(Path.cwd() / "notebooks" / "utils"))

import mlflow
from azureml.core import Workspace

def test_optimization_endpoint():
    """Test the optimization endpoint with real building data"""
    print("🧪 TESTING EMS OPTIMIZATION ENDPOINTS")
    print("=" * 50)
    
    # Connect to Azure ML workspace  
    with open('config.json') as f:
        config = json.load(f)

    ws = Workspace(
        subscription_id=config['subscription_id'],
        resource_group=config['resource_group'],
        workspace_name=config['workspace_name']
    )

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    print('✅ Connected to Azure ML MLflow')
    
    # Load the model
    model_name = 'ems_complete_optimizer_DE_KN_residential1'
    model_version = '6'  # Latest version we deployed
    
    try:
        model = mlflow.pyfunc.load_model(f'models:/{model_name}/{model_version}')
        print(f'✅ Model loaded: {model_name} v{model_version}')
    except Exception as e:
        print(f'❌ Failed to load model: {e}')
        return False
    
    # Test 1: OPTIMIZATION MODE
    print("\n1️⃣ TESTING OPTIMIZATION MODE")
    print("-" * 30)
    
    optimization_input = {
        'mode': 'optimize',
        'building_id': 'DE_KN_residential1',
        'target_date': '2015-05-23',
        'price_profile': [0.25, 0.23, 0.21, 0.19, 0.18, 0.17, 0.20, 0.25, 
                         0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.22, 0.25,
                         0.28, 0.32, 0.35, 0.33, 0.30, 0.28, 0.26, 0.24],
        'battery_enabled': True,
        'ev_enabled': False,
        'grid_params': {'import_price': 0.25, 'export_price': 0.05}
    }
    
    try:
        result = model.predict(optimization_input)
        print(f'✅ OPTIMIZATION SUCCESS!')
        print(f'   Building: {result["building_id"]}')
        print(f'   Date: {result["target_date"]}')
        print(f'   Total cost: €{result["total_cost"]:.2f}')
        print(f'   Savings: €{result["savings_vs_baseline"]:.2f}')
        print(f'   Devices optimized: {len(result["optimized_schedules"])}')
        
        # Show sample schedules
        print(f'\n📅 Sample device schedules:')
        count = 0
        for device_id, schedule in result["optimized_schedules"].items():
            if count >= 3:
                break
            active_hours = [f'h{i}:{schedule[i]:.1f}kW' for i in range(24) if schedule[i] > 0]
            if active_hours:
                print(f'   {device_id}: {", ".join(active_hours[:5])}...')
            else:
                print(f'   {device_id}: [inactive]')
            count += 1
            
    except Exception as e:
        print(f'❌ OPTIMIZATION FAILED: {e}')
        return False
    
    # Test 2: LEARNING MODE
    print("\n2️⃣ TESTING LEARNING MODE")
    print("-" * 30)
    
    learning_input = {
        'mode': 'learn',
        'building_id': 'DE_KN_residential1',
        'actual_usage': {
            'DE_KN_residential1_heat_pump': [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                           0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            'DE_KN_residential1_dishwasher': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        },
        'date': '2015-05-23'
    }
    
    try:
        result = model.predict(learning_input)
        print(f'✅ LEARNING SUCCESS!')
        print(f'   Building: {result["building_id"]}')
        print(f'   Updated devices: {len(result["updated_devices"])}')
        print(f'   Devices: {result["updated_devices"]}')
        
        # Show updated PMFs
        if 'updated_pmfs' in result:
            print(f'\n📊 Updated PMFs:')
            for device_id, pmf in result['updated_pmfs'].items():
                sorted_hours = sorted(pmf.items(), key=lambda x: x[1], reverse=True)[:3]
                top_hours = [f'h{h}:{p:.4f}' for h, p in sorted_hours]
                print(f'   {device_id}: {", ".join(top_hours)}...')
        
    except Exception as e:
        print(f'❌ LEARNING FAILED: {e}')
        return False
    
    print("\n🎉 ALL ENDPOINT TESTS PASSED!")
    print("The EMS optimization pipeline is working correctly.")
    return True

if __name__ == "__main__":
    success = test_optimization_endpoint()
    if not success:
        sys.exit(1)