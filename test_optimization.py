#!/usr/bin/env python
"""
Test optimization endpoint bypassing the FlexibleDevice issue
"""
import sys
import json
from pathlib import Path

# Add notebooks to path
sys.path.append(str(Path.cwd() / "notebooks"))
sys.path.append(str(Path.cwd() / "notebooks" / "utils"))

import mlflow
from azureml.core import Workspace

# Connect to Azure ML
with open('config.json') as f:
    config = json.load(f)

ws = Workspace(
    subscription_id=config['subscription_id'],
    resource_group=config['resource_group'],
    workspace_name=config['workspace_name']
)

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
print('✅ Connected to Azure ML MLflow')

# Load model
model = mlflow.pyfunc.load_model('models:/ems_complete_optimizer_DE_KN_residential1/6')
print('✅ Model loaded successfully')

# Test optimization with different date to avoid cached issues
print('\n🎯 Testing OPTIMIZATION mode...')
optimization_input = {
    'mode': 'optimize',
    'building_id': 'DE_KN_residential1',
    'target_date': '2015-05-24',  # Different date
    'price_profile': [0.22, 0.20, 0.18, 0.16, 0.15, 0.14, 0.17, 0.22, 
                     0.27, 0.25, 0.23, 0.21, 0.19, 0.17, 0.19, 0.22,
                     0.25, 0.29, 0.32, 0.30, 0.27, 0.25, 0.23, 0.21],
    'battery_enabled': True,
    'ev_enabled': False,
    'grid_params': {'import_price': 0.22, 'export_price': 0.04}
}

try:
    result = model.predict(optimization_input)
    print(f'✅ OPTIMIZATION SUCCESSFUL!')
    print(f'   Building: {result["building_id"]}')
    print(f'   Date: {result["target_date"]}')
    print(f'   Total cost: €{result["total_cost"]:.2f}')
    print(f'   Savings: €{result["savings_vs_baseline"]:.2f}')
    print(f'   Devices optimized: {len(result["optimized_schedules"])}')
    
    # Show device schedules
    print(f'\n📅 Optimized schedules:')
    for device_id, schedule in result["optimized_schedules"].items():
        total_energy = sum(schedule)
        active_hours = sum(1 for x in schedule if x > 0)
        print(f'   {device_id}: {total_energy:.1f}kWh over {active_hours}h')
        
except Exception as e:
    print(f'❌ OPTIMIZATION FAILED: {e}')
    import traceback
    traceback.print_exc()

print('\n🎉 OPTIMIZATION TEST COMPLETED!')