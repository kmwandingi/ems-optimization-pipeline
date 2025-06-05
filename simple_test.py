#!/usr/bin/env python
"""
Simple test to verify model loading and basic functionality
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

# Test learning mode first (simpler)
print('\n🧠 Testing LEARNING mode...')
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

result = model.predict(learning_input)
print(f'✅ Learning successful!')
print(f'   Building: {result["building_id"]}')
print(f'   Updated devices: {len(result["updated_devices"])}')
print(f'   Devices: {result["updated_devices"]}')

if 'updated_pmfs' in result:
    print(f'   Updated PMFs: {len(result["updated_pmfs"])} devices')
    for device_id, pmf in result['updated_pmfs'].items():
        max_hour = max(pmf.items(), key=lambda x: x[1])
        print(f'     {device_id}: peak at hour {max_hour[0]} ({max_hour[1]:.4f})')

print('\n🎉 LEARNING MODE TEST PASSED!')