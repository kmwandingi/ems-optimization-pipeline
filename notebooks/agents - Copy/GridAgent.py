import numpy as np
import pandas as pd
import logging
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from utils.config import GRID_PARAMS

###########################################################
#                   GridAgent
###########################################################

class GridAgent:
    def __init__(self, params: Dict[str, Any] = None):
        if params is None:
            params = GRID_PARAMS
        
        self.import_price = params.get('import_price', 0.0)
        self.export_price = params.get('export_price', 0.0)
        self.export_price_factor = params.get('export_price_factor', 0.9) # Default to 0.9 if not in config
        self.max_import = params.get('max_import', None)
        self.max_export = params.get('max_export', None)

    def get_grid_info(self) -> Dict[str, Any]:
        return {
            'import_price': self.import_price,
            'export_price': self.export_price,
            'max_import': self.max_import,
            'max_export': self.max_export
        }