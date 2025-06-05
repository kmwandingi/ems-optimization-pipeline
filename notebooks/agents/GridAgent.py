import numpy as np
import pandas as pd
import logging
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

###########################################################
#                   GridAgent
###########################################################

class GridAgent:
    def __init__(self, import_price: float, export_price: float, max_import: float = None, max_export: float = None):
        self.import_price = import_price
        self.export_price = export_price
        self.max_import = max_import
        self.max_export = max_export

    def get_grid_info(self) -> Dict[str, Any]:
        return {
            'import_price': self.import_price,
            'export_price': self.export_price,
            'max_import': self.max_import,
            'max_export': self.max_export
        }