import numpy as np
import pandas as pd
import logging

###########################################################
#                   GlobalConnectionLayer
###########################################################

class GlobalConnectionLayer:
    def __init__(self, max_building_load: float, total_hours: int, export_price: float = 0.0):
        self.max_building_load = max_building_load
        self.hourly_load = np.zeros(total_hours)
        self.total_hours = total_hours
        self.export_price = export_price
        # Keep a list of devices that register themselves with this layer
        self.devices: list = []

    def get_average_load(self, hours: np.ndarray) -> np.ndarray:
        """Return the average load for the given hours"""
        valid_hours = [hour for hour in hours if 0 <= hour < len(self.hourly_load)]
        return np.array([self.hourly_load[hour] if hour in valid_hours else 0.0 for hour in hours])

    def register_device(self, device) -> None:
        """Register a device with the connection layer.

        Currently this only stores the reference so that aggregate-load logic
        can be added later without touching call-sites.
        """
        self.devices.append(device)