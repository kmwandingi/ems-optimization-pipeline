"""
Streamlit demo app for MILP Optimizer with onboarding system
"""

import streamlit as st
st.set_page_config(
    page_title="EMS Scheduler", 
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Remove default top padding
st.markdown("""
    <style>
        .block-container { padding-top: 0rem !important; }
    </style>
""", unsafe_allow_html=True)

# Import everything else after set_page_config
import pandas as pd
import numpy as np
import altair as alt
import json
import pandas as pd
import os
import sys
import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from streamlit.components.v1 import html

# Import custom draggable window component
from draggable_window import render_draggable_window

from savings import (
    calculate_schedule_cost,
    calculate_baseline_cost,
    update_savings_tracking,
)

# ─── Global Constants ────────────────────────────────────────────────────────
# Single source of truth for all device information
DEVICE_CATALOG = {
    "Appliances": {
        "icon": "fa-solid fa-blender-phone",
        "devices": {
            "washing_machine": {"icon": "👕", "energy": "0.5-1.5 kWh/cycle", "description": "Flexible load with multiple cycles"},
            "dishwasher": {"icon": "🍽️", "energy": "1-2 kWh/cycle", "description": "Flexible loads that can run any time"}
        }
    },
    "Temperature Control": {
        "icon": "fa-solid fa-thermometer-half",
        "devices": {
            "refrigerator": {"icon": "❄️", "energy": "0.5-1 kWh/day", "description": "Continuous operation with cycling"},
            "freezer": {"icon": "🧊", "energy": "0.8-1.2 kWh/day", "description": "Continuous operation with cycling"},
            "electric_heating": {"icon": "🔥", "energy": "Variable", "description": "Weather-dependent operation"}
        }
    },
    "Vehicles & Battery": {
        "icon": "fa-solid fa-car-battery",
        "devices": {
            "electric_vehicle": {"icon": "🚗", "energy": "10-20 kWh/charge", "description": "High capacity, flexible charging"},
            "battery": {"icon": "🔋", "energy": "N/A", "description": "Stores and discharges energy"}
        }
    }
}
# Create a flat list of all device names for easier access
ALL_DEVICES = [device for category in DEVICE_CATALOG.values() for device in category["devices"]]

# ─── Welcome Dialog ──────────────────────────────────────────────────────
@st.dialog("👋 Welcome to EMS Scheduler!")
def welcome_modal():
    """I show the first slide of the onboarding tour."""
    st.markdown("""
Welcome to **EMS Scheduler** – your personal energy co‑pilot.

    This tool helps you:
    • 🌱 reduce energy costs  
    • ⚡ optimise renewable usage  
    • 🔋 balance household load

**In the next 60 seconds you will…**

① Tell us which appliances you own  
② Show us the hour you *usually* start them  
③ Get tomorrow's cheapest schedule – and track the savings

Ready?
""")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Skip Tour"):
            st.session_state.onboarding_complete = True
            st.session_state.show_step = None
            st.rerun()
    with col2:
        if st.button("Start Tour", type="primary"):
            st.session_state.show_step = "device"
            st.rerun()

# ─── Device-help Dialog ──────────────────────────────────────────────────
@st.dialog("🔌 Tell Us About Your Devices")
def device_help_modal():
    """Interactive dialog to select devices and set baseline usage."""
    st.markdown("""
    **First, select the devices you own.**
    Then, for each device, tell us the hour you typically start using it. This helps us learn your habits!
    """)

    # Get current selections from session state
    selections = st.session_state.onboarding_selections

    # --- Device Button Grid ---
    st.markdown("**Select your devices:**")
    cols = st.columns(3) # Create a 3-column grid
    for i, device in enumerate(ALL_DEVICES):
        with cols[i % 3]:
            selected = device in selections["devices"]
            device_details = next((details for cat_info in DEVICE_CATALOG.values() for d, details in cat_info["devices"].items() if d == device), None)
            icon = device_details['icon'] if device_details else ''
            
            if st.button(f"{icon} {device.replace('_', ' ').title()}", key=f"onboarding_btn_{device}", use_container_width=True, type="primary" if selected else "secondary"):
                # This is the final fix: ensure we are always working with a list.
                if not isinstance(selections["devices"], list):
                    selections["devices"] = list(selections["devices"])

                if selected:
                    if device in selections["devices"]:
                        selections["devices"].remove(device)
                else:
                    if device not in selections["devices"]:
                        selections["devices"].append(device)
                st.rerun()

    selected_devices = list(selections["devices"]) # Ensure it's a list
    st.divider()

    # --- Sliders and Inputs for Selected Devices ---
    if selected_devices:
        st.markdown("**Set your habits and constraints for each device:**")
        for device in selected_devices:
            st.markdown(f"**{device.replace('_', ' ').title()}**")
            c1, c2 = st.columns(2)
            
            # Input for typical start time
            with c1:
                default_start = selections["hours"].get(device, 18)
                selections["hours"][device] = st.number_input(
                    "Typical start hour (0-23)", 
                    min_value=0, max_value=23, value=default_start, 
                    key=f"onboarding_start_{device}"
                )

            # Slider for allowed run hours
            with c2:
                default_constraints = st.session_state.device_constraints.get(device, {"earliest_hour": 8, "latest_hour": 22})
                earliest, latest = st.slider(
                    "Allowed run hours", 
                    0, 23, 
                    (default_constraints["earliest_hour"], default_constraints["latest_hour"]),
                    key=f"onboarding_range_{device}"
                )
                st.session_state.device_constraints[device] = {"earliest_hour": earliest, "latest_hour": latest}
                
            st.caption(
                "Tip – a **wider window** gives the optimiser more freedom, often saving more €."
            )
    
    # Navigation buttons pushed to opposite ends
    col1, _, col2 = st.columns([1, 3, 1])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.show_step = "welcome"
            st.rerun()
    with col2:
        if st.button("Next ➡️", use_container_width=True, type="primary"):
            # Store selections in session state
            st.session_state.onboarding_selections = selections
            st.session_state.selected_devices = list(selections["devices"])
            st.session_state.baseline_usage = selections["hours"]
            
            # Initialize the PMF for each selected device based on the user's typical start time
            for device, hour in selections["hours"].items():
                if hasattr(st.session_state.service, 'initialize_pmf_with_baseline'):
                    st.session_state.service.initialize_pmf_with_baseline(device, hour)

            # Clear the old schedule to force a regeneration on the next run
            st.session_state.schedule = {}

            st.session_state.show_step = "schedule"
            st.rerun()

# ─── Schedule-help Dialog ────────────────────────────────────────────────
@st.dialog("📊 Schedule Panel")
def schedule_help_modal():
    st.markdown("""
    **Inside the schedule panel** you'll:
    
    • See the optimiser's 24 h plan  
    • Drag the blue bar to log actual run-time  
    • Compare potential vs. actual savings
    """)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            st.session_state.show_step = "device"
            st.rerun()
    with col2:
        if st.button("Finish ✅", type="primary"):
            st.session_state.onboarding_complete = True
            st.session_state.show_step = None
            st.rerun()

# ─── Device-help Contextual Dialog ────────────────────────────────
@st.dialog("🔌 Device Selection Help")
def device_help_contextual():
    """Contextual help dialog for device selection"""
    st.markdown("""
    ## Device Selection Help
    
    **How to select devices:**
    1. Click on a device button to select it (it will change color)
    2. Use the slider to set allowed hours for the device to run
    3. Select multiple devices if needed
    
    **Device Categories:**
    - Kitchen appliances: dishwasher, refrigerator, etc.
    - Laundry: washing machine, dryer
    - HVAC: heat pump, air conditioner
    - Other: EV charger, water heater, etc.
    
    **Tips:**
    - Wider time windows give the optimizer more flexibility
    - Energy-intensive devices benefit most from optimization
    - Select devices that you can flexibly schedule
    """)
    
    st.button("Close", type="primary")

# ─── Schedule-help Contextual Dialog ───────────────────────────
@st.dialog("📊 Schedule Panel Help")
def schedule_help_contextual():
    """Contextual help dialog for schedule panel"""
    st.markdown("""
    ## Schedule Panel Help
    
    **Understanding the schedule:**
    - Blue bars show optimized run times for each device
    - Darker blue indicates higher energy consumption
    - Price curve (top row) shows electricity prices throughout the day
    
    **Recording actual usage:**
    - Drag the confirmation bar to indicate when you actually ran the device
    - This helps track actual vs. potential savings
    - Your energy savings are updated based on actual usage patterns
    
    **Savings Information:**
    - Potential savings: If you follow all optimized schedules perfectly
    - Actual savings: Based on your recorded usage patterns
    - Cumulative savings tracked across all days
    """)
    
    st.button("Close", type="primary")

# Add project root to path to enable imports
# Set up sys.path for proper imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import our mock implementation directly
# When running as a script, we need to import from the same directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Setup paths for imports
app_root = project_root / 'app'
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

# First try to import the RealOptimizationService adapter
try:
    # This adapter handles the integration with the real optimization service
    from real_optimization_service import RealOptimizationService
    print("Successfully imported real optimization service adapter")
    # Check if it can access the real optimization components
    from real_optimization_service import REAL_COMPONENTS_AVAILABLE
    USING_REAL_SERVICE = REAL_COMPONENTS_AVAILABLE
    print(f"Real optimization components available: {REAL_COMPONENTS_AVAILABLE}")
except Exception as e:
    print(f"Failed to import real optimization service: {e}")
    USING_REAL_SERVICE = False

# Always import the mock service as fallback
try:
    from mock_optimization_service import MockOptimizationService
    print("Successfully imported mock optimization service")
except ImportError:
    print("Failed to import MockOptimizationService. Check import paths.")
    sys.exit(1)

class OptimisationService:
    """
    Wrapper service for the OptimizationService that handles:
    - Device constraint configuration
    - Next-day schedule generation
    - Actual usage submission
    - Schedule history management
    """
    
    def __init__(self):
        """Initialize the optimization service"""
        self.schedules_dir = Path("schedules")
        self.schedules_dir.mkdir(exist_ok=True)
        
        # Track which service we're using for UI messaging
        self.using_real_service = False
        
        # Try to use the real service if available
        if USING_REAL_SERVICE:
            try:
                # Import should already be handled at module level
                self.service = RealOptimizationService()
                self.using_real_service = True
                print("Initialized with REAL optimization service and probability model")
            except Exception as e:
                print(f"Failed to initialize real service: {e}")
                print("Falling back to mock service")
                self.service = MockOptimizationService()
        else:
            # Use mock service if real components not available
            self.service = MockOptimizationService()
            print("Initialized with mock optimization service")
        
    def next_day(self, building_id: str, device_constraints: Dict, baseline_usage: Dict):
        """
        Calculate the next day schedule for a building.

        Args:
            building_id: ID of the building
            device_constraints: Dictionary mapping device names to constraints
            baseline_usage: Dictionary of baseline start hours for devices

        Returns:
            Tuple of (schedule_dict, price_curve)
        """
        # Use current date for optimization
        target_date = date.today()

        # If we're using the mock service, leverage its purpose-built next_day
        if not self.using_real_service and hasattr(self.service, "next_day"):
            # Get schedule and price curve from mock service
            schedule, price_curve = self.service.next_day(building_id, device_constraints, baseline_usage)

            # Persist to file for history / debugging consistency
            schedule_file = self.schedules_dir / f"{building_id}_{target_date.isoformat()}.json"
            with open(schedule_file, "w") as f:
                # Add price curve to the schedule data for historical reference
                schedule_data = {
                    "devices": schedule,
                    "price_curve": price_curve
                }
                json.dump(schedule_data, f, indent=2)

            return schedule, price_curve

        # Fallback for real service or if mock service is missing the method
        # This part would contain the logic for the real optimization service
        # For now, it returns an empty schedule and price curve as a placeholder
        print("Falling back to default schedule generation.")
        return {}, []

        result = {}
        
        # Get price curve from the real service
        # For now, we'll create a reasonable mock price curve
        # This should be replaced with real price data when available
        base_price = 0.26  # Base price in €/kWh
        price_curve = []
        for hour in range(24):
            if 0 <= hour < 6:  # Nighttime hours (cheaper)
                price_curve.append(base_price - 0.10)
            elif 18 <= hour < 21:  # Evening peak hours (more expensive)
                price_curve.append(base_price + 0.15)
            else:  # Regular daytime hours
                price_curve.append(base_price)
        
        # Get device schedules
        for service in selected_services:
            # Call appropriate optimization function
            if service == "battery":
                soc_df = battery_agent.get_soc(target_date)
                result["battery_soc"] = soc_df["soc"].tolist()
            else:
                # Call service specific optimization
                schedule = self.service.optimize(building_id, service, target_date)
                
                # Extract device schedules
                for device, values in self._extract_device_schedules(schedule, service).items():
                    if device in device_constraints:
                        result[device] = values
        
        # Persist to file for history/debugging consistency
        schedule_file = self.schedules_dir / f"{building_id}_{target_date.isoformat()}.json"
        with open(schedule_file, "w") as f:
            # Save both schedule and price curve
            schedule_data = {
                "devices": result,
                "price_curve": price_curve
            }
            json.dump(schedule_data, f, indent=2)
        
        # Return both schedule and price curve
        return result, price_curve
    
    def update_with_actuals(self, date_str: str, actual_usage: Dict[str, List[float]]) -> None:
        """Update the model with actual usage data
        
        Args:
            date_str: Date string in ISO format
            actual_usage: Dictionary mapping device names to 24-hour usage arrays
        """
        # Save actuals to file
        actuals_file = self.schedules_dir / f"{date_str}_actuals.json"
        with open(actuals_file, "w") as f:
            json.dump(actual_usage, f, indent=2)
            
        # Pass to service
        self.service.update_with_actuals(date_str, actual_usage)
        
    def initialize_pmf_with_baseline(self, device: str, start_hour: int):
        """Initialize the PMF for a device with a baseline start hour."""
        if hasattr(self.service, 'initialize_pmf_with_baseline'):
            self.service.initialize_pmf_with_baseline(device, start_hour)
        else:
            print(f"Service {type(self.service).__name__} does not support PMF initialization.")

    def get_device_pmf(self, device_name: str) -> Dict[str, List]:
        """Get the probability mass function for a device
        
        Args:
            device_name: Name of the device
            
        Returns:
            Dictionary with PMF data for visualization
        """
        # Delegate to the mock service
        return self.service.get_device_pmf(device_name)

    def get_schedule_history(self) -> List[Dict[str, Any]]:
        """
        Get list of all schedules generated
        
        Returns:
            List of schedule metadata
        """
        history = []
        
        if not self.schedules_dir.exists():
            return history
            
        for file in self.schedules_dir.glob("*.json"):
            if "_actuals" in file.name:
                continue
                
            # Parse building_id and date from filename
            parts = file.stem.split("_")
            if len(parts) >= 2:
                building_id = parts[0]
                date_str = parts[1]
                
                # Add to history
                history.append({
                    "building_id": building_id,
                    "date": date_str,
                    "file": str(file)
                })
                
        return sorted(history, key=lambda x: x["date"], reverse=True)


# Initialize session state
def init_session_state() -> None:
    """Initialize Streamlit session state with default values"""
    # Core app state
    if "service" not in st.session_state:
        st.session_state.service = OptimisationService()
    if "building_id" not in st.session_state:
        st.session_state.building_id = "default_building"
    if "devices" not in st.session_state:
        # Default devices
        st.session_state.devices = [
            "dishwasher", "washing_machine", "tumble_dryer", 
            "water_heater", "heat_pump", "refrigerator", "freezer"
        ]
    if 'selected_devices' not in st.session_state:
        st.session_state.selected_devices = [] # ALWAYS use a list for consistency
    # Ensure it's a list even if it exists from a previous session
    if not isinstance(st.session_state.selected_devices, list):
        st.session_state.selected_devices = list(st.session_state.selected_devices)
    if "device_constraints" not in st.session_state:
        st.session_state.device_constraints = {}
    if "schedule" not in st.session_state:
        st.session_state.schedule = {}
    if "actual_usage" not in st.session_state:
        st.session_state.actual_usage = {}
    if "current_day" not in st.session_state:
        st.session_state.current_day = 1
    if "current_date" not in st.session_state:
        st.session_state.current_date = date.today()
        
    # Savings tracking
    if "total_potential_savings" not in st.session_state:
        st.session_state.total_potential_savings = 0.0  # Savings if schedule is followed perfectly
    if "total_actual_savings" not in st.session_state:
        st.session_state.total_actual_savings = 0.0     # Actual savings based on user behavior
    if "daily_savings" not in st.session_state:
        st.session_state.daily_savings = {}             # Track daily savings by date
        
    # Onboarding flow state
    if "first_visit" not in st.session_state:
        st.session_state.first_visit = True
    if "onboarding_complete" not in st.session_state:
        st.session_state.onboarding_complete = False
    if "onboarding_step" not in st.session_state:
        st.session_state.onboarding_step = 1
    if "show_welcome_modal" not in st.session_state:
        st.session_state.show_welcome_modal = True
    if "show_device_help" not in st.session_state:
        st.session_state.show_device_help = False
    if "show_schedule_help" not in st.session_state:
        st.session_state.show_schedule_help = False
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    if "baseline_usage" not in st.session_state:
        st.session_state.baseline_usage = {}
    if 'onboarding_selections' not in st.session_state:
        st.session_state.onboarding_selections = {
            "devices": st.session_state.get("selected_devices", []), # Initialize with existing or empty list
            "hours": st.session_state.get("baseline_usage", {})
        }
    
    # PMF refresh flags
    if "pmf_refresh_needed" not in st.session_state:
        st.session_state.pmf_refresh_needed = False
    if "pmf_refresh_device" not in st.session_state:
        st.session_state.pmf_refresh_device = None


# Add project root to path to enable imports



def render_feedback_system() -> None:
    """Render feedback form in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📝 Feedback")
        
        if st.session_state.feedback_submitted:
            st.success("Thank you for your feedback!")
        else:
            with st.form("feedback_form"):
                st.write("Help us improve the EMS Scheduler!")
                
                # Rating
                rating = st.slider("How would you rate your experience?", 1, 5, 3, help="1 = Poor, 5 = Excellent")
                
                # Feedback categories
                feedback_category = st.selectbox(
                    "What area would you like to give feedback on?", 
                    ["User Interface", "Scheduling Algorithm", "Device Controls", "Energy Savings", "Other"]
                )
                
                # Comments
                comments = st.text_area("Comments or suggestions:", height=100)
                
                # Optional email
                email = st.text_input("Email (optional, for follow-up)", "")
                
                # Submit button
                submit_button = st.form_submit_button("Submit Feedback")
                
                if submit_button:
                    # Here you would typically save the feedback to a database
                    # For now, we'll just show a success message
                    st.session_state.feedback_submitted = True
                    st.success("Thank you for your feedback!")
                    st.rerun()


def render_header() -> None:
    """Render the app header with gradient styling using the palette colors"""
    st.markdown(
        f"""
        <style>
        .header-gradient {{            
            background: linear-gradient(135deg, #00838f 0%, #0e6072 100%);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            color: white;
            box-shadow: 0 4px 6px rgba(20, 43, 66, 0.1);
        }}
        .header-gradient h1 {{            
            margin: 0;
            font-size: 2rem;
        }}
        .header-gradient p {{            
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }}

        </style>
        <div class='header-gradient'>
            <h1>MILP Optimizer Demo - Day {st.session_state.current_day}</h1>
            <p>Interactive energy management scheduling demo - {st.session_state.current_date.strftime('%A, %B %d, %Y')}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )


def device_button(device: str, selected: bool) -> bool:
    """
    Render a device button with the given name and selection state
    
    Args:
        device: Device name
        selected: Whether the device is currently selected
        
    Returns:
        Whether the button was clicked
    """
    if selected:
        # For selected devices, use a different button style
        clicked = st.button(
            f"✓ {device.replace('_', ' ').title()}",
            key=f"device_{device}",
            use_container_width=True,
            type="primary"
        )
    else:
        # For unselected devices, use regular button
        clicked = st.button(
            device.replace("_", " ").title(),
            key=f"device_{device}",
            use_container_width=True,
        )
    
    return clicked


def toggle_device(device: str):
    """Toggle a device's selection status
    
    Args:
        device: The device to toggle
    """
    if device in st.session_state.selected_devices:
        st.session_state.selected_devices.remove(device)
    else:
        st.session_state.selected_devices.append(device)
        # Initialize constraints if not already set
        if device not in st.session_state.device_constraints:
            st.session_state.device_constraints[device] = {
                "earliest_hour": 0,
                "latest_hour": 23
            }


def render_device_picker() -> None:
    """Render the device picker panel with categorized devices and energy usage info"""
    # Header with Generate Schedule button and help button
    col1, col2, col3 = st.columns([2, 1.5, 0.5])
    with col1:
        st.subheader("Select Devices")

    with col2:
        # Only enable the button if devices are selected
        has_devices = len(st.session_state.selected_devices) > 0
        if st.button(
            "Generate", 
            key="generate_schedule_btn", 
            type="primary", 
            use_container_width=True,
            disabled=not has_devices
        ):
            generate_schedule()
    with col3:
        if st.button("❓", key="device_help_btn"):
            device_help_contextual()
    
    # Show a message about selected devices
    if st.session_state.selected_devices:
        selected_names = [dev.replace("_", " ").title() for dev in st.session_state.selected_devices]
        st.success(f"Selected: {', '.join(selected_names)}")
    else:
        st.info("No devices selected. Click on one or more devices below.")
    
    # Render all devices from the global catalog in a single list
    for category, info in DEVICE_CATALOG.items():
        for device, details in info["devices"].items():
                icon = details["icon"]
                energy = details["energy"]
                description = details["description"]
                
                selected = device in st.session_state.selected_devices
                
                # Use a container for each device row to manage layout
                device_container = st.container()
                
                # Apply CSS for buttons and sliders using palette colors
                st.markdown("""
                <style>
                /* Custom class to vertically align button and slider */
                .vertical-align-container {
                    display: flex;
                    align-items: center; /* This vertically aligns the content */
                    height: 100%;
                }

                /* Button styling with palette colors */
                .stButton button {    
                    white-space: nowrap;
                    overflow: visible;
                    text-overflow: clip;
                }
                /* Primary button (selected device) */
                .stButton button[kind="primary"] {
                    background-color: #00838f !important;
                    border-color: #00838f !important;
                    color: white !important;
                }
                .stButton button[kind="primary"]:hover {
                    background-color: #0e6072 !important;
                    border-color: #0e6072 !important;
                }
                /* Secondary button (unselected device) */
                .stButton button:not([kind="primary"]) {
                    border-color: #142b42 !important;
                    color: #142b42 !important;
                }
                .stButton button:not([kind="primary"]):hover {
                    background-color: rgba(244, 169, 138, 0.1) !important;
                    border-color: #f4a98a !important;
                }
                /* Generate schedule button */
                button[data-testid="baseButton-primary"] {
                    background-color: #00838f !important;
                    border-color: #00838f !important;
                }
                button[data-testid="baseButton-primary"]:hover {
                    background-color: #0e6072 !important;
                    border-color: #0e6072 !important;
                }
                
                /* Slider styling with palette colors */
                .stSlider [data-baseweb="slider"] div[role="slider"] {
                    background-color: #00838f !important;
                    border-color: #00838f !important;
                }
                .stSlider [data-baseweb="slider"] div[role="slider"]:hover {
                    background-color: #0e6072 !important;
                    border-color: #0e6072 !important;
                }
                .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {
                    color: #142b42 !important;
                    font-weight: bold;
                }
                .stSlider [data-baseweb="slider"] div[class$="Track"] div {
                    background-color: rgba(0, 131, 143, 0.4) !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Render the device button and constraints in a single row
                with device_container:
                    # Get current constraints
                    constraints = st.session_state.device_constraints.get(device, {"earliest_hour": 0, "latest_hour": 23})
                    
                    # If the device is selected, show button and slider on one row
                    if selected:
                        # Create columns for button (wider) and slider (narrower)
                        btn_col, slider_col = st.columns([3, 4], vertical_alignment="center")
                        
                        # Create a unique key for this device button
                        button_key = f"device_{device}"
                        
                        # Create the button with icon (no info icon) and handle click via on_click callback
                        # Use CSS to ensure text doesn't wrap
                        with btn_col:
                            if st.button(
                                f"✓ {icon} {device.replace('_', ' ').title()}",
                                key=button_key,
                                use_container_width=True,
                                type="primary",
                                on_click=toggle_device,
                                args=(device,)
                            ):
                                pass  # The on_click handles the action
                                
                        # Use a range slider for allowed hours in the same row
                        with slider_col:
                            hours_range = st.slider(
                                "",  # No label needed since it's next to the button
                                min_value=0,
                                max_value=23,
                                value=(constraints["earliest_hour"], constraints["latest_hour"]),
                                key=f"hours_range_{device}",
                                help=f"Set when this device is allowed to run. Current: {constraints['earliest_hour']:02d}:00 to {constraints['latest_hour']:02d}:59"
                            )
                        
                        # Update constraints
                        st.session_state.device_constraints[device] = {
                            "earliest_hour": hours_range[0],
                            "latest_hour": hours_range[1]
                        }
                        
                    else:
                        # For unselected devices, just show the button
                        # Create a unique key for this device button
                        button_key = f"device_{device}"
                        
                        # Create the button with icon
                        # Use CSS to ensure text doesn't wrap
                        if st.button(
                            f"{icon} {device.replace('_', ' ').title()}",
                            key=button_key,
                            use_container_width=True,
                            type="secondary",
                            on_click=toggle_device,
                            args=(device,)
                        ):
                            pass  # The on_click handles the action
                        
                        # st.markdown("<hr style='margin: 5px 0px; border-width: 1px;'>", unsafe_allow_html=True)


def generate_schedule() -> None:
    """Generate a new schedule using the optimization service"""
    # Check if any devices are selected
    if not st.session_state.selected_devices:
        st.toast("Please select at least one device first.", icon="⚠️")
        return
        
    # Filter device constraints to only include selected devices
    selected_device_constraints = {}
    for device in st.session_state.selected_devices:
        if device in st.session_state.device_constraints:
            selected_device_constraints[device] = st.session_state.device_constraints[device]
    
    with st.spinner("Generating schedule..."):
        try:
            # Call the optimization service which now returns both schedule and price curve
            schedule, price_curve = st.session_state.service.next_day(
                st.session_state.building_id,
                selected_device_constraints,
                st.session_state.baseline_usage
            )
            # Update session state with schedule and price curve
            st.session_state.schedule = schedule
            st.session_state.price_curve = price_curve
            
            # CRITICAL FIX: Initialize actual_usage to be identical to the optimized schedule.
            # This ensures that if the user doesn't interact, actual_usage is correct.
            # The draggable window logic will then correctly modify this baseline.
            st.session_state.actual_usage = { 
                device: usage for device, usage in schedule.items() 
                if device != "battery_soc" 
            }
            # Reset toggle states for a fresh day
            st.session_state.toggle_states = {device: {f"{device}_{h}_toggle": False for h in range(24)} for device in st.session_state.actual_usage}
            # Reset submission validation flag
            st.session_state.last_action_validated = False
            
            st.toast("Schedule generated successfully!", icon="✅")
            
        except Exception as e:
            st.toast(f"Error generating schedule: {str(e)}", icon="❌")
            import traceback
            st.code(traceback.format_exc())


def hour_cell(hour: int, device: str, value: float, is_actual: bool = False, max_val: float = 1.0) -> None:
    """
    Render an hour cell with color intensity based on kWh value and PMF data
    
    Args:
        hour: Hour (0-23)
        device: Device name
        value: kWh value
        is_actual: Whether this is for actual usage or scheduled usage
        max_val: Maximum value in the row for normalization
    """
    # Ensure value is a number and convert to float
    try:
        value = float(value)
    except (TypeError, ValueError):
        print(f"Warning: Invalid value for {device} at hour {hour}: {value}, using 0.0")
        value = 0.0
    
    # Get PMF data for this device if available and determine intensity from it
    pmf_intensity = 0
    if not is_actual and device in st.session_state.selected_devices:
        try:
            # Get the PMF data for this device
            pmf_data = st.session_state.service.get_device_pmf(device)
            
            # PMF data is in 2-hour blocks, map hour (0-23) to block index (0-11)
            block_idx = hour // 2
            
            if 'current_probabilities' in pmf_data and len(pmf_data['current_probabilities']) > block_idx:
                # Get probability for this time block (0.0-1.0)
                probability = pmf_data['current_probabilities'][block_idx]
                
                # Scale to intensity 0-100 and round to nearest 10
                pmf_intensity = min(int(probability * 200 / 10) * 10, 100)
                
                # Print for debugging
                print(f"PMF for {device} at hour {hour} (block {block_idx}): {probability} -> intensity {pmf_intensity}")
        except Exception as e:
            print(f"Error getting PMF data for {device}: {e}")
    
    # Use either PMF intensity or value intensity, depending on the context
    if is_actual:
        # For actual usage, just use the value intensity as before
        # Scale color intensity relative to the row maximum
        max_value = max_val if max_val > 0 else 1.0
        intensity = min(int(value / max_value * 100 / 10) * 10, 100)
        color_class = "actual"
    else:
        # For all non-actual-usage hours (scheduled or not), use PMF-based coloring.
        # The "schedule" is now indicated by the overlay border, not cell color.
        intensity = pmf_intensity
        if value > 0:
            # Optionally, can still boost intensity slightly for scheduled hours
            # to make them stand out in the heatmap, but without changing the color class.
            intensity = min(intensity + 30, 100)
        
        # Always use the 'pmf' color class for the heatmap effect.
        color_class = "pmf"
    
    # Determine cell label based on hour
    hour_str = f"{hour:02d}"
    
    # Add CSS for the PMF intensity classes if not already added
    if not hasattr(st.session_state, "pmf_css_added"):
        st.markdown("""
        <style>
        /* PMF-based hour cell styling using golden yellow from palette */
        .hour-cell.pmf-0 { background-color: rgba(244, 192, 109, 0.05); color: #142b42; }
        .hour-cell.pmf-10 { background-color: rgba(244, 192, 109, 0.1); color: #142b42; }
        .hour-cell.pmf-20 { background-color: rgba(244, 192, 109, 0.2); color: #142b42; }
        .hour-cell.pmf-30 { background-color: rgba(244, 192, 109, 0.3); color: #142b42; }
        .hour-cell.pmf-40 { background-color: rgba(244, 192, 109, 0.4); color: #142b42; }
        .hour-cell.pmf-50 { background-color: rgba(244, 192, 109, 0.5); color: #142b42; }
        .hour-cell.pmf-60 { background-color: rgba(244, 192, 109, 0.6); color: #142b42; }
        .hour-cell.pmf-70 { background-color: rgba(244, 192, 109, 0.7); color: #142b42; }
        .hour-cell.pmf-80 { background-color: rgba(244, 192, 109, 0.8); color: #142b42; }
        .hour-cell.pmf-90 { background-color: rgba(244, 192, 109, 0.9); color: #142b42; }
        .hour-cell.pmf-100 { background-color: rgba(244, 192, 109, 1.0); color: #142b42; font-weight: bold; }
        
        /* Schedule classes - NO BACKGROUND COLOR, only default styling */
        .hour-cell.schedule-0 { border: 1px solid rgba(0,0,0,.05); background-color: transparent; color: #142b42; }
        .hour-cell.schedule-10 { border: 1px solid rgba(0,0,0,.05); background-color: transparent; color: #142b42; }
        .hour-cell.schedule-20 { border: 1px solid rgba(0,0,0,.05); background-color: transparent; color: #142b42; }
        .hour-cell.schedule-30 { border: 1px solid rgba(0,0,0,.05); background-color: transparent; color: #142b42; }
        .hour-cell.schedule-40 { border: 1px solid rgba(0,0,0,.05); background-color: transparent; color: #142b42; }
        .hour-cell.schedule-50 { border: 1px solid rgba(0,0,0,.05); background-color: transparent; color: #142b42; }
        .hour-cell.schedule-60 { border: 1px solid rgba(0,0,0,.05); background-color: transparent; color: #142b42; }
        .hour-cell.schedule-70 { border: 1px solid rgba(0,0,0,.05); background-color: transparent; color: #142b42; }
        .hour-cell.schedule-80 { border: 1px solid rgba(0,0,0,.05); background-color: transparent; color: #142b42; }
        .hour-cell.schedule-90 { border: 1px solid rgba(0,0,0,.05); background-color: transparent; color: #142b42; }
        .hour-cell.schedule-100 { border: 1px solid rgba(0,0,0,.05); background-color: transparent; color: #142b42; }
        
        /* Actual usage classes */
        .hour-cell.actual-0 { background-color: rgba(136, 35, 62, 0.05); color: #142b42; }
        .hour-cell.actual-10 { background-color: rgba(136, 35, 62, 0.1); color: #142b42; }
        .hour-cell.actual-20 { background-color: rgba(136, 35, 62, 0.2); color: #142b42; }
        .hour-cell.actual-30 { background-color: rgba(136, 35, 62, 0.3); color: #142b42; }
        .hour-cell.actual-40 { background-color: rgba(136, 35, 62, 0.4); color: #142b42; }
        .hour-cell.actual-50 { background-color: rgba(136, 35, 62, 0.5); color: #142b42; }
        .hour-cell.actual-60 { background-color: rgba(136, 35, 62, 0.6); color: white; }
        .hour-cell.actual-70 { background-color: rgba(136, 35, 62, 0.7); color: white; }
        .hour-cell.actual-80 { background-color: rgba(136, 35, 62, 0.8); color: white; }
        .hour-cell.actual-90 { background-color: rgba(136, 35, 62, 0.9); color: white; }
        .hour-cell.actual-100 { background-color: rgba(136, 35, 62, 1.0); color: white; font-weight: bold; }
        
        /* Basic hour cell styling */
        .hour-cell {
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            font-size: 0.75rem;
            height: 28px;
            margin: 2px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        st.session_state.pmf_css_added = True
    
    # Create a colored cell with the proper color intensity
    st.markdown(
        f"""<div class="hour-cell {color_class}-{intensity}" title="{device}: {value:.2f} kWh at hour {hour}">
            {hour_str}
        </div>""", 
        unsafe_allow_html=True
    )
# ─────────────────────────────────────────────────────────────────────────────
# helper – one device row (schedule cells + draggable window, perfectly aligned)
# ─────────────────────────────────────────────────────────────────────────────
from streamlit.components.v1 import html as _components_html  # local alias

def _render_device_row_html(device: str, values, duration: int, start_idx: int):
    """
    Paints 24 schedule cells and adds schedule overlay blocks **inside the same iframe** that hosts
    the draggable confirmation bar, ensuring perfect alignment.
    Returns the updated start-hour (int) or None (older Streamlit versions).
    """
    dom = device.replace(" ", "_")
    
    # Find contiguous scheduled blocks (cells with non-zero values)
    scheduled_blocks = []
    block_start = None
    for h in range(24):
        if float(values[h]) > 0:
            if block_start is None:
                block_start = h
        elif block_start is not None:
            scheduled_blocks.append((block_start, h - 1))
            block_start = None
    # Handle case where last block extends to hour 23
    if block_start is not None:
        scheduled_blocks.append((block_start, 23))
    
    # Get PMF data for coloring cells based on usage probability
    pmf_data = None
    try:
        # Get PMF for this device from the optimization service
        service = st.session_state.service
        if service:
            pmf_data = service.get_device_pmf(device)
    except Exception as e:
        print(f"Error getting PMF data for {device}: {e}")
    
    # Default PMF if we couldn't get real data
    if not pmf_data or not isinstance(pmf_data, dict):
        # Create a flat distribution as fallback
        pmf_data = {'current_probabilities': [1/12] * 12}
        
    # Extract the current probabilities from the PMF data
    # These are 2-hour blocks (12 blocks for 24 hours)
    block_probabilities = pmf_data.get('current_probabilities', [1/12] * 12)
    
    # Convert block probabilities to hourly probabilities by repeating each value for 2 hours
    hourly_probabilities = []
    for prob in block_probabilities:
        hourly_probabilities.extend([prob, prob])  # Repeat each probability for 2 consecutive hours
    
    # If we don't have enough values, pad with default probability
    while len(hourly_probabilities) < 24:
        hourly_probabilities.append(1/12)
    
    # Find the maximum probability for normalization
    max_prob = max(hourly_probabilities) if hourly_probabilities else 1/12
    
    # --- build the 24 hour cells with PMF-based color intensity --------
    cells_html = ""
    
    for h in range(24):
        # Get hourly probability and convert to intensity level (0-100, rounded to nearest 10)
        hourly_prob = hourly_probabilities[h]
        # Scale probability to intensity (0-100) and round to nearest 10
        intensity = min(int(hourly_prob / max_prob * 100 / 10) * 10, 100)
        
        value = float(values[h])
        
        # Always apply PMF-based color directly using inline rgba styling
        # This ensures the gradient is visible regardless of other styling
        PMF_OPACITY_COEFF = 0.6  # max 60 % instead of 100 %
        bg_opacity = (intensity / 100) * PMF_OPACITY_COEFF
        
        # For all hours (scheduled or not), use the golden yellow PMF heatmap color.
        # The schedule is indicated SOLELY by the border overlay.
        bg_color = f'rgba(244, 192, 109, {bg_opacity})'
        text_color = '#142b42'
        
        # Add the cell HTML with direct styling to ensure the gradient is visible
        # Remove individual cell borders - the overlay will provide the border
        cells_html += (
            f'<div class="hour-cell" id="cell-{dom}-{h}" '
            f'style="background-color:{bg_color}; color:{text_color}; border:1px solid rgba(0,0,0,.05);" '
            f'title="{value:.2f} kWh, Usage probability: {hourly_prob:.3f}">{h:02d}</div>'
        )
    
    # --- build overlay HTML for each scheduled block -------------------------
    overlays_html = ""
    for i, (start, end) in enumerate(scheduled_blocks):
        # Calculate width and position of the overlay
        overlay_id = f"schedule-overlay-{dom}-{i}"
        overlays_html += f'''
            <div id="{overlay_id}" class="schedule-block-overlay">
                <div class="schedule-label">Schedule</div>
            </div>
            <script>
                (function() {{
                    const startCell = document.getElementById("cell-{dom}-{start}");
                    const endCell = document.getElementById("cell-{dom}-{end}");
                    const overlay = document.getElementById("{overlay_id}");
                    
                    if (startCell && endCell && overlay) {{
                        const startRect = startCell.getBoundingClientRect();
                        const endRect = endCell.getBoundingClientRect();
                        const width = (endRect.right - startRect.left);
                        
                        overlay.style.left = "0px";
                        overlay.style.width = width + "px";
                        overlay.style.transform = `translateX(${{startRect.left - startCell.parentElement.getBoundingClientRect().left}}px)`;
                    }}
                }})();
            </script>
        '''
    

    html_code = f"""
    <style>
      /* Base cell styling */
      #wrap-{dom} {{
        position:relative;
        display:grid;
        grid-template-columns:repeat(24,1fr);
        column-gap:0.25rem;
        width:100%; height:44px;
      }}
      #wrap-{dom} .hour-cell {{
        position:relative;
        display:flex; justify-content:center; align-items:center;
        font-size:.75rem; font-weight:600;
        background:#f0f2f6; color:#142b42; /* Using palette colors */
        border:1px solid rgba(0,0,0,.05); box-sizing:border-box; border-radius:4px;
      }}
      
      /* Schedule block overlay styling - ONLY A BORDER with NO FILL */
      #wrap-{dom} .schedule-block-overlay {{
        position:absolute;
        top:0; 
        height:calc(100% - 1px); /* Slightly reduced to ensure bottom border visibility */
        margin-bottom: 1px; /* Add a small margin at bottom */
        border: 4px solid #00838f;   /* twice as thick */ /* Main teal-blue color */
        border-radius:6px;
        pointer-events:none; /* Allow clicking through to cells */
        z-index:5;
        box-sizing:border-box;
        background:transparent; /* Explicitly ensure no background/fill */
      }}
      
      /* Schedule label above the block */
      #wrap-{dom} .schedule-label {{
        position:absolute;
        top:-8px; left:50%;
        transform:translateX(-50%);
        background:#fff;
        border:1.5px solid #00838f; /* Main teal-blue color */
        border-radius:4px;
        padding:1px 6px; /* More padding for better visibility */
        font-size:12px; /* one step larger */
        color:#00838f;
        white-space:nowrap;
        font-weight:bold;
        z-index:10;
        box-shadow: 0 0 0 2px white; /* Thicker white outline to ensure border visibility */
      }}
      #win-{dom} {{
        position:absolute; top:0;
        height:44px; border-radius:8px;
        background:rgba(20, 43, 66, 0.05); /* #142b42 with opacity */
        cursor:move;
        display:flex; align-items:center; justify-content:space-between;
        box-shadow:0px 2px 5px rgba(20, 43, 66, 0.15); /* #142b42 with opacity */
        padding:0 4px;
        border:1px dashed #0e6072; /* Darker teal outline */
      }}
      /* Actual label above the draggable window */
      #win-{dom} .actual-label {{
        position:absolute;
        top:-8px; /* Position above the bar */
        left:50%;
        transform:translateX(-50%);
        font-size:12px; /* one step larger */
        border:1.5px solid #88233e; /* Burgundy color from palette */
        border-radius:4px;
        background:#fff;
        padding:1px 6px; /* Match Schedule label padding */
        color:#88233e; /* Burgundy color */
        white-space:nowrap;
        font-weight:bold;
        z-index:10;
        box-shadow: 0 0 0 2px white; /* White outline for visibility */
        display: none; /* Hidden by default, shown after user moves it */
      }}
      .grip {{
        width:8px; height:20px; border-radius:4px;
        background:#0e6072; /* Darker teal for grips */
        cursor:pointer;
      }}
    </style>

    <div id="wrap-{dom}">
      {cells_html}
      {overlays_html}
      <div id="win-{dom}">
        <div class="grip"></div>
        <div class="actual-label">Actual</div>
        <div class="grip"></div>
      </div>
    </div>

    <script>
      (function() {{
        const DUR   = {duration};
        const MAX_I = 24 - DUR;
        let idx     = {start_idx};

        const wrap = document.getElementById("wrap-{dom}");
        const win  = document.getElementById("win-{dom}");

        // measure cell width + gap
        let cellW = 0, gap = 0;
        function measure() {{
          const r0 = wrap.children[0].getBoundingClientRect();
          const r1 = wrap.children[1].getBoundingClientRect();
          cellW    = r0.width;
          gap      = Math.round(r1.left - r0.right);
          win.style.width = (cellW*DUR + gap*(DUR-1)) + "px";
          win.style.left  = (idx   * (cellW+gap))     + "px";
        }}
        window.addEventListener("load",   measure);
        window.addEventListener("resize", measure);

        // Track if the window has been moved by the user
        let hasBeenMoved = false;
        const actualLabel = win.querySelector('.actual-label');
        
        // drag behaviour
        let down=false, sx=0, sl=0;
        win.addEventListener("pointerdown", e=>{{down=true; sx=e.clientX;
          sl=idx*(cellW+gap); win.setPointerCapture(e.pointerId);}});
        const end=e=>{{if(!down)return; down=false;
          win.releasePointerCapture(e.pointerId);
          // Show the actual label after the user moves it
          if (hasBeenMoved) {{              
            actualLabel.style.display = 'block';
          }}
          Streamlit.setComponentValue(idx); }};
        win.addEventListener("pointerup",end); win.addEventListener("pointercancel",end);
        win.addEventListener("pointermove", e=>{{if(!down)return;
          const ni=Math.round((sl+e.clientX-sx)/(cellW+gap));
          const newIdx = Math.max(0,Math.min(MAX_I,ni));
          if (newIdx !== idx) {{              
            hasBeenMoved = true; // Mark as moved only if position changed
          }}
          idx = newIdx;
          win.style.left=(idx*(cellW+gap))+"px";}});

        Streamlit.setFrameHeight(44);
      }})();
    </script>
    """

    return _components_html(html_code, height=46)   # old-API, no key

# ─────────────────────────────────────────────────────────────────────────────
# helper – first contiguous scheduled block
# ─────────────────────────────────────────────────────────────────────────────
def _first_contiguous_block(values, duration):
    """
    Return the first hour 'h' such that values[h:h+duration] are all > 0.
    If none found, fall back to the first hour with any load, else 0.
    """
    for h in range(24 - duration + 1):
        if all(values[h + j] > 0 for j in range(duration)):
            return h
    for h, v in enumerate(values):
        if v > 0:
            return h
    return 0


# ────────────────────────────────────────────────────────────────
# helper – savings calculation functions─────────────
def calculate_schedule_cost(schedule: Dict[str, List[float]], prices: List[float]) -> float:
    """
    Calculate the cost of a schedule based on energy prices
    
    Args:
{{ ... }}
        schedule: Dictionary of device schedules (kWh per hour)
        prices: List of hourly prices (€/kWh)
        
    Returns:
        Total cost in euros
    """
    total_cost = 0.0
    
    # Sum up energy usage across all devices (excluding battery SoC)
    hourly_total_kwh = [0.0] * 24
    for device, hourly_usage in schedule.items():
        if device == "battery_soc":  # Skip battery state of charge
            continue
        
        for hour, kwh in enumerate(hourly_usage):
            hourly_total_kwh[hour] += kwh
    
    # Calculate cost
    for hour, kwh in enumerate(hourly_total_kwh):
        total_cost += kwh * prices[hour]
    
    return total_cost
    

# ─────────────────────────────────────────────────────────────────────────────
# Savings helper – cost of the TYPICAL (baseline) run‑time
# ─────────────────────────────────────────────────────────────────────────────
def calculate_baseline_cost(
        schedule: Dict[str, List[float]],
        baseline_usage: Dict[str, int],
        prices: List[float]) -> float:
    """
    I reconstruct what the user would have paid *without* optimisation:
    every device keeps its original energy pattern but is shifted to the
    user's typical start hour (learned during onboarding).

    Args
    ----
    schedule        : current optimised schedule (holds the canonical kWh pattern)
    baseline_usage  : {"device": typical_start_hour}
    prices          : 24‑element €/kWh array

    Returns
    -------
    float – € the user would normally spend
    """
    cost = 0.0
    for dev, hourly_loads in schedule.items():
        if dev == "battery_soc":
            continue                          # ignore SoC trace
        # extract contiguous kWh pattern from the optimizer
        pattern = [k for k in hourly_loads if k > 0]
        if not pattern:
            continue
        dur = len(pattern)
        start = baseline_usage.get(dev, 0)    # fallback to midnight
        for j, kwh in enumerate(pattern):
            h = (start + j) % 24              # wrap‑around safety
            cost += kwh * prices[h]
    return cost


# ─────────────────────────────────────────────────────────────────────────────
# Savings helper – update tracker (baseline‑aware)
# ─────────────────────────────────────────────────────────────────────────────
def update_savings_tracking(
        schedule: Dict[str, List[float]],
        actual_usage: Dict[str, List[float]],
        prices: List[float],
        date_str: str,
        baseline_usage: Dict[str, int]) -> Tuple[float, float]:
    """
    Now I compare three alternative worlds, all sharing the SAME baseline:
    1. Typical run‑time  →  €baseline_cost
    2. Optimised run‑time → €optimised_cost
    3. Actual run‑time   → €actual_cost
    Savings = baseline − scenario
    """
    # Add debug logging to track the calculations
    print(f"\n\n=== SAVINGS DEBUG ===\nSchedule: {schedule}\nActual usage: {actual_usage}\nBaseline usage: {baseline_usage}")
    
    baseline_cost  = calculate_baseline_cost(schedule, baseline_usage, prices)
    optimised_cost = calculate_schedule_cost(schedule, prices)          # ✓ existing helper
    actual_cost    = calculate_schedule_cost(actual_usage, prices)      # ✓ existing helper
    
    # Print costs for debugging
    print(f"Baseline cost: {baseline_cost:.2f}")
    print(f"Optimised cost: {optimised_cost:.2f}")
    print(f"Actual cost: {actual_cost:.2f}")

    potential_sav  = baseline_cost - optimised_cost                     # what we *could* save
    actual_sav     = baseline_cost - actual_cost                        # what we *did* save
    
    # Print savings for debugging
    print(f"Potential savings: {potential_sav:.2f}")
    print(f"Actual savings: {actual_sav:.2f}")
    print("=== END DEBUG ===\n")

    # accumulate in the session
    st.session_state.total_potential_savings += potential_sav
    st.session_state.total_actual_savings    += actual_sav
    st.session_state.daily_savings[date_str]  = {
        "baseline_cost":  baseline_cost,
        "optimised_cost": optimised_cost,
        "actual_cost":    actual_cost,
        "potential":      potential_sav,
        "actual":         actual_sav,
    }
    return potential_sav, actual_sav



# ─────────────────────────────────────────────────────────────────────────────
#  render_schedule_panel  – schedule grid + price lane + draggable bars
# ─────────────────────────────────────────────────────────────────────────────
def render_schedule_panel() -> None:
    """Render the price lane, 24-hour schedule and draggable confirmation bars."""
    # Header with help button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Generated Schedule")
    with col2:
        if st.button("❓ Help", key="schedule_help_button", use_container_width=True):
            schedule_help_contextual()
    
    # Enhanced description with tips
    st.write(
        f"Based on your usage patterns, here's your optimized energy schedule for Day "
        f"{st.session_state.current_day}. The schedule is optimized for cost and energy efficiency."
    )
    
    # Display savings information
    if st.session_state.total_potential_savings > 0 or st.session_state.current_day > 1:
        # Create columns for the savings displays
        savings_col1, savings_col2 = st.columns(2)
        
        with savings_col1:
            # Format the savings with 2 decimal places and thousands separator
            formatted_potential = f"{st.session_state.total_potential_savings:.2f}"
            st.info(f"💰 **Total potential savings: €{formatted_potential}** \nIf you always follow the optimized schedule")
        
        with savings_col2:
            formatted_actual = f"{st.session_state.total_actual_savings:.2f}"
            st.success(f"✅ **Your actual savings: €{formatted_actual}** \nBased on your usage patterns")
    
   


    # ── guard ────────────────────────────────────────────────────────────────
    if not st.session_state.schedule:
        st.toast("Generate a schedule first ↖️", icon="ℹ️")
        return

    # ── make sure we have a price curve (€/kWh) in session_state -------------
    if "price_curve" not in st.session_state:
        # mock: flat @ 0.26 € with a cheap night valley & pricey peak
        base = np.full(24, 0.26)
        base[0:6]  -= 0.10          # cheap 00-05
        base[18:22] += 0.12         # expensive 18-21
        st.session_state.price_curve = base.round(3).tolist()

    prices = st.session_state.price_curve
    p_min, p_max = min(prices), max(prices)

    # ── initialise day-state fields (same as before) -------------------------
    st.session_state.setdefault("draggable_selections", {})
    st.session_state.setdefault("actual_usage",        {})

    # ── UNIFIED VIEW LAYOUT (COLLAPSED TABS) --------------------------------
    # Create a container to hold all content (replaces tabs)
    unified_view = st.container()
    
    # Initialize a variable to hold what was previously in tab2 (PMF view)
    # This preserves all PMF functionality while using a single-view layout
    st.session_state.setdefault("pmf_view", {})

    # ════════════════════════════════════════════════════════════════════════
    # UNIFIED VIEW – schedule + price lane (previously Tab 1)
    # ════════════════════════════════════════════════════════════════════════
    with unified_view:

        # ──────────────────────────────────────────────────────────────────
        # PRICE LANE — identical 24-column grid, perfectly aligned
        # ──────────────────────────────────────────────────────────────────
        def _price_row_html(prices):
            """Return an HTML component that shows one 24-cell row coloured
            using the palette colors for price gradient."""
            p_min, p_max = min(prices), max(prices)
            cells = ""
            for h, p in enumerate(prices):
                # Use the palette colors: low=#00838f, medium=#f4a98a, high=#88233e
                ratio = (p - p_min) / (p_max - p_min + 1e-9)
                if ratio < 0.5:  # 0-0.5: teal blue → light salmon
                    t = ratio * 2  # Scale to [0, 1]
                    r = int(0 + (244 - 0) * t)      # 0x00 to 0xf4
                    g = int(131 + (169 - 131) * t)  # 0x83 to 0xa9
                    b = int(143 + (138 - 143) * t)  # 0x8f to 0x8a
                else:  # 0.5-1: light salmon → burgundy
                    t = (ratio - 0.5) * 2  # Scale to [0, 1]
                    r = int(244 - (244 - 136) * t)  # 0xf4 to 0x88
                    g = int(169 - (169 - 35) * t)   # 0xa9 to 0x23
                    b = int(138 - (138 - 62) * t)   # 0x8a to 0x3e
                # Set text color to white for darker backgrounds, black for lighter backgrounds
                fg = "#ffffff" if ratio > 0.6 else "#000000"
                cells += (f"<div class='price-cell' "
                          f"style='background:rgb({r},{g},{b});color:{fg};'"
                          f"title='€{p:.3f}/kWh'>{p:.2f}</div>")
            html_code = f"""
            <style>
              #price-wrap {{
                display:grid;grid-template-columns:repeat(24,1fr);
                column-gap:0.25rem;width:100%;height:30px;
              }}
              #price-wrap .price-cell {{
                display:flex;align-items:center;justify-content:center;
                font-size:.75rem;font-weight:600;box-sizing:border-box;border:1px solid transparent;border-radius:4px;
              }}
            </style>
            <div id='price-wrap'>{cells}</div>
            """
            return _components_html(html_code, height=34)

        # left label + price chart using Altair (smoother visualization)
        price_cols = st.columns([3, 24])
        price_cols[0].markdown("<b>Price (€/kWh)</b>", unsafe_allow_html=True)
        with price_cols[1]:
            # Create price curve chart with Altair instead of HTML heat-bar
            import altair as alt
            import pandas as pd
            
            # Create dataframe for the chart with guaranteed non-empty data
            df = pd.DataFrame({
                'Hour': list(range(24)),  # Ensure all 24 hours are present
                'Price (€/kWh)': prices
            })
            
            # Ensure min and max have sufficient separation to display properly
            p_min, p_max = min(prices), max(prices)
            if p_max - p_min < 0.05:  # If prices are too flat, add some visual range
                p_min = p_min * 0.95
                p_max = p_max * 1.05
            
            # Create the Altair chart with proper hour alignment and increased height
            base = alt.Chart(df).encode(
                x=alt.X('Hour:Q', 
                       axis=alt.Axis(
                           title=None, 
                           values=list(range(24)),  # Force all 24 hour labels
                           labelAngle=0,
                           grid=True
                       ),
                       scale=alt.Scale(domain=[0, 23], nice=False)  # Exact alignment with 24h grid
                      )
            )
            
            # Area chart for prices with color gradient
            area = base.mark_area(opacity=0.8, line=True).encode(
                y=alt.Y('Price (€/kWh):Q', 
                       title="", 
                       scale=alt.Scale(domain=[p_min*0.95, p_max*1.05], nice=False)),
                color=alt.Color('Price (€/kWh):Q', 
                               scale=alt.Scale(domain=[p_min, p_max],
                                             range=['#00838f', '#f4a98a', '#88233e']), 
                               legend=None)
            ).properties(
                height=120,  # Increased height for better visibility
                width='container'
            )
            
            # Show the chart with container width to match schedule grid
            st.altair_chart(area, use_container_width=True)
        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
        st.write("👉 Drag the blue window to confirm when devices actually ran.")
        
        # Simple legend with icons
        st.markdown("<small>🟥 deeper red = higher price | 🟦 blue bar = optimiser window | 🟨 golden shade = usual start</small>", unsafe_allow_html=True)

        # ---------- one row per selected device (UNCHANGED) ------------------
        for device, sched_row in st.session_state.schedule.items():
            if device == "battery_soc" or device not in st.session_state.selected_devices:
                continue

            # contiguous run-time
            try:
                from notebooks.utils.device_specs import device_specs
                phases   = device_specs.get(device, {}).get("phases", [])
                dur      = int(sum(int(p.get("duration",1)) for p in phases)) or 2
            except Exception:
                dur = 2

            # initial start-hour
            if device in st.session_state.draggable_selections:
                s0 = st.session_state.draggable_selections[device]["start_hour"]
            else:
                s0 = _first_contiguous_block(sched_row, dur)
                st.session_state.draggable_selections[device] = {"start_hour": s0}

            # label + grid
            row = st.columns([3, 24])
            row[0].write(f"**{device.replace('_',' ').title()}**")
            with row[1]:
                new_s = _render_device_row_html(device, sched_row, dur, s0)

            # If the component returns a new start hour, it means the user dragged the window.
            if isinstance(new_s, int) and s0 != new_s:
                # Update the start hour in session state
                s0 = new_s
                st.session_state.draggable_selections[device]["start_hour"] = s0

                # Mark this device as needing a PMF refresh
                st.session_state.pmf_refresh_needed = True
                st.session_state.pmf_refresh_device = device

                # ─── CRITICAL FIX: ONLY UPDATE ACTUAL USAGE WHEN THE USER CHANGES IT ───
                # This was the source of the bug. This block now only runs when new_s is returned.
                
                # Capture the contiguous kWh profile that the optimiser proposed
                s_opt = _first_contiguous_block(sched_row, dur)  # Original start
                energy_pattern = [float(sched_row[s_opt + j]) for j in range(dur)]  # kWh for each phase hour

                # Replay that exact pattern at the user-selected start hour (s0)
                new_actual_usage = [0.0] * 24
                for h in range(24):
                    if s0 <= h < s0 + dur:  # Inside the new blue bar position
                        idx = h - s0  # Offset in the energy pattern
                        new_actual_usage[h] = energy_pattern[idx]
                
                # Update the session state with the new usage pattern
                st.session_state.actual_usage[device] = new_actual_usage
                
                # Rerun to reflect the change immediately and update the PMF
                st.rerun()
                    
            # Immediately update PMF with actual usage if it changed
            if hasattr(st.session_state, 'pmf_refresh_needed') and st.session_state.pmf_refresh_needed \
               and st.session_state.pmf_refresh_device == device:
                try:
                    # Submit the updated actual usage for this device to update PMF
                    iso = st.session_state.current_date.isoformat()
                    device_actual = {device: st.session_state.actual_usage[device]}
                    st.session_state.service.update_with_actuals(iso, device_actual)
                    st.session_state.pmf_refresh_needed = False
                    st.session_state.pmf_refresh_device = None
                    # Force the component to rerender to show updated PMF
                    st.rerun()
                except Exception as e:
                    print(f"Error updating PMF for {device}: {e}")
                    # Reset the refresh flags even on error to avoid infinite loops
                    st.session_state.pmf_refresh_needed = False
                    st.session_state.pmf_refresh_device = None
            # ──────────────────────────────────────────────────────────────────────

        # ---------- SUBMIT button --------------------------
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Create an empty column and a right-aligned button column
        empty_col, warning_col, submit_col = st.columns([0.2, 1.8, 1])

        missing = [
            d for d in st.session_state.selected_devices
            if not any(st.session_state.actual_usage.get(d, []))
        ]
        ready = len(missing) == 0

        with warning_col:
            if not ready:
                st.warning("Confirm all devices: " + ", ".join(
                    m.replace('_',' ').title() for m in missing))

        with submit_col:
            if st.button(
                f"Submit All & Advance to Day {st.session_state.current_day + 1}",
                type="primary", disabled=not ready,
                use_container_width=True
            ):
                iso = st.session_state.current_date.isoformat()
                
                # Calculate and update savings before submitting actuals
                if st.session_state.schedule and st.session_state.actual_usage and "price_curve" in st.session_state:
                    update_savings_tracking(
                        st.session_state.schedule,
                        st.session_state.actual_usage,
                        st.session_state.price_curve,
                        iso,
                        st.session_state.baseline_usage
                    )
                    # Show a toast with savings for this day
                    daily_savings = st.session_state.daily_savings.get(iso, {})
                    if daily_savings:
                        today_savings = daily_savings.get("actual", 0.0)
                        st.toast(f"Today's savings: €{today_savings:.2f}", icon="💰")
                
                # Submit actuals to service
                st.session_state.service.update_with_actuals(
                    iso, st.session_state.actual_usage
                )
                
                # advance
                st.session_state.current_day  += 1
                st.session_state.current_date += timedelta(days=1)
                # reset & regenerate
                st.session_state.schedule = {}
                st.session_state.actual_usage = {}
                st.session_state.draggable_selections = {}
                generate_schedule()
                st.success("New day started ✓")
                st.rerun()  # safe for old Streamlit: defined earlier




def render_history_panel() -> None:
    """Render the history panel in the sidebar with proper key management"""
    st.sidebar.header("History")
    
    # Get history items (don't render if empty)
    history = st.session_state.service.get_schedule_history()
    if not history:
        st.sidebar.info("No schedule history available")
        return
    
    # Initialize history state if needed
    if "selected_history_index" not in st.session_state:
        st.session_state.selected_history_index = 0
    if "selected_history_file" not in st.session_state:
        st.session_state.selected_history_file = None
    
    # Simple approach: use a selectbox for history selection
    with st.sidebar.expander("Schedule History", expanded=False):
        # Create display labels
        options = [(f"{item['building_id']} - {item['date']}", item['file']) for item in history]
        labels = [option[0] for option in options]
        
        if labels:
            # Use a consistent key based on history length
            history_key = f"history_select_{len(history)}"
            
            # Use index rather than selection
            index = st.selectbox(
                "Select a schedule:",
                options=range(len(labels)),
                format_func=lambda i: labels[i],
                key=history_key,
                index=st.session_state.selected_history_index
            )
            
            # Update selected index
            st.session_state.selected_history_index = index
            
            # Show view button with a consistent key
            view_key = f"view_btn_{len(history)}"
            if st.button("View Selected Schedule", key=view_key):
                st.session_state.selected_history_file = options[index][1]
        
        # Display selected schedule (if any)
        if st.session_state.selected_history_file:
            try:
                with open(st.session_state.selected_history_file, 'r') as f:
                    schedule_data = json.load(f)
                
                st.markdown("---")
                st.subheader("Schedule Details")
                
                # Get building and date from filename
                building_id = next((item['building_id'] for item in history if item['file'] == st.session_state.selected_history_file), "Unknown")
                date_str = next((item['date'] for item in history if item['file'] == st.session_state.selected_history_file), "Unknown")
                
                # Show details
                st.write(f"**Building:** {building_id}")
                st.write(f"**Date:** {date_str}")
                
                # Handle both old and new schedule format
                if "devices" in schedule_data:
                    # New format with devices and price_curve
                    devices = schedule_data.get("devices", {})
                    price_curve = schedule_data.get("price_curve", [0.26] * 24)
                    battery_soc = devices.get("battery_soc", [0.0] * 24)
                else:
                    # Old format (direct device schedule)
                    devices = schedule_data
                    battery_soc = schedule_data.get("battery_soc", [0.0] * 24)
                    # Default price curve if not available
                    price_curve = [0.26] * 24
                    
                # Show price curve
                if price_curve and any(price_curve):
                    df_price = pd.DataFrame({
                        "Hour": list(range(24)),
                        "Price (€/kWh)": price_curve
                    })
                    st.write("**Hourly Price Curve:**")
                    st.line_chart(df_price, x="Hour", y="Price (€/kWh)")
                
                # Show battery SoC
                if any(battery_soc):
                    df = pd.DataFrame({
                        "Hour": list(range(24)),
                        "State of Charge (%)": [soc * 100 for soc in battery_soc]
                    })
                    st.write("**Battery State of Charge:**")
                    st.line_chart(df, x="Hour", y="State of Charge (%)")
                    
                # Show device schedules
                device_keys = [k for k in devices.keys() if k != "battery_soc"]
                if device_keys:
                    st.write("**Device Schedules:**")
                    df_devices = pd.DataFrame({
                        "Hour": list(range(24))
                    })
                    for device in device_keys:
                        df_devices[device.replace('_', ' ').title()] = devices[device]
                    st.line_chart(df_devices, x="Hour")
                
                # Show the raw data
                with st.expander("Raw Schedule Data"):
                    st.json(schedule_data)
                    
                # Clear button with consistent key
                clear_key = f"clear_btn_{len(history)}"
                if st.button("Close Details", key=clear_key):
                    st.session_state.selected_history_file = None
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading schedule: {e}")
                st.session_state.selected_history_file = None


def render_lifetime_kpis():
    """Display always-visible lifetime savings KPIs in the sidebar."""
    # Only show if we have schedule data and have calculated savings
    if not st.session_state.schedule or st.session_state.current_day < 2:
        return
    
    st.markdown("""
    <div style="margin-top:24px; margin-bottom:8px">
    <h4 style="margin-bottom:4px">🏆 Lifetime Performance</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Create metrics for lifetime statistics
    col1, col2 = st.columns(2)
    
    # Historical savings
    with col1:
        st.metric(
            "💰 Total Savings", 
            f"€{st.session_state.total_actual_savings:.2f}", 
            delta=None
        )
    
    # Optimization rate
    with col2:
        # Calculate percent of days where actual matched schedule (within 10%)
        days_optimized = st.session_state.get("days_optimized", 0)
        total_days = st.session_state.current_day - 1
        if total_days > 0:
            optimization_rate = min(100, int(days_optimized / total_days * 100))
            st.metric(
                "⚡ Days Optimized", 
                f"{optimization_rate}%", 
                delta=None
            )


def main() -> None:
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Initialize the optimization service
    global service 
    service = OptimisationService()
    
    # Show the appropriate dialog based on onboarding step
    step = st.session_state.get("show_step", "welcome" if not st.session_state.get("onboarding_complete", False) else None)
    
    # if step == "welcome":
    #     welcome_modal()        # shows immediately
    if step == "device":
        device_help_modal()
    elif step == "schedule":
        schedule_help_modal()
    
    # Page already configured at top of file via st.set_page_config()
    # Note: Set initial_sidebar_state in the page config at the top of the file instead
    
    # Hide Streamlit menu and footer, but keep sidebar available
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    
    # Load CSS
    with open(Path(__file__).parent / "styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Load Font Awesome for icons
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">', unsafe_allow_html=True)
    
    # Render header
    render_header()
    
    # ── NEW: lightweight breadcrumb (3‑state) ─────────────────────────────
    step_map = {                                     # simple finite‑state map
        False: 1 if not st.session_state.schedule else 2,   # onboarding or gen
        True:  3                                           # confirmation stage
    }[st.session_state.onboarding_complete]
    
    step_label = {1: "Choose devices",
                  2: "Generate schedule",
                  3: "Confirm actual run‑time"}[step_map]
    
    st.markdown(
        f"<div style='font-size:0.9rem; margin-bottom:0.3rem;'>"
        f"🔄 <b>Step&nbsp;{step_map}/3</b>&nbsp;&nbsp;{step_label}"
        f"</div>", unsafe_allow_html=True)
    # ──────────────────────────────────────────────────────────────────────
    
    # Create main layout
    left_col, right_col = st.columns([1, 4])
    
    # Left column - Device picker, constraints and lifetime KPIs
    with left_col:
        render_device_picker()
        
        # Always-visible lifetime KPI sidebar
        render_lifetime_kpis()
        
        # Add prominent Generate Schedule button with some spacing
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.button(
            "Generate Schedule", 
            on_click=generate_schedule, 
            use_container_width=True, 
            type="primary",
            key="generate_schedule_button"
        )
        
        # Add a warning message if no devices are selected
        if not st.session_state.selected_devices:
            st.warning("Please select at least one device before generating a schedule")
            
        # Show constraints summary if devices are selected
        elif st.session_state.selected_devices:
            with st.expander("View all device constraints", expanded=False):
                for device in st.session_state.selected_devices:
                    constraints = st.session_state.device_constraints.get(device, {"earliest_hour": 0, "latest_hour": 23})
                    st.write(f"**{device.replace('_', ' ').title()}**: Allowed hours {constraints['earliest_hour']:02d}:00 - {constraints['latest_hour']:02d}:59")

    
    # Right column - Schedule panel
    with right_col:
        # Schedule panel
        if st.session_state.schedule:
            render_schedule_panel()
    
    # Add implementation status information
    st.sidebar.markdown("---")
    with st.sidebar.expander("Implementation Status", expanded=False):
        if service.using_real_service:
            st.success(
                "**✅ Successfully Integrated:** This dashboard is using the **real MILP optimization service**. "  
                "The optimization engine and probability model agent are fully connected and operational. \n\n"  
                "**Features Available:** \n"  
                "1. Full MILP optimization for device scheduling \n"  
                "2. Adaptive PMF updates based on actual usage \n"  
                "3. Battery SoC forecasting and integration \n"  
                "4. Real-time constraint-based schedule generation \n\n"  
                "All UI functionality and data visualization components are fully operational."
            )
        else:
            st.info(
                "**Current Status:** This dashboard is using the mock optimization service. \n\n"  
                "The real MILP optimizer integration has been implemented but is encountering "  
                "import path or initialization issues. The adapter layer is ready once the core "  
                "components are accessible. \n\n"  
                "**Next Steps:** \n"  
                "1. Resolve Python package structure issues \n"  
                "2. Fix import paths between web app and core components \n"  
                "3. Complete final testing of the real service integration \n\n"  
                "All UI functionality, PMF history tracking, and schedule generation remain fully operational."
            )
            
    # Sidebar - History and Feedback
    render_history_panel()
    render_feedback_system()
    
    # Handle dialogs with priority system to ensure only one is shown at a time
    # Priority: welcome > device help > schedule help
    # We'll store which dialog to show in session state and clear other dialog flags
    
    # Initialize active dialog flag if needed
    if "active_dialog" not in st.session_state:
        st.session_state.active_dialog = None
    
    # Determine which dialog to show based on priority
    if st.session_state.first_visit and not st.session_state.onboarding_complete and st.session_state.active_dialog != "shown":
        st.session_state.active_dialog = "welcome"
        # Reset other dialog flags to prevent multiple dialogs
        st.session_state.show_device_help = False
        st.session_state.show_schedule_help = False
    elif st.session_state.show_device_help and st.session_state.active_dialog != "shown":
        st.session_state.active_dialog = "device_help"
        # Reset other dialog flags
        st.session_state.show_schedule_help = False
    elif st.session_state.show_schedule_help and st.session_state.active_dialog != "shown":
        st.session_state.active_dialog = "schedule_help"
    
    # Show the appropriate dialog based on the active_dialog flag
    if st.session_state.active_dialog == "welcome":
        welcome_modal()
        st.session_state.active_dialog = "shown"
    elif st.session_state.active_dialog == "device_help":
        device_help_contextual()
        st.session_state.active_dialog = "shown"
        st.session_state.show_device_help = False
    elif st.session_state.active_dialog == "schedule_help":
        schedule_help_contextual()
        st.session_state.active_dialog = "shown"
        st.session_state.show_schedule_help = False
    else:
        # Reset the active dialog flag at the end of each script run if no dialog was shown
        # This allows a new dialog to be shown on the next run
        st.session_state.active_dialog = None
    
    # Mark first visit complete after initial page load
    if st.session_state.first_visit:
        st.session_state.first_visit = False


if __name__ == "__main__":
    main()
