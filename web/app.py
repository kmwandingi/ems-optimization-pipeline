"""
Streamlit demo app for MILP Optimizer with onboarding system
"""

import streamlit as st
st.set_page_config(page_title="EMS Scheduler", layout="wide")
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
        
    def next_day(self, building_id: str, device_constraints: Dict[str, Dict[str, int]]) -> Dict[str, List[float]]:
        """
        Calculate the next day schedule for a building.
        
        Args:
            building_id: ID of the building
            device_constraints: Dictionary mapping device names to constraints
                                {device_name: {"earliest_hour": int, "latest_hour": int}}
        
        Returns:
            Dictionary mapping device names to 24-hour schedules (list of kWh values)
        """
        # Use current date for optimization
        target_date = date.today()

        # If we're using the mock service, leverage its purpose-built next_day
        if not self.using_real_service and hasattr(self.service, "next_day"):
            # Get schedule and price curve from mock service
            schedule, price_curve = self.service.next_day(building_id, device_constraints)
            
            # Persist to file for history / debugging consistency
            schedule_file = self.schedules_dir / f"{building_id}_{target_date.isoformat()}.json"
            with open(schedule_file, "w") as f:
                # Add price curve to the schedule data for historical reference
                schedule_data = {
                    "devices": schedule,
                    "price_curve": price_curve
                }
                json.dump(schedule_data, f, indent=2)
                
            # Return the device schedules and price curve separately
            return schedule, price_curve

        # ----- Real service path below -----
        # Get battery agent
        battery_agent = self.service.get_battery_agent(building_id)
        
        # Get actual optimization service by devicetype
        service_device_map = {
            "battery": "battery",
            "dishwasher": "wet",
            "refrigerator": "ct",
            "freezer": "ct",
            "washing_machine": "wet",
            "electric_vehicle": "ev",
            "electric_heating": "th"
        }
        
        # Map selected devices to appropriate service names
        selected_services = set()
        for device in device_constraints.keys():
            service = service_device_map.get(device)
            if service: selected_services.add(service)
        
        # Get schedules and combine them
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
        
    def get_device_pmf(self, device_name: str) -> dict:
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
    if "selected_devices" not in st.session_state:
        st.session_state.selected_devices = set()
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


# ─────────────────────────────────────────────────────────────────────────────
#  Modal Overlay System
# ─────────────────────────────────────────────────────────────────────────────

def render_modal_overlay(id: str, content_function, show: bool = True) -> None:
    """
    Wrap `content_function()` inside a Streamlit modal. The modal opens only
    when `show` is True and closes automatically when the user clicks the ✕,
    the backdrop, or any button inside that flips the session-state flag.
    """
    if not show:
        return

    # st.modal creates the overlay & backdrop for us
    with st.modal(key=id, use_container_width=True):
        content_function()


def close_modal(modal_id: str) -> None:
    """Close a specific modal
    
    Args:
        modal_id: ID of the modal to close
    """
    # Update session state to close the modal
    if modal_id == "welcome_modal":
        st.session_state.show_welcome_modal = False
    elif modal_id == "device_help_modal":
        st.session_state.show_device_help = False
    elif modal_id == "schedule_help_modal":
        st.session_state.show_schedule_help = False
        
    # Force app to rerun to reflect the change
    st.rerun()


def welcome_modal_content() -> None:
    """Content for the welcome modal"""
    # Modal container div for styling
    with st.container():
        # Header
        st.markdown("""<div class='modal-header'>
                      <h3>👋 Welcome to EMS Scheduler!</h3>
                      <button class='modal-close'>×</button>
                     </div>""", unsafe_allow_html=True)
        
        # Body
        st.markdown("""<div class='modal-body'>
                      <p>Welcome to <strong>EMS Scheduler</strong>, your smart energy management system!</p>
                      <p>This tool helps you schedule your household devices to:</p>
                      <ul>
                        <li>🌱 Reduce energy costs</li>
                        <li>⚡ Optimize for renewable energy usage</li>
                        <li>🔋 Balance load across your home</li>
                      </ul>
                      <p>Let's get you started with a quick tour!</p>
                     </div>""", unsafe_allow_html=True)
        
        # Footer
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Skip Tour", key="skip_tour"):
                st.session_state.onboarding_complete = True
                st.session_state.show_welcome_modal = False
                st.rerun()
        with col2:
            if st.button("Start Tour", key="start_tour", type="primary"):
                st.session_state.onboarding_step = 1
                st.session_state.show_welcome_modal = False
                st.session_state.show_device_help = True
                st.rerun()


def device_help_modal_content() -> None:
    """Content for the device selection help modal"""
    with st.container():
        # Header
        st.markdown("""<div class='modal-header'>
                      <h3>🔌 Device Selection</h3>
                      <button class='modal-close'>×</button>
                     </div>""", unsafe_allow_html=True)
        
        # Body
        st.markdown("""<div class='modal-body'>
                      <p>This is the <strong>Device Selection</strong> panel:</p>
                      <ol>
                        <li>Click on device buttons to select which appliances you want to schedule</li>
                        <li>For each device, set when it should run (earliest and latest hour)</li>
                        <li>The system will optimize the schedule for all selected devices</li>
                      </ol>
                      <p>Different devices have different energy requirements:</p>
                      <ul>
                        <li>🍽️ <strong>Dishwasher</strong>: 1-2 kWh per cycle</li>
                        <li>👕 <strong>Washing Machine</strong>: 0.5-1.5 kWh per cycle</li>
                        <li>👖 <strong>Tumble Dryer</strong>: 2-3 kWh per cycle</li>
                        <li>🚿 <strong>Water Heater</strong>: 1.5-4 kWh per day</li>
                        <li>🔥 <strong>Heat Pump</strong>: Variable usage based on temperature</li>
                      </ul>
                     </div>""", unsafe_allow_html=True)
        
        # Footer
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Back", key="device_help_back"):
                st.session_state.show_device_help = False
                st.session_state.show_welcome_modal = True
                st.rerun()
        with col2:
            if st.button("Next: Schedules", key="device_help_next", type="primary"):
                st.session_state.onboarding_step = 2
                st.session_state.show_device_help = False
                st.session_state.show_schedule_help = True
                st.rerun()


def schedule_help_modal_content() -> None:
    """Content for the schedule panel help modal"""
    with st.container():
        # Header
        st.markdown("""<div class='modal-header'>
                      <h3>📊 Schedule Panel</h3>
                      <button class='modal-close'>×</button>
                     </div>""", unsafe_allow_html=True)
        
        # Body
        st.markdown("""<div class='modal-body'>
                      <p>This is the <strong>Schedule Panel</strong> where you'll see:</p>
                      <ol>
                        <li>Hourly schedules for each selected device (blue shades)</li>
                        <li>Actual usage tracking when you confirm running times (green shades)</li>
                        <li>Drag the confirmation slider to log when a device actually ran</li>
                      </ol>
                      <p>Energy usage is color-coded by intensity:</p>
                      <ul>
                        <li>🔵 <strong>Light blue</strong>: Low energy usage</li>
                        <li>🟦 <strong>Medium blue</strong>: Moderate energy usage</li>
                        <li>🟪 <strong>Dark blue</strong>: High energy usage</li>
                      </ul>
                      <p>After making your device selections, click <strong>Generate Schedule</strong> to create an optimized schedule for tomorrow.</p>
                     </div>""", unsafe_allow_html=True)
        
        # Footer
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Back", key="schedule_help_back"):
                st.session_state.show_schedule_help = False
                st.session_state.show_device_help = True
                st.rerun()
        with col2:
            if st.button("Finish Tour", key="schedule_help_finish", type="primary"):
                st.session_state.onboarding_complete = True
                st.session_state.show_schedule_help = False
                st.rerun()


def render_floating_help_button() -> None:
    """Render a floating help button that can open help modals"""
    help_button_html = """
    <div class="help-button-container">
        <button class="help-button" id="help-button" title="Get Help">
            <i class="fas fa-question"></i>
        </button>
    </div>
    <script>
        // Help button click handler
        document.getElementById('help-button').addEventListener('click', function() {
            // Send message to Streamlit to show help modal
            window.parent.postMessage({type: "streamlit:help", action: "show"}, "*");
        });
    </script>
    """
    html(help_button_html, height=60)


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
    """Render the app header with gradient styling"""
    st.markdown(
        f"""
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
        st.session_state.selected_devices.add(device)
        # Initialize constraints if not already set
        if device not in st.session_state.device_constraints:
            st.session_state.device_constraints[device] = {
                "earliest_hour": 0,
                "latest_hour": 23
            }


def render_device_picker() -> None:
    """Render the device picker panel with categorized devices and energy usage info"""
    # Header with Generate Schedule button
    col1, col2 = st.columns([2, 2])
    with col1:
        st.subheader("Select Devices")
    with col2:
        # Only enable the button if devices are selected
        has_devices = len(st.session_state.selected_devices) > 0
        if st.button(
            "📅 Generate Schedule", 
            key="generate_schedule_btn", 
            type="primary", 
            use_container_width=True,
            disabled=not has_devices
        ):
            generate_schedule()
    
    # Show a message about selected devices
    if st.session_state.selected_devices:
        selected_names = [dev.replace("_", " ").title() for dev in st.session_state.selected_devices]
        st.success(f"Selected: {', '.join(selected_names)}")
    else:
        st.info("No devices selected. Click on one or more devices below.")
    
    # Define device categories and their properties
    device_categories = {
        "Kitchen Appliances": [
            {"name": "dishwasher", "icon": "🍽️", "energy": "1-2 kWh/cycle", "description": "Flexible loads that can run any time"}, 
            {"name": "refrigerator", "icon": "❄️", "energy": "0.5-1 kWh/day", "description": "Continuous operation with cycling"}, 
            {"name": "freezer", "icon": "🧊", "energy": "0.8-1.2 kWh/day", "description": "Continuous operation with cycling"}
        ],
        "Laundry": [
            {"name": "washing_machine", "icon": "👕", "energy": "0.5-1.5 kWh/cycle", "description": "Flexible load with multiple cycles"}, 
            {"name": "tumble_dryer", "icon": "👖", "energy": "2-3 kWh/cycle", "description": "High power, flexible timing"}
        ],
        "Heating & Cooling": [
            {"name": "water_heater", "icon": "🚿", "energy": "1.5-4 kWh/day", "description": "Thermal storage capability"}, 
            {"name": "heat_pump", "icon": "🔥", "energy": "Variable", "description": "Weather-dependent operation"}
        ]
    }
    
    # Render devices by category
    for category, devices in device_categories.items():
        with st.expander(category, expanded=True):
            for device_info in devices:
                device = device_info["name"]
                icon = device_info["icon"]
                energy = device_info["energy"]
                description = device_info["description"]
                
                selected = device in st.session_state.selected_devices
                
                # Create a container for the device button and its constraints
                device_container = st.container()
                
                # Apply minimal CSS just for button text
                st.markdown("""
                <style>
                /* Button styling to prevent text cutoff */
                .stButton button {    
                    white-space: nowrap;
                    overflow: visible;
                    text-overflow: clip;
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
                        btn_col, slider_col = st.columns([3, 4])
                        
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
                        
                        st.markdown("<hr style='margin: 5px 0px; border-width: 1px;'>", unsafe_allow_html=True)
    
    # Add help button at the bottom of the device picker
    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
    if st.button("❔ Help", key="device_help_button", use_container_width=True):
        st.session_state.show_device_help = True
        st.rerun()


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
                selected_device_constraints
            )
            
            # Update session state with schedule and price curve
            st.session_state.schedule = schedule
            st.session_state.price_curve = price_curve
            
            # Initialize actual usage with zeros
            st.session_state.actual_usage = {
                device: [0.0] * 24 for device in schedule 
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
    Render an hour cell with color intensity based on kWh value
    
    Args:
        hour: Hour (0-23)
        device: Device name
        value: kWh value
        is_actual: Whether this is for actual usage or scheduled usage
    """
    # Ensure value is a number and convert to float
    try:
        value = float(value)
    except (TypeError, ValueError):
        print(f"Warning: Invalid value for {device} at hour {hour}: {value}, using 0.0")
        value = 0.0
        
    # Scale color intensity relative to the row maximum so highest cell is 100
    max_value = max_val if max_val > 0 else 1.0
    intensity = min(int(value / max_value * 100 / 10) * 10, 100)
    
    # Choose color class based on whether this is schedule or actual
    color_class = "actual" if is_actual else "schedule"
    
    # Determine cell label based on hour
    hour_str = f"{hour:02d}"
    
    # Create a colored cell instead of a button with the proper color intensity
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
    Paints 24 colour-coded schedule cells **inside the same iframe** that hosts
    the draggable confirmation bar, so colours and bar are guaranteed to align.
    Returns the updated start-hour (int) or None (older Streamlit versions).
    """
    dom      = device.replace(" ", "_")
    row_max  = max(float(v) for v in values) if any(values) else 1.0

    # --- local copy of the schedule palette (identical to styles.css) ----------
    schedule_css = """
      .schedule-0   { background:#f8fafc; color:#64748b; }
      .schedule-10  { background:#eff6ff; color:#64748b; }
      .schedule-20  { background:#dbeafe; color:#1e40af; }
      .schedule-30  { background:#bfdbfe; color:#1e40af; }
      .schedule-40  { background:#93c5fd; color:#1e40af; }
      .schedule-50  { background:#60a5fa; color:#ffffff; }
      .schedule-60  { background:#3b82f6; color:#ffffff; }
      .schedule-70  { background:#2563eb; color:#ffffff; }
      .schedule-80  { background:#1d4ed8; color:#ffffff; }
      .schedule-90  { background:#1e40af; color:#ffffff; }
      .schedule-100 { background:#1e3a8a; color:#ffffff; }
    """

    # --- build the 24 hour cells ------------------------------------------------
    cells_html = ""
    for h in range(24):
        intensity = min(int(float(values[h]) / row_max * 100 // 10 * 10), 100)
        cells_html += (
            f'<div class="hour-cell schedule-{intensity}" '
            f'title="{values[h]:.2f} kWh">{h:02d}</div>'
        )

    html_code = f"""
    <style>
      {schedule_css}
      #wrap-{dom} {{
        position:relative;
        display:grid;
        grid-template-columns:repeat(24,1fr);
        column-gap:0.25rem;
        width:100%; height:44px;
      }}
      #wrap-{dom} .hour-cell {{
        display:flex; justify-content:center; align-items:center;
        font-size:.75rem; font-weight:600;
        border:1px solid rgba(0,0,0,.05); box-sizing:border-box; border-radius:4px;
      }}
      #win-{dom} {{
        position:absolute; top:0;
        height:44px; border-radius:8px;
        background:rgba(6,182,212,.25);
        border:2px solid var(--secondary-color);
        box-sizing:border-box;
        cursor:grab;
        display:flex; justify-content:space-between;
        z-index:5;
      }}
      #win-{dom} .grip {{
        width:6px; height:100%;
        background:var(--secondary-color); cursor:ew-resize;
      }}
    </style>

    <div id="wrap-{dom}">
      {cells_html}
      <div id="win-{dom}"><div class="grip"></div><div class="grip"></div></div>
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

        // drag behaviour
        let down=false, sx=0, sl=0;
        win.addEventListener("pointerdown", e=>{{down=true; sx=e.clientX;
          sl=idx*(cellW+gap); win.setPointerCapture(e.pointerId);}});
        const end=e=>{{if(!down)return; down=false;
          win.releasePointerCapture(e.pointerId);
          Streamlit.setComponentValue(idx); }};
        win.addEventListener("pointerup",end); win.addEventListener("pointercancel",end);
        win.addEventListener("pointermove", e=>{{if(!down)return;
          const ni=Math.round((sl+e.clientX-sx)/(cellW+gap));
          idx=Math.max(0,Math.min(MAX_I,ni));
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


# ─────────────────────────────────────────────────────────────────────────────
# helper – savings calculation functions
# ─────────────────────────────────────────────────────────────────────────────
def calculate_schedule_cost(schedule: Dict[str, List[float]], prices: List[float]) -> float:
    """
    Calculate the cost of a schedule based on energy prices
    
    Args:
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


def calculate_unoptimized_cost(schedule: Dict[str, List[float]], prices: List[float]) -> float:
    """
    Calculate what the cost would be if the same energy was used during peak price hours
    (Worst case scenario for cost comparison)
    
    Args:
        schedule: Dictionary of device schedules (kWh per hour)
        prices: List of hourly prices (€/kWh)
        
    Returns:
        Unoptimized cost in euros
    """
    # Calculate total kWh per device
    device_total_kwh = {}
    for device, hourly_usage in schedule.items():
        if device == "battery_soc":  # Skip battery state of charge
            continue
        device_total_kwh[device] = sum(hourly_usage)
    
    # Sort prices from highest to lowest
    sorted_hours = sorted(range(24), key=lambda h: prices[h], reverse=True)
    
    # Allocate energy usage to the most expensive hours
    total_kwh = sum(device_total_kwh.values())
    allocated_kwh = 0.0
    unoptimized_cost = 0.0
    
    for hour in sorted_hours:
        price = prices[hour]
        kwh_to_allocate = min(total_kwh - allocated_kwh, 1.0)  # Allocate up to 1 kWh per hour
        if kwh_to_allocate <= 0:
            break
            
        unoptimized_cost += kwh_to_allocate * price
        allocated_kwh += kwh_to_allocate
    
    return unoptimized_cost


def update_savings_tracking(schedule: Dict[str, List[float]], actual_usage: Dict[str, List[float]], prices: List[float], date_str: str) -> Tuple[float, float]:
    """
    Update savings tracking based on scheduled and actual usage
    
    Args:
        schedule: Dictionary of device schedules
        actual_usage: Dictionary of actual device usage
        prices: List of hourly prices (€/kWh)
        date_str: Date string for tracking
        
    Returns:
        Tuple of (optimized_savings, actual_savings) for this period
    """
    # Calculate costs based on the optimized schedule
    optimized_cost = calculate_schedule_cost(schedule, prices)
    # Worst case baseline cost for the optimized schedule
    unoptimized_cost = calculate_unoptimized_cost(schedule, prices)
    # Actual cost based on user's chosen run times
    actual_cost = calculate_schedule_cost(actual_usage, prices)
    
    # Calculate the worst case baseline for the actual usage pattern
    # This ensures actual_savings reflects the user's choices
    actual_unoptimized_cost = calculate_unoptimized_cost(actual_usage, prices)
    
    # Calculate savings
    potential_savings = unoptimized_cost - optimized_cost
    # Use the actual usage pattern's worst case as the baseline
    actual_savings = actual_unoptimized_cost - actual_cost
    
    # Update session state
    st.session_state.total_potential_savings += potential_savings
    st.session_state.total_actual_savings += actual_savings
    
    # Track daily savings
    st.session_state.daily_savings[date_str] = {
        "potential": potential_savings,
        "actual": actual_savings,
        "optimized_cost": optimized_cost,
        "unoptimized_cost": unoptimized_cost,
        "actual_cost": actual_cost
    }
    
    return potential_savings, actual_savings



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
            st.session_state.show_schedule_help = True
            st.rerun()
    
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

    # ── TAB LAYOUT -----------------------------------------------------------
    tab1, tab2 = st.tabs(
        ["Schedule Hours", "Usage Patterns (PMF)"]
    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1  –  schedule + price lane
    # ════════════════════════════════════════════════════════════════════════
    with tab1:

        # ──────────────────────────────────────────────────────────────────
        # PRICE LANE — identical 24-column grid, perfectly aligned
        # ──────────────────────────────────────────────────────────────────
        def _price_row_html(prices):
            """Return an HTML component that shows one 24-cell row coloured
            green→red according to price."""
            p_min, p_max = min(prices), max(prices)
            cells = ""
            for h, p in enumerate(prices):
                # improved 3-stop gradient  green→yellow→red
                ratio = (p - p_min) / (p_max - p_min + 1e-9)
                if ratio < 0.5:                           # 0-0.5: green→yellow
                    t = ratio * 2                        # 0-1
                    r = int(34  + (250-34)  * t)         #  #22c55e → #facc15
                    g = int(197 + (204-197) * t)
                    b = int(94  + (21 -94)  * t)
                else:                                    # 0.5-1: yellow→red
                    t = (ratio - 0.5) * 2               # 0-1
                    r = int(250 + (220-250) * t)         # #facc15 → #dc2626
                    g = int(204 + (38 -204) * t)
                    b = int(21  + (38 -21)  * t)
                fg = "#ffffff" if ratio > 0.55 else "#000000"
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

        # left label + HTML grid (using same column ratio as device rows for alignment)
        price_cols = st.columns([3, 24])
        price_cols[0].markdown("<b>Price (€/kWh)</b>", unsafe_allow_html=True)
        with price_cols[1]:
            _price_row_html(prices)
        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
        st.write("👉 Drag the blue window to confirm when devices actually ran.")

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

            if isinstance(new_s, int):
                s0 = new_s
                st.session_state.draggable_selections[device]["start_hour"] = s0

            # ─── ACTUAL USAGE  – shift device energy to the user-chosen window ───
            # I first capture the contiguous kWh profile that the optimiser proposed
            s_opt = _first_contiguous_block(sched_row, dur)            # original start
            energy_pattern = [float(sched_row[s_opt + j])              # kWh for each
                               for j in range(dur)]                    # phase hour

            # Now I replay that exact pattern at the user-selected start hour (s0)
            st.session_state.actual_usage.setdefault(device, [0.0] * 24)
            for h in range(24):
                if s0 <= h < s0 + dur:                                 # inside blue bar
                    idx = h - s0                                       # offset in pattern
                    st.session_state.actual_usage[device][h] = energy_pattern[idx]
                else:
                    st.session_state.actual_usage[device][h] = 0.0
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
                        iso
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



# Removed obsolete functions since the draggable window component now handles usage confirmation


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


def main() -> None:
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Initialize the optimization service
    global service 
    service = OptimisationService()
    
    # Configure page in full screen mode and hide menu
    st.set_page_config(
        page_title="EMS Scheduler",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
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
    
    # Create main layout
    left_col, right_col = st.columns([1, 4])
    
    # Left column - Device picker and constraints
    with left_col:
        render_device_picker()
        
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
    
    # Render floating help button
    render_floating_help_button()
    
    # Render modals based on session state
    if st.session_state.first_visit and st.session_state.show_welcome_modal:
        render_modal_overlay("welcome_modal", welcome_modal_content)
        
    if st.session_state.show_device_help:
        render_modal_overlay("device_help_modal", device_help_modal_content)
        
    if st.session_state.show_schedule_help:
        render_modal_overlay("schedule_help_modal", schedule_help_modal_content)
    
    # JavaScript handlers for modal close events
    modal_close_js = """
    <script>
        // Find all modal close buttons and add click handlers
        document.querySelectorAll('.modal-close').forEach(function(button) {
            button.addEventListener('click', function() {
                // Find parent modal
                const modal = this.closest('.modal-overlay');
                if (modal) {
                    // Send close event to Streamlit
                    const data = {
                        modal_id: modal.id,
                        action: 'close'
                    };
                    window.parent.postMessage({type: "streamlit:modal", action: data}, "*");
                }
            });
        });
        
        // Listen for messages from Streamlit
        window.addEventListener('message', function(event) {
            if (event.data.type === "streamlit:modal" && event.data.action.action === "close") {
                // Redirect to Streamlit to handle closing the modal
                window.location.href = "/?modal_id=" + event.data.action.modal_id;
            }
            // Handle help button click
            if (event.data.type === 'streamlit:help') {
                if (event.data.action === 'show') {
                    // Show help menu
                    window.parent.postMessage({type: "streamlit:showHelp"}, "*");
                }
            }
        });
    </script>
    """
    st.markdown(modal_close_js, unsafe_allow_html=True)
    
    # Mark first visit complete after initial page load
    if st.session_state.first_visit:
        st.session_state.first_visit = False


if __name__ == "__main__":
    main()
