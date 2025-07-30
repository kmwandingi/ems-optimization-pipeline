# ────────────────────────────────────────────────────────────────
#  ProbabilityModelAgent – Adaptive-PMF edition with DuckDB priors
#  100 % compatible with your existing notebooks & scripts
#  2025-05-20
# ────────────────────────────────────────────────────────────────
from __future__ import annotations

import math, types, warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


# ╔══════════════════════════════════════════════════════════════╗
#  Hyper-parameters (can be overwritten per-instance)
# ╚══════════════════════════════════════════════════════════════╝
# Base learning rate parameters
LR_TAU      = 20          # Controls learning rate decay (higher = slower decay)
LR_MIN      = 0.002      # Minimum learning rate
LR_MAX      = 0.10       # Maximum learning rate

# Update capping parameters
CAP_MIN     = 0.005      # Minimum update cap
CAP_MAX     = 0.03       # Maximum update cap

# Burn-in period
BURNIN_DAYS = 0          # Number of days for burn-in period
LR_BURNIN   = 0.005      # Learning rate during burn-in

# Other parameters
PROBABILITY_THRESHOLD = 0.05      # Used for allowed_hours pruning


# ╔══════════════════════════════════════════════════════════════╗
#  ProbabilityModelAgent
# ╚══════════════════════════════════════════════════════════════╝
class ProbabilityModelAgent:
    """Per-event Adaptive PMF with DuckDB priors support."""

    # class-level defaults
    LR_TAU      = LR_TAU
    LR_MIN      = LR_MIN
    LR_MAX      = LR_MAX
    CAP_MIN     = CAP_MIN
    CAP_MAX     = CAP_MAX
    BURNIN_DAYS = BURNIN_DAYS
    LR_BURNIN   = LR_BURNIN
    PROBABILITY_THRESHOLD = PROBABILITY_THRESHOLD

    # ───────── initialisation ───────────────────────────────────
    def __init__(self, prob_dist_df: Optional[pd.DataFrame] = None, building_id: str = None, use_duckdb_priors: bool = True):
        self.prob_dist_df = prob_dist_df            # optional parquet of priors (backward compatibility)
        self.building_id = building_id
        self.use_duckdb_priors = use_duckdb_priors
        self.latest_distributions: Dict[str, Dict[int,float]] = {}
        self.observation_counts: Dict[str, int] = {}
        self.probability_updates_history: Dict[str, List[Dict]] = {}
        self.first_day_seen: Dict[str, str] = {}   # key → "YYYY-MM-DD"
        print("ProbabilityModelAgent ready (adaptive PMF with DuckDB priors)")

    # ───────── PRIORS FROM DUCKDB ───────────────────────────────
    def get_prior_from_duckdb(self, device_type: Optional[str]) -> Dict[int,float]:
        """Return device-specific hour probabilities from parquet file or uniform distribution."""
        if not self.use_duckdb_priors or not device_type:
            return {h: 1.0/24.0 for h in range(24)}
        
        # Skip DuckDB entirely and directly load from parquet file
        try:
            import os
            import pandas as pd
            from pathlib import Path
            
            # Find the project root directory using relative path
            script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            project_root = script_dir.parent.parent  # Go up two levels from agents/ folder
            
            # Known location for the probability file
            prob_file = project_root / 'notebooks' / 'probabilities' / 'device_hourly_probabilities.parquet'
            
            if prob_file.exists():
                # Load the parquet file - device_type is the index with integer columns 0-23 for hours
                df = pd.read_parquet(prob_file)
                
                # Check if the requested device type exists in the index
                if device_type in df.index:
                    # Get the row for this device type
                    row = df.loc[device_type]
                    
                    # Extract hour probabilities (columns are integer values 0-23)
                    prior = {h: float(row[h]) for h in range(24)}
                    
                    # Normalize to ensure it's a valid probability distribution
                    total = sum(prior.values())
                    if total > 0:
                        prior = {h: v/total for h, v in prior.items()}
                        print(f"✓ Loaded prior for {device_type}: max={max(prior.values()):.3f}, min={min(prior.values()):.3f}")
                        return prior
                    
                print(f"⚠ Device '{device_type}' not found in probability file. Available devices: {df.index.tolist()[:5]}...")
        except Exception as e:
            print(f"⚠ Failed to load probability file: {e}")
        
        # Fallback to uniform distribution
        print(f"→ Using uniform fallback for {device_type}")
        return {h: 1.0/24.0 for h in range(24)}

    # ───────── PRIORS (backward compatibility) ──────────────────
    def get_prior_from_parquet(self, device_type: Optional[str]) -> Dict[int,float]:
        """Return parquet prior if available else DuckDB/uniform."""
        if (self.prob_dist_df is None) or (device_type not in self.prob_dist_df.index):
            return self.get_prior_from_duckdb(device_type)
        row   = self.prob_dist_df.loc[device_type]
        prior = {h: float(row[h]) for h in range(24)}
        s     = sum(prior.values())
        return {h:v/s for h,v in prior.items()}

    # ───────── tiny helpers ─────────────────────────────────────
    @staticmethod
    def _days_since(start: Optional[str], today:str) -> int:
        if start is None or start == 'PRIOR':
            return 0
        return (pd.to_datetime(today) - pd.to_datetime(start)).days

    @staticmethod
    def js_div(p: Dict[int,float], q: Dict[int,float]) -> float:
        pa = np.array([p.get(h,0) for h in range(24)])+1e-12
        qa = np.array([q.get(h,0) for h in range(24)])+1e-12
        pa, qa = pa/pa.sum(), qa/qa.sum()
        return float(jensenshannon(pa, qa))

    def _recent_js(self, updates: List[Dict], window:int=5) -> float:
        seq = [u for u in updates if u['day']!='PRIOR']
        if len(seq) < 2: return 0.0
        seq = seq[-min(window+1,len(seq)):]
        vals=[ self.js_div(seq[i-1]['distribution'], seq[i]['distribution'])
               for i in range(1,len(seq)) ]
        return float(np.mean(vals))

    @staticmethod
    def entropy(pmf: Dict[int,float]) -> float:
        return -sum(p*math.log(p+1e-12) for p in pmf.values())

    @staticmethod
    def day_type(date_str:str) -> str:
        try:
            return 'weekend' if pd.to_datetime(date_str).weekday()>=5 else 'weekday'
        except Exception:
            return 'weekday'

    # ───────── adaptive learning-rate ────────────────────────────
    def _adaptive_lr(self, device, day_type: str, today: str) -> float:
        # Handle PRIOR case
        if today == "PRIOR":
            return self.LR_BURNIN
            
        # first calendar day: always LR_BURNIN
        k = device.device_name.split("_")[0]  # Remove phase suffix if present
        if k not in self.first_day_seen:
            self.first_day_seen[k] = today
        if today == self.first_day_seen[k]:
            return self.LR_BURNIN
            
        # original burn-in over multiple days
        if self._days_since(device.first_real_day, today) < self.BURNIN_DAYS:
            return self.LR_BURNIN
            
        # JS-boost + Dirichlet decay
        lr_base = 1.0 / (device.observation_count + self.LR_TAU)
        recent_js = self._recent_js(device.probability_updates)
        boost = 1.0 + min(recent_js * 50.0, 0.5)
        return max(self.LR_MIN, min(self.LR_MAX, lr_base * boost))
    
    # ───────── core per-event update (delta+cap) ────────────────
    def update_user_probability_model(
            self,
            device: Any,
            day: str,
            actual_hour: int,
            max_per_day_update: bool=False,
            day_type: Optional[str]=None,
            daily_distribution: Optional[Dict[int,float]]=None
    ) -> Any:
        # — first-time setup —
        if not hasattr(device,'first_update_day'):
            device.first_update_day = day
        
        # Initialize proper priors (learned or uniform) - override helper.py defaults
        needs_prior_init = (not hasattr(device,'hour_probability') or 
                           day == "PRIOR" or
                           (hasattr(device, 'probability_updates') and 
                            device.probability_updates and 
                            device.probability_updates[0].get('day') == 'INITIAL_PRIOR'))
        
        if needs_prior_init:
            dtype = getattr(device,'device_type',None)
            if dtype is None and hasattr(device,'device_name'):
                # Extract device type from full name
                dn = device.device_name.lower()
                # Try to extract device type from name patterns
                for possible_type in ['dishwasher', 'ev', 'electric_vehicle', 'freezer', 'heat_pump', 'refrigerator', 'washing_machine', 'tumble_dryer']:
                    if possible_type in dn or possible_type.replace('_', '') in dn:
                        dtype = possible_type
                        break
                # If still no match, use the last part of the device name
                if dtype is None:
                    dtype = dn.split('_')[-1]
                        
            # Use DuckDB priors first, then parquet fallback
            if self.use_duckdb_priors:
                device.hour_probability = self.get_prior_from_duckdb(dtype)
            else:
                device.hour_probability = self.get_prior_from_parquet(dtype)
                
            device.observation_count = 0
            device.probability_updates=[{
                'day':'PRIOR','actual_hour':None,
                'distribution':device.hour_probability.copy(),
                'learning_rate':0.0,
                'entropy':self.entropy(device.hour_probability),
                'day_type': day_type or 'weekday',
                'js_prior':0.0,'js_prev':0.0
            }]

        # skip duplicate same-day call
        if max_per_day_update and getattr(device,'last_update_day',None)==day:
            return device

        # count updates on this calendar day
        device.updates_today = getattr(device, "updates_today", 0) + 1
        if not hasattr(device, "first_real_day"):
            device.first_real_day = day
        device.last_update_day = day

        p_old = device.hour_probability.copy()
        lr = self._adaptive_lr(device, day_type or self.day_type(day), day)
        
        # cap so the *sum* of today's deltas ≤ CAP_MAX
        cap_day = max(self.CAP_MIN, min(self.CAP_MAX, self.CAP_MAX * (lr / self.LR_MAX)))
        effective_cap = cap_day / device.updates_today

        # target vector
        if daily_distribution is None:
            target = {h: 1.0 if h==actual_hour else 0.0 for h in range(24)}
            peak_h = actual_hour
        else:
            target = daily_distribution
            peak_h = max(target, key=target.get)

        # Calculate update with momentum-like behavior
        if not hasattr(device, 'update_momentum'):
            device.update_momentum = {h: 0.0 for h in range(24)}
            
        # Simple delta update with capping
        for h in range(24):
            delta = lr * (target[h] - p_old[h])
            delta = np.clip(delta, -effective_cap, effective_cap)
            device.hour_probability[h] = max(0.0, p_old[h] + delta)
        
        # Normalize to ensure valid probability distribution
        total = sum(device.hour_probability.values())
        if total > 0:
            for h in device.hour_probability:
                device.hour_probability[h] /= total
        else:
            # Fallback to uniform if something went wrong
            for h in device.hour_probability:
                device.hour_probability[h] = 1.0 / 24.0
                
        # divergence metrics
        js_prior = self.js_div(device.hour_probability,
                               device.probability_updates[0]['distribution'])
        js_prev  = self.js_div(device.hour_probability, p_old)

        # bookkeeping
        device.observation_count += 1
        ent = self.entropy(device.hour_probability)
        device.probability_updates.append({
            'day':day,'actual_hour':peak_h,'distribution':device.hour_probability.copy(),
            'learning_rate':lr,'entropy':ent,
            'day_type': day_type or self.day_type(day),
            'js_prior':js_prior,'js_prev':js_prev
        })
        self.track_convergence_metrics(device, p_old, device.hour_probability)
        return device

    # ───────── convergence metrics (KL, JS, etc.) ───────────────
    def track_convergence_metrics(self, device, old_p, new_p):
        pa = np.array([old_p.get(h,0) for h in range(24)])+1e-12
        qa = np.array([new_p.get(h,0) for h in range(24)])+1e-12
        pa, qa = pa/pa.sum(), qa/qa.sum()
        js = float(jensenshannon(pa, qa))
        kl = float(np.sum(qa*np.log(qa/pa)))
        max_h = int(max(new_p, key=new_p.get))
        max_p = float(new_p[max_h])
        top3  = sorted(new_p, key=new_p.get, reverse=True)[:3]
        device.probability_updates[-1].update({
            'js_divergence':js,'kl_divergence':kl,
            'max_probability':max_p,'max_hour':max_h,'top3_hours':top3
        })

    # ╔═══════════════════════════════════════════════════════════╗
    #  UTILITY: enforce consecutive hours  (unchanged)
    # ╚═══════════════════════════════════════════════════════════╝
    def enforce_consecutive_hours(self, allowed_hours: List[int], phases: List[Dict]) -> List[int]:
        total_duration = sum(ph.get("duration",1) for ph in phases) if phases else 1
        if total_duration <= 1: return sorted(allowed_hours)

        ah_sorted = sorted(set(allowed_hours))
        final_set = set()
        for h in ah_sorted:
            needed = [h+i for i in range(total_duration)]
            if all(hh in ah_sorted for hh in needed): final_set.update(needed)

        result = sorted(final_set)
        if not result and total_duration>1:
            center = ah_sorted[len(ah_sorted)//2] if ah_sorted else 12
            start  = max(0, center-total_duration)
            end    = min(24, start+total_duration*2)
            result = list(range(start,end))
            print(f"WARNING: enforce_consecutive_hours fallback window {start}-{end}")
        return result

    # ╔═══════════════════════════════════════════════════════════╗
    #  TRAIN  –  identical to your previous version except we
    #  call update_user_probability_model once PER EVENT.
    # ╚═══════════════════════════════════════════════════════════╝
    def train(
        self,
        building_id: str,
        days_list: List[str],
        device_specs: Dict,
        weather_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        parquet_dir: str = "processed_data",
        max_building_load: float = 10.0,
        battery_params: Optional[Dict] = None,
        flexible_params: Optional[Dict] = None,
        grid_params: Optional[Dict] = None,
        pv_params: Optional[Dict] = None,
        cleaner: Any = None
    ) -> Tuple[Dict, Dict]:
        updated_specs = device_specs.copy()
        device_probabilities: Dict[str, Dict[str,Any]] = {}
        print(f"Training PMFs for building={building_id} over {len(days_list)} days")
        
        # Set building_id for DuckDB priors if not already set
        if self.building_id is None:
            self.building_id = building_id

        for i,single_day in enumerate(days_list):
            print(f"  Day {i+1}/{len(days_list)} : {single_day}")
            # load raw devices for the day
            from utils.helper import load_building_day_devices
            raw_devices = load_building_day_devices(
                building_id,single_day,parquet_dir,updated_specs)

            for dev in raw_devices:
                dev_name = dev.device_name
                # first encounter → create prior history entry
                if dev_name not in self.latest_distributions:
                    self.update_user_probability_model(
                        device=dev, day="PRIOR", actual_hour=0,
                        max_per_day_update=True, day_type=self.day_type(single_day))
                    self.latest_distributions[dev_name] = dev.hour_probability.copy()
                    self.observation_counts[dev_name]   = dev.observation_count
                    self.probability_updates_history[dev_name]=dev.probability_updates.copy()
                else:
                    dev.hour_probability  = self.latest_distributions[dev_name].copy()
                    dev.observation_count = self.observation_counts[dev_name]
                    dev.probability_updates=self.probability_updates_history[dev_name].copy()

                # Check if data has required columns
                if 'hour' not in dev.data.columns:
                    if 'utc_timestamp' in dev.data.columns:
                        dev.data['hour'] = dev.data['utc_timestamp'].dt.hour
                    else:
                        print(f"Warning: No hour data for {dev_name} on {single_day}")
                        continue
                
                # Get usage by hour
                usage_by_hour = dev.data.groupby("hour")[dev_name].sum()
                # NEW ▼ — trigger exactly once for each *hour* that saw any usage
                active_hours = usage_by_hour[usage_by_hour > 0].index.tolist()
                for hr in active_hours:
                    self.update_user_probability_model(
                        device=dev, day=single_day, actual_hour=int(hr),
                        max_per_day_update=False, day_type=self.day_type(single_day))
                # ▲ NEW
                # cache back
                self.latest_distributions[dev_name] = dev.hour_probability.copy()
                self.observation_counts[dev_name]   = dev.observation_count
                self.probability_updates_history[dev_name]=dev.probability_updates.copy()
                device_probabilities[dev_name] = {
                    'hour_probability':dev.hour_probability.copy(),
                    'observation_count':dev.observation_count,
                    'estimated_preferred_hour':max(dev.hour_probability,key=dev.hour_probability.get),
                    'probability_updates':dev.probability_updates.copy()
                }

        return updated_specs, device_probabilities

    # ╔═══════════════════════════════════════════════════════════╗
    #  Scheduling + helper (unchanged, except pmfs already updated)
    # ╚═══════════════════════════════════════════════════════════╝
    def add_update_device_constraints_method(self, optimizer):
        """Monkey-patch an optimizer instance with a constraints updater."""
        def update_device_constraints(self_opt, devices):
            print("  Updating optimizer device constraints …")
            self_opt.devices = devices
            device_specs = {}
            for dev in devices:
                if not hasattr(dev,'spec'): continue
                device_id = dev.device_name
                device_specs[device_id] = dev.spec.copy()
                if 'allowed_hours' in dev.spec:
                    dev.allowed_hours = dev.spec['allowed_hours']
                if 'phases' in dev.spec and not hasattr(dev,'phases'):
                    dev.phases = dev.spec['phases']
            self_opt.device_specs = device_specs
            if hasattr(self_opt,'constraints_initialized'):
                self_opt.constraints_initialized = False
            return True
        optimizer.update_device_constraints = types.MethodType(update_device_constraints, optimizer)

    def run_next_day_scheduling(
        self,
        evaluation_days: List[str],
        device_specs: Dict,
        building_id: str,
        weather_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        parquet_dir: str = "processed_data",
        max_building_load: float = 10.0,
        battery_params: Optional[Dict] = None,
        flexible_params: Optional[Dict] = None,
        grid_params: Optional[Dict] = None,
        pv_params: Optional[Dict] = None,
        peak_prob_cut: float = 0.80,
        min_margin: int = 1,
        max_margin: int = 4,
        cleaner: Any = None
    ) -> List[Tuple[List[Any], Any, bool]]:
        results = []
        print(f"\n[Day-Ahead Scheduling] building={building_id}")
        for i,single_day in enumerate(evaluation_days):
            print(f"  Scheduling day {i+1}/{len(evaluation_days)} : {single_day}")
            # load optimizer + devices
            from utils.direct_optimizer import run_building_optimization_single_day_direct
            devices, optimizer, has_pv = run_building_optimization_single_day_direct(
                building_id, single_day, False, device_specs,
                parquet_dir, weather_df, forecast_df,
                max_building_load,battery_params,flexible_params,
                grid_params,pv_params)
            # transfer learned pmfs
            for dev in devices:
                if dev.device_name in self.latest_distributions:
                    dev.hour_probability=self.latest_distributions[dev.device_name].copy()
            # patch optimizer and optimize
            if optimizer and not hasattr(optimizer,'update_device_constraints'):
                self.add_update_device_constraints_method(optimizer)
            if optimizer:
                optimizer.update_device_constraints(devices)
                optimizer.optimize_for_weekday()
            results.append((devices,optimizer,has_pv))
        return results