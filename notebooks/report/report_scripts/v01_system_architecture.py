# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ V-1: SYSTEM ARCHITECTURE DIAGRAM                                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
#
# Objective:
# My purpose is to generate a high-level system architecture diagram that visually
# represents the key components of the EMS and their interactions. This diagram must
# be clean, professional, and accurately reflect the actual agent-based structure
# of the codebase, as per the user's strict rules. I will use the Graphviz library
# for this purpose.

import os
from pathlib import Path
from graphviz import Digraph

# --- Configuration & Setup ---
# I'm setting up the output path. The diagram will be saved in the 'assets' folder
# for easy inclusion in the LaTeX report.
try:
    project_root = Path(__file__).resolve().parents[3]
except (NameError, IndexError):
    project_root = Path.cwd()

OUTPUT_DIR = project_root / 'notebooks' / 'report' / 'report_1' / 'assets'
OUTPUT_FILE = OUTPUT_DIR / 'v01_system_architecture'

# I ensure the output directory exists.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 1: DIAGRAM DEFINITION                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def create_architecture_diagram():
    """
    Creates and saves the system architecture diagram using Graphviz.
    My design choices here are guided by clarity and academic standards. I will use
    subgraphs to group related components (e.g., Data Sources, Core Agents) and
    standard shapes and colors to ensure readability.
    """
    dot = Digraph('EMS_Architecture', comment='Energy Management System Architecture')
    dot.attr('graph', rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')

    # --- Node Styles ---
    # I'm defining styles to maintain visual consistency.
    node_styles = {
        'optimizer': {'shape': 'octagon', 'style': 'filled', 'fillcolor': '#ff7f0e', 'fontname': 'Helvetica', 'fontsize': '12'},
        'agent': {'shape': 'box', 'style': 'filled', 'fillcolor': '#1f77b4', 'fontcolor': 'white', 'fontname': 'Helvetica', 'fontsize': '11'},
        'data': {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#2ca02c', 'fontname': 'Helvetica', 'fontsize': '10'},
        'external': {'shape': 'house', 'style': 'filled', 'fillcolor': 'lightgrey', 'fontname': 'Helvetica'}
    }

    # --- Main Components ---
    # I am defining the nodes based on the actual components in the project.
    dot.node('GlobalOptimizer', 'Global Optimizer', **node_styles['optimizer'])

    # --- Core Agents Subgraph ---
    with dot.subgraph(name='cluster_agents') as c:
        c.attr(label='Core Agents', style='filled', color='lightgrey')
        c.node('PVAgent', 'PV Agent', **node_styles['agent'])
        c.node('BatteryAgent', 'Battery Agent', **node_styles['agent'])
        c.node('EVAgent', 'EV Agent', **node_styles['agent'])
        c.node('FlexibleDevice', 'Flexible Devices\n(e.g., Washing Machine)', **node_styles['agent'])
        c.node('GridAgent', 'Grid Agent', **node_styles['agent'])
        c.node('ProbabilityModelAgent', 'Probability Model\nAgent', **node_styles['agent'])

    # --- Data Sources Subgraph ---
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='Data Sources & External Systems', style='filled', color='lightgrey')
        c.node('WeatherForecast', 'Weather Forecast', **node_styles['data'])
        c.node('PriceData', 'Price Data', **node_styles['data'])
        c.node('DeviceSpecs', 'Device Specs', **node_styles['data'])
        c.node('HistoricalUsage', 'Historical Usage', **node_styles['data'])
        c.node('Grid', 'Utility Grid', **node_styles['external'])

    # --- Edges (Interactions) ---
    # I am now defining the relationships and data flows between components.
    # These connections reflect the actual logic of the optimization pipeline.

    # Data sources to agents
    dot.edge('WeatherForecast', 'PVAgent', label='Irradiance')
    dot.edge('PriceData', 'GridAgent', label='Tariffs')
    dot.edge('DeviceSpecs', 'FlexibleDevice', label='Constraints')
    dot.edge('HistoricalUsage', 'ProbabilityModelAgent', label='Trains on past usage')

    # Probability agent to flexible devices
    dot.edge('ProbabilityModelAgent', 'FlexibleDevice', label='Provides P(start)')

    # Agents to Optimizer
    dot.edge('PVAgent', 'GlobalOptimizer', label='PV Generation Forecast')
    dot.edge('BatteryAgent', 'GlobalOptimizer', label='State & Constraints')
    dot.edge('EVAgent', 'GlobalOptimizer', 'label')
    dot.edge('FlexibleDevice', 'GlobalOptimizer', label='Flexibility Model')
    dot.edge('GridAgent', 'GlobalOptimizer', label='Grid Constraints & Cost')

    # Optimizer to Agents (Control Signals)
    dot.edge('GlobalOptimizer', 'BatteryAgent', label='Charge/Discharge Schedule')
    dot.edge('GlobalOptimizer', 'EVAgent', label='Charge Schedule')
    dot.edge('GlobalOptimizer', 'FlexibleDevice', label='Optimal Start Time')
    dot.edge('GlobalOptimizer', 'GridAgent', label='Grid Import/Export Plan')

    # Grid to System
    dot.edge('Grid', 'GridAgent', label='Physical Connection')

    return dot

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 2: MAIN EXECUTION BLOCK                                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    try:
        print("--- Generating Visual V-1: System Architecture Diagram ---")
        diagram = create_architecture_diagram()
        # I'm saving the output in both PNG and PDF for flexibility.
        diagram.render(OUTPUT_FILE, format='png', view=False, cleanup=True)
        diagram.render(OUTPUT_FILE, format='pdf', view=False, cleanup=True)
        print(f"✅ Visual V-1 successfully generated and saved to: {OUTPUT_FILE}.png/.pdf")
    except Exception as e:
        print(f"❌ An error occurred during diagram generation: {e}")
        print("Please ensure Graphviz is installed and accessible in your system's PATH.")
        print("You can install it from https://graphviz.org/download/ or via a package manager.")
