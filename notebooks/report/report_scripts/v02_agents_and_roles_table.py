# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ V-2: AGENTS AND ROLES TABLE                                                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
#
# Objective:
# My goal is to create a clear and professional LaTeX table that defines the roles
# and interactions of each agent in the EMS. This script will programmatically
# generate a standalone .tex file, ensuring the table is consistent, easy to
# maintain, and ready for direct inclusion in the report.

import os
from pathlib import Path
import pandas as pd

# --- Configuration & Setup ---
try:
    project_root = Path(__file__).resolve().parents[3]
except (NameError, IndexError):
    project_root = Path.cwd()

OUTPUT_DIR = project_root / 'notebooks' / 'report' / 'report_1' / 'assets'
OUTPUT_FILE = OUTPUT_DIR / 'v02_agents_and_roles_table.tex'

# I ensure the output directory exists.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 1: TABLE CONTENT DEFINITION                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

# I'm defining the table's content here as a list of dictionaries.
# This structure is clean, readable, and easy to update if agent roles change.
# The descriptions are based on the actual functionality of the agents in the codebase.
AGENT_DATA = [
    {
        'Agent': 'Global Optimizer',
        'Role': 'Central decision-making unit. Solves a master optimization problem to coordinate all assets and minimize system-wide costs.',
        'Key Interactions': 'Receives constraints and forecasts from all agents; sends optimal schedules and control signals back to them.'
    },
    {
        'Agent': 'Probability Model Agent',
        'Role': 'Learns and predicts user behavior patterns for flexible devices from historical data. Provides probabilistic forecasts for device usage.',
        'Key Interactions': 'Consumes historical usage data; provides start-hour probability distributions to Flexible Device agents.'
    },
    {
        'Agent': 'Flexible Device Agent',
        'Role': 'Models the operational constraints and user preferences for a single deferrable appliance (e.g., washing machine, dishwasher).',
        'Key Interactions': 'Receives start-hour probabilities; provides a flexibility model (energy, duration, constraints) to the Global Optimizer.'
    },
    {
        'Agent': 'Battery Agent',
        'Role': 'Manages the state of the battery energy storage system (BESS), including state of charge (SoC), charge/discharge limits, and efficiency.',
        'Key Interactions': 'Provides its current state and operational constraints to the Global Optimizer; receives and executes the optimal charge/discharge schedule.'
    },
    {
        'Agent': 'EV Agent',
        'Role': 'Manages the electric vehicle charging process, considering target SoC, departure time, and charging constraints.',
        'Key Interactions': 'Provides its state and charging requirements to the Global Optimizer; receives and executes the optimal charging schedule.'
    },
    {
        'Agent': 'PV Agent',
        'Role': 'Forecasts solar power generation based on weather data (e.g., solar irradiance) and PV panel specifications.',
        'Key Interactions': 'Consumes weather forecasts; provides the predicted PV generation profile to the Global Optimizer.'
    },
    {
        'Agent': 'Grid Agent',
        'Role': 'Represents the connection to the utility grid. Manages electricity tariffs (import/export prices) and grid capacity constraints.',
        'Key Interactions': 'Provides price signals and grid constraints to the Global Optimizer; executes the final grid import/export plan.'
    },
]

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 2: LATEX TABLE GENERATION                                         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

def generate_latex_table(data: list, output_path: Path):
    """
    Generates and saves a LaTeX table from the structured agent data.
    I'm using the 'booktabs' package for a more professional look and feel, which is
    standard in academic publications. The column widths are set to ensure the table
    fits well on the page.
    """
    # I'm using a pandas DataFrame to easily convert the structured data to a LaTeX string.
    df = pd.DataFrame(data)
    
    # I need to escape special LaTeX characters to prevent compilation errors.
    for col in df.columns:
        df[col] = df[col].str.replace('&', '\\&', regex=False)
        df[col] = df[col].str.replace('%', '\\%', regex=False)
        df[col] = df[col].str.replace('$', '\\$', regex=False)
        df[col] = df[col].str.replace('#', '\\#', regex=False)
        df[col] = df[col].str.replace('_', '\\_', regex=False)
        df[col] = df[col].str.replace('{', '\\{', regex=False)
        df[col] = df[col].str.replace('}', '\\}', regex=False)
        df[col] = df[col].str.replace('~', '\\textasciitilde{}', regex=False)
        df[col] = df[col].str.replace('^', '\\textasciicircum{}', regex=False)

    # The to_latex() method is a powerful way to generate the table.
    # The formatters ensure text wraps correctly within the cells.
    latex_string = df.to_latex(
        index=False,
        header=True,
        column_format='>{\\raggedright\\arraybackslash}p{0.15\\textwidth} >{\\raggedright\\arraybackslash}p{0.4\\textwidth} >{\\raggedright\\arraybackslash}p{0.35\\textwidth}',
        longtable=False,
        escape=False, # I've already handled escaping
        caption='Description of Agents, Roles, and Key Interactions within the EMS.',
        label='tab:agents_and_roles',
        position='H' # Using 'H' from the 'float' package for strict positioning
    )

    # I'm replacing the default pandas top/bottom rules with booktabs for a cleaner look.
    latex_string = latex_string.replace('\\toprule', '\\toprule ')
    latex_string = latex_string.replace('\\midrule', '\\midrule ')
    latex_string = latex_string.replace('\\bottomrule', '\\bottomrule ')
    
    # Adding required LaTeX packages to the header of the file
    header = """\\NeedsTeXFormat{LaTeX2e}
\\documentclass{standalone}
\\usepackage{booktabs}
\\usepackage{graphicx}
\\usepackage[T1]{fontenc}
\\begin{document}
"""
    footer = "\\end{document}"
    final_latex_code = header + latex_string + footer

    with open(output_path, 'w') as f:
        f.write(final_latex_code)

    print(f"✅ Visual V-2 successfully generated and saved to: {output_path}")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ SECTION 3: MAIN EXECUTION BLOCK                                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    try:
        print("--- Generating Visual V-2: Agents and Roles Table ---")
        generate_latex_table(AGENT_DATA, OUTPUT_FILE)
    except Exception as e:
        print(f"❌ An error occurred during table generation: {e}")
