import os
from pathlib import Path

project_root = Path("D:/Kenneth - TU Eindhoven/Jads/Graduation Project 2024-2025/ems_project/ems-optimization-pipeline")
file_path = project_root / "test_write_output.txt"

try:
    with open(file_path, "w") as f:
        f.write("Hello from test_file_write.py!\n")
        f.write(f"Attempting to write to: {file_path}\n")
    print(f"Successfully attempted to write to {file_path}")
except Exception as e:
    print(f"Error writing file: {e}")