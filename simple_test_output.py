import os
from pathlib import Path

# Define a known absolute path within the project directory
output_file_path = Path("D:/Kenneth - TU Eindhoven/Jads/Graduation Project 2024-2025/ems_project/ems-optimization-pipeline/simple_test_output.txt")

try:
    with open(output_file_path, "w") as f:
        f.write("This is a test message from simple_test_output.py\n")
        f.write(f"Current working directory: {os.getcwd()}\n")
    print(f"Successfully wrote to {output_file_path}")
except Exception as e:
    print(f"Error writing file: {e}")

