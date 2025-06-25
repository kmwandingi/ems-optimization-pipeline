#!/usr/bin/env python
"""
Simple test script to verify Python execution
"""

import sys
import os
import datetime

print("==========================================")
print("TEST SCRIPT EXECUTING")
print("==========================================")
print(f"Current time: {datetime.datetime.now()}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Create a simple file to verify file writing works
with open("test_output.txt", "w") as f:
    f.write("Test script executed successfully\n")
    f.write(f"Time: {datetime.datetime.now()}\n")

print("Test completed - check for test_output.txt")
print("==========================================")
