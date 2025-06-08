#!/usr/bin/env python
"""
Grep-based Check for Forbidden Terms

This test suite scans the codebase for forbidden terminology and patterns
that violate the clean agent-based architecture.
"""

import unittest
import os
import re
from pathlib import Path

class TestForbiddenTerms(unittest.TestCase):
    """Test that forbidden terms don't exist in the codebase."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent
        
        # Define forbidden terms
        self.forbidden_terms = [
            "real agent",
            "real-agent", 
            "true agent",
            "fake agent",
            "real_agent",
            "true_agent",
            "fake_agent"
        ]
        
        # Define forbidden function patterns
        self.forbidden_functions = [
            "optimize_device_with_agent_logic",
            "run_simple_centralized_optimization", 
            "manual_optimization",
            "fallback_optimization",
            "simple_optimization"
        ]
        
        # Files and directories to check
        self.check_paths = [
            self.project_root / "scripts",
            self.project_root / "notebooks" / "agents",
            self.project_root / "notebooks" / "utils",
            self.project_root / "tests",
            self.project_root / "utils"
        ]
        
        # Files to check for documentation
        self.doc_files = [
            self.project_root / "README.md",
            self.project_root / "MLFLOW_INTEGRATION_SUMMARY.md"
        ]
        
        # Exclude test files that might mention forbidden terms for testing
        self.exclude_files = [
            "test_forbidden_terms.py",  # This file
            "test_no_fallback.py"      # May mention terms in test context
        ]

    def test_no_real_agent_terminology_in_python_files(self):
        """Test that Python files don't contain 'real agent' terminology."""
        violations = []
        
        for check_path in self.check_paths:
            if check_path.exists():
                for py_file in check_path.rglob("*.py"):
                    # Skip excluded files
                    if py_file.name in self.exclude_files:
                        continue
                        
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for term in self.forbidden_terms:
                        if term.lower() in content.lower():
                            # Find line numbers
                            lines = content.split('\n')
                            for i, line in enumerate(lines, 1):
                                if term.lower() in line.lower():
                                    violations.append(f"{py_file}:{i} - '{term}' found in: {line.strip()}")
        
        if violations:
            self.fail(f"Found forbidden terms in Python files:\n" + "\n".join(violations))

    def test_no_real_agent_terminology_in_docs(self):
        """Test that documentation files don't contain 'real agent' terminology."""
        violations = []
        
        for doc_file in self.doc_files:
            if doc_file.exists():
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for term in self.forbidden_terms:
                    if term.lower() in content.lower():
                        # Find line numbers
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if term.lower() in line.lower():
                                violations.append(f"{doc_file}:{i} - '{term}' found in: {line.strip()}")
        
        if violations:
            self.fail(f"Found forbidden terms in documentation:\n" + "\n".join(violations))

    def test_no_forbidden_function_names(self):
        """Test that forbidden function names don't exist."""
        violations = []
        
        for check_path in self.check_paths:
            if check_path.exists():
                for py_file in check_path.rglob("*.py"):
                    # Skip excluded files
                    if py_file.name in self.exclude_files:
                        continue
                        
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for func_name in self.forbidden_functions:
                        # Look for function definitions
                        pattern = rf"def\s+{re.escape(func_name)}\s*\("
                        if re.search(pattern, content):
                            violations.append(f"{py_file} - Forbidden function definition: {func_name}")
                            
                        # Look for function calls (less strict, only if very obvious)
                        pattern = rf"{re.escape(func_name)}\s*\("
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            # Get line number
                            line_num = content[:match.start()].count('\n') + 1
                            line = content.split('\n')[line_num - 1].strip()
                            violations.append(f"{py_file}:{line_num} - Forbidden function call: {line}")
        
        if violations:
            self.fail(f"Found forbidden functions:\n" + "\n".join(violations))

    def test_no_fallback_patterns_in_optimization_code(self):
        """Test that optimization code doesn't contain fallback patterns."""
        violations = []
        
        # Patterns that suggest fallback logic
        fallback_patterns = [
            r"try.*agent.*except.*fallback",
            r"if.*agent.*failed.*use.*manual",
            r"except.*optimization.*failed.*use",
            r"fallback.*to.*original",
            r"fallback.*to.*simple"
        ]
        
        for check_path in self.check_paths:
            if check_path.exists():
                for py_file in check_path.rglob("*.py"):
                    # Skip test files which might have these patterns for testing
                    if py_file.name.startswith("test_"):
                        continue
                        
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern in fallback_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            line = content.split('\n')[line_num - 1].strip()
                            violations.append(f"{py_file}:{line_num} - Fallback pattern: {line}")
        
        if violations:
            self.fail(f"Found fallback patterns in optimization code:\n" + "\n".join(violations))

    def test_no_manual_optimization_loops(self):
        """Test that manual optimization loops don't exist."""
        violations = []
        
        # Patterns that suggest manual optimization
        manual_patterns = [
            r"for.*price.*sort",
            r"manual.*optimization",
            r"greedy.*loop",
            r"simple.*centralized.*optimization",
            r"price.*sort.*manual"
        ]
        
        for check_path in self.check_paths:
            if check_path.exists():
                for py_file in check_path.rglob("*.py"):
                    # Skip test files
                    if py_file.name.startswith("test_"):
                        continue
                        
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern in manual_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            line = content.split('\n')[line_num - 1].strip()
                            violations.append(f"{py_file}:{line_num} - Manual optimization pattern: {line}")
        
        if violations:
            self.fail(f"Found manual optimization patterns:\n" + "\n".join(violations))

    def test_imports_use_agent_terminology(self):
        """Test that import statements use clean agent terminology."""
        violations = []
        
        # Forbidden import patterns
        forbidden_import_patterns = [
            r"from.*real.*agent",
            r"import.*real.*agent", 
            r"from.*fake.*agent",
            r"import.*fake.*agent"
        ]
        
        for check_path in self.check_paths:
            if check_path.exists():
                for py_file in check_path.rglob("*.py"):
                    # Skip this test file
                    if py_file.name in self.exclude_files:
                        continue
                        
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern in forbidden_import_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            line = content.split('\n')[line_num - 1].strip()
                            violations.append(f"{py_file}:{line_num} - Forbidden import: {line}")
        
        if violations:
            self.fail(f"Found forbidden import patterns:\n" + "\n".join(violations))

    def test_comments_use_clean_terminology(self):
        """Test that comments use clean agent terminology."""
        violations = []
        
        for check_path in self.check_paths:
            if check_path.exists():
                for py_file in check_path.rglob("*.py"):
                    # Skip test files
                    if py_file.name.startswith("test_"):
                        continue
                        
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    for i, line in enumerate(lines, 1):
                        # Check comments (lines starting with # or containing #)
                        if '#' in line:
                            comment_part = line[line.index('#'):]
                            for term in self.forbidden_terms:
                                if term.lower() in comment_part.lower():
                                    violations.append(f"{py_file}:{i} - '{term}' in comment: {line.strip()}")
        
        if violations:
            self.fail(f"Found forbidden terms in comments:\n" + "\n".join(violations))

if __name__ == '__main__':
    unittest.main(verbosity=2)