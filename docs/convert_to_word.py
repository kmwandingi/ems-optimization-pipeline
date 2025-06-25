"""Convert HTML with MathJax to Word document using Pandoc."""

import os
import subprocess
import sys
from pathlib import Path

def check_pandoc_installed():
    """Check if Pandoc is installed on the system."""
    # First try the normal way (pandoc in PATH)
    try:
        result = subprocess.run(["pandoc", "--version"], capture_output=True, text=True)
        return ("pandoc", result.returncode == 0)
    except FileNotFoundError:
        pass
    
    # Try common installation paths on Windows
    windows_paths = [
        r"C:\Program Files\Pandoc\pandoc.exe",
        r"C:\Program Files (x86)\Pandoc\pandoc.exe",
        os.path.expanduser(r"~\AppData\Local\Pandoc\pandoc.exe"),
        os.path.expanduser(r"~\AppData\Roaming\Pandoc\pandoc.exe"),
        # Add additional likely locations
    ]
    
    # Check each path
    for path in windows_paths:
        if os.path.exists(path):
            try:
                result = subprocess.run([path, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    return (path, True)
            except Exception:
                pass
    
    # Not found
    return (None, False)

def convert_html_to_docx(html_path, docx_path, pandoc_path="pandoc"):
    """Convert HTML to DOCX using Pandoc."""
    print(f"Converting {html_path} to Word document...")
    
    # Run pandoc to convert HTML to DOCX
    try:
        # Use --mathjax to ensure proper math formula conversion
        # Use --reference-doc to specify a template docx if you have one
        cmd = [
            pandoc_path,
             "--from=markdown+pipe_tables+grid_tables+multiline_tables",  # Enhanced table support
            "--to=docx",
            "--mathjax",
            "--standalone",
            f"--output={docx_path}",
            html_path
        ]
        
        # If you have a reference docx template, uncomment the following line
        # cmd.insert(5, "--reference-doc=template.docx")
        
        subprocess.run(cmd, check=True)
        print(f"Successfully converted to {docx_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {e}")
        return False

def main():
    # Define file paths
    project_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    html_path = project_dir / "EMS_Technical_Report.md"
    docx_path = project_dir / "EMS_Technical_Report.docx"
    
    # Check if HTML file exists
    if not html_path.exists():
        print(f"Error: HTML file {html_path} does not exist.")
        print("Please run create_html_version.py first to generate the HTML file.")
        sys.exit(1)
    
    # Check if Pandoc is installed
    pandoc_path, is_installed = check_pandoc_installed()
    if not is_installed:
        print("Error: Pandoc is not installed or not found in common locations.")
        print("Please install Pandoc from https://pandoc.org/installing.html")
        print("If already installed, add it to your system PATH or provide its location.")
        sys.exit(1)
    
    print(f"Found Pandoc at: {pandoc_path}")
    
    # Convert HTML to DOCX
    success = convert_html_to_docx(html_path, docx_path, pandoc_path)
    
    if success:
        print("\nConversion complete!")
        print(f"Word document saved to: {docx_path}")
        print("\nNote: Some complex formulas may require manual adjustment in Word.")
        print("For best results with mathematical formulas, consider using the HTML version or converting to PDF instead.")
    else:
        print("\nConversion failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
