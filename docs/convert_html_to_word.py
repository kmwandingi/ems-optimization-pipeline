"""Convert HTML to Word document using python-docx.

This is a simplified conversion that won't handle math formulas as well as Pandoc.
For best results with math formulas, install Pandoc and use convert_to_word.py instead.
"""

import os
import re
from pathlib import Path
from bs4 import BeautifulSoup
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def extract_text_from_html(html_path):
    """Extract text from HTML file using BeautifulSoup."""
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract title
    title = "EMS Technical Report"
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.text
    
    # Extract content from body
    body = soup.body
    
    # Return structured content
    return {
        'title': title,
        'body': body
    }

def create_word_document(content, docx_path):
    """Create a Word document from the extracted content."""
    doc = Document()
    
    # Set document properties
    doc.core_properties.title = content['title']
    
    # Add title
    title = doc.add_heading(content['title'], level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Process body content
    body = content['body']
    
    # Warning about math formulas
    warning = doc.add_paragraph()
    warning_run = warning.add_run("NOTE: Mathematical formulas in this document require special handling. "
                                 "For accurate formula rendering, please refer to the HTML version or install Pandoc.")
    warning_run.italic = True
    doc.add_paragraph()
    
    # Simple extraction of text (this won't handle math formulas properly)
    if body:
        for element in body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'blockquote']):
            if element.name.startswith('h'):
                level = int(element.name[1])
                doc.add_heading(element.text, level=level)
            elif element.name == 'p':
                # Check if this paragraph contains math (simplistic check)
                text = element.text
                if '$' in text:
                    p = doc.add_paragraph()
                    # Add a note about math formula
                    p.add_run("[Math Formula: ").italic = True
                    # Extract the formula text
                    formula_text = text
                    # Clean up the formula text (remove $ signs)
                    formula_text = re.sub(r'\$\$(.*?)\$\$', r'\1', formula_text)
                    formula_text = re.sub(r'\$(.*?)\$', r'\1', formula_text)
                    p.add_run(formula_text)
                    p.add_run("]").italic = True
                else:
                    doc.add_paragraph(text)
            elif element.name == 'ul':
                for li in element.find_all('li'):
                    doc.add_paragraph(li.text, style='List Bullet')
            elif element.name == 'ol':
                for li in element.find_all('li'):
                    doc.add_paragraph(li.text, style='List Number')
            elif element.name == 'blockquote':
                quote = doc.add_paragraph(element.text)
                quote.style = 'Quote'
    
    # Save the document
    doc.save(docx_path)

def main():
    try:
        # Check if required packages are installed
        import bs4
        import docx
    except ImportError:
        print("Required packages not installed. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "beautifulsoup4", "python-docx"])
        print("Packages installed. Running conversion...")
    
    # Define file paths
    project_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    html_path = project_dir / "EMS_Technical_Report.html"
    docx_path = project_dir / "EMS_Technical_Report_simple.docx"
    
    # Check if HTML file exists
    if not html_path.exists():
        print(f"Error: HTML file {html_path} does not exist.")
        print("Please run create_html_version.py first to generate the HTML file.")
        return
    
    # Extract content from HTML
    print(f"Extracting content from {html_path}...")
    content = extract_text_from_html(html_path)
    
    # Create Word document
    print(f"Creating Word document {docx_path}...")
    create_word_document(content, docx_path)
    
    print(f"\nSimplified Word document created: {docx_path}")
    print("\nWARNING: This conversion doesn't properly handle mathematical formulas.")
    print("For better results with math formulas:")
    print("1. Install Pandoc from https://pandoc.org/installing.html")
    print("2. Run the convert_to_word.py script")
    print("Alternatively, use the HTML version or convert to PDF instead.")

if __name__ == "__main__":
    main()
