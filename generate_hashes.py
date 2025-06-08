#!/usr/bin/env python3
"""
Generate SHA-256 hashes for every paragraph in the EMS Technical Report.
"""

import hashlib
from pathlib import Path

def generate_paragraph_hashes(file_path):
    """Generate SHA-256 hashes for every paragraph in a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into paragraphs (double newline separated)
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    hashes = []
    for i, paragraph in enumerate(paragraphs, 1):
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(paragraph.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:12]  # First 12 characters for readability
        
        # Show first 50 chars of paragraph for identification
        preview = paragraph.replace('\n', ' ')[:50]
        if len(paragraph) > 50:
            preview += "..."
        
        hashes.append(f"{hash_hex}: {preview}")
    
    return hashes

if __name__ == "__main__":
    report_path = Path("docs/EMS_Technical_Report.md")
    if not report_path.exists():
        print(f"File not found: {report_path}")
        exit(1)
    
    hashes = generate_paragraph_hashes(report_path)
    
    print("### BASELINE HASHES (sha-256 of every paragraph in EMS_Technical_Report.md)")
    for hash_line in hashes:
        print(hash_line)
    
    print(f"\nTotal paragraphs: {len(hashes)}")