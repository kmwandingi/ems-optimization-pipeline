"""Create an HTML version of the Markdown file with MathJax for formula rendering"""

import os
import sys
import re
import markdown
import tempfile
import webbrowser
from pathlib import Path


def process_math_formulas(content):
    """Process LaTeX math formulas for correct rendering with MathJax."""
    # Extract and temporarily replace math formulas
    inline_math_blocks = []
    display_math_blocks = []
    
    # Pattern for inline math: $...$
    def replace_inline_math(match):
        formula = match.group(1)
        inline_math_blocks.append(formula)
        return f"INLINE_MATH_PLACEHOLDER_{len(inline_math_blocks)-1}_"
    
    # Pattern for display math: $$...$$
    def replace_display_math(match):
        formula = match.group(1)
        display_math_blocks.append(formula)
        return f"DISPLAY_MATH_PLACEHOLDER_{len(display_math_blocks)-1}_"
    
    # Replace math blocks with placeholders
    content = re.sub(r'\$\$([\s\S]*?)\$\$', replace_display_math, content)
    content = re.sub(r'\$([^\$\n]+?)\$', replace_inline_math, content)
    
    # Clean up formulas
    def fix_math_formula(formula):
        """Fix only the specific problematic patterns identified in the images."""
        
        # TARGETED FIXES FOR THE SPECIFIC BROKEN FORMULAS:
        
        # 1. Fix spaces inside superscript braces (the main culprit)
        formula = re.sub(r'\^\{\s*\+\s*\}', r'^{+}', formula)
        formula = re.sub(r'\^\{\s*-\s*\}', r'^{-}', formula)
        
        # 2. Fix missing subscript markers (like g^{ + }t should be g^{+}_t)
        formula = re.sub(r'g\^\{\s*\+\s*\}t\b', r'g^{+}_t', formula)
        formula = re.sub(r'g\^\{\s*-\s*\}t\b', r'g^{-}_t', formula)
        formula = re.sub(r'b\^\{\s*\+\s*\}t\b', r'b^{+}_t', formula)
        formula = re.sub(r'b\^\{\s*-\s*\}t\b', r'b^{-}_t', formula)
        
        # CRITICAL: Convert all superscript-before-subscript to subscript-before-superscript format
        # This fixes the b^+_t → b_t^+ pattern which is critical for MathJax/KaTeX rendering
        
        # Simple cases: b^+_t → b_t^+
        formula = re.sub(r'([a-zA-Z0-9])\^(\+|-)_([a-zA-Z0-9]+)', r'\1_\3^\2', formula)
        
        # Cases with braces around the superscript: b^{+}_t → b_t^{+}
        formula = re.sub(r'([a-zA-Z0-9])\^\{(\+|-)\}_([a-zA-Z0-9]+)', r'\1_\3^{\2}', formula)
        
        # Cases with general superscripts: b^{something}_t → b_t^{something}
        formula = re.sub(r'([a-zA-Z0-9])\^\{([^}]+)\}_([a-zA-Z0-9]+)', r'\1_\3^{\2}', formula)
        
        # Special case for subscripts with braces: b^+_{t-1} → b_{t-1}^+
        formula = re.sub(r'([a-zA-Z0-9])\^(\+|-)_\{([^}]+)\}', r'\1_{\3}^\2', formula)
        formula = re.sub(r'([a-zA-Z0-9])\^\{(\+|-)\}_\{([^}]+)\}', r'\1_{\3}^{\2}', formula)
        
        # 3. Fix malformed max/min subscripts
        formula = re.sub(r'G\\max\b', r'G_{\\max}', formula)
        formula = re.sub(r'G\{\\max\}', r'G_{\\max}', formula)
        formula = re.sub(r'G\{\\max\}\^\{\s*-\s*\}', r'G_{\\max}^{-}', formula)
        
        # 4. Fix sum notation with missing underscores
        formula = re.sub(r'\\sum\{([^}]+)\}', r'\\sum_{\1}', formula)
        
        # Keep all the original working fixes from your code
        spacing_commands = [
            'forall', 'exists', 'in', 'subset', 'subseteq', 'cup', 'cap',
            'leq', 'geq', 'neq', 'approx', 'equiv', 'cong', 'sim',
            'quad', 'qquad', 'sum', 'prod', 'int', 'max', 'min',
            'cdot', 'times', 'div', 'pm', 'mp', 'ldots', 'Delta', 'text'
        ]
        
        for cmd in spacing_commands:
            pattern = f"\\\\{cmd}(?![\\s{{}}^_])"
            replacement = f"\\\\{cmd} "
            formula = re.sub(pattern, replacement, formula)
        
        # Updated replacements with subscript-before-superscript pattern
        replacements = {
            '$$s_t = s_{t-1} + \eta^+ \cdot b^+_t - \frac{1}{\eta^-} \cdot b^-_t': '$$s_t = s_{t-1} + \eta^{+} \cdot b_t^{+} - \frac{1}{\eta^{-}} \cdot b_t^{-}',
            '$$|b^+_t - b^+_{t-1}| \leq R^{\max}': '$$|b_t^{+} - b_{t-1}^{+}| \leq R^{\max}',
            '$$\sum_{t=1}^{T} y_{d,t} \cdot P_d \cdot \Delta t = E^{\text{req}}_d': '$$\sum_{t=1}^{T} y_{d,t} \cdot P_d \cdot \Delta t = E_d^{\text{req}}',
            '$$y_{d,t} \cdot P_d \leq P^{\max}_d': '$$y_{d,t} \cdot P_d \leq P_d^{\max}',
            '$$g^+_t - g^-_t = \sum{d \in \mathcal{D}} \sum_{h=-H_d}^{H_d} x_{d,t-h,h} \cdot c_{d,t-h} + b^+_t - b^-_t - PV_t': 
            '$$g_t^{+} - g_t^{-} = \sum_{d \in \mathcal{D}} \sum_{h=-H_d}^{H_d} x_{d,t-h,h} \cdot c_{d,t-h} + b_t^{+} - b_t^{-} - PV_t',
            '$$\sum_{h=-H_d}^{H_d} x_{d,t,h} \cdot c_{d,t} \leq M \cdot z_{d,t}': 
            '$$\sum_{h=-H_d}^{H_d} x_{d,t,h} \cdot c_{d,t} \leq M \cdot z_{d,t}',
            '$$\min \sum_{t=1}^{T} \left( p_t \cdot g^+_t - p^- \cdot g^-_t + p^{\text{degradation}} \cdot (b^+_t + b^-_t) \right)': 
            '$$\min \sum_{t=1}^{T} \left( p_t \cdot g_t^{+} - p^{-} \cdot g_t^{-} + p^{\text{degradation}} \cdot (b_t^{+} + b_t^{-}) \right)',
            '$$\min \sum_{t=0}^{T-1} \left[ p_t \cdot \left( \sum_{d \in D} c_{d,t} \cdot x_{d,t} + b^+_t - b^-_t - s_t \right) + p^{\text{degradation}} \cdot (b^+_t + b^-_t) + \sum_{d \in D} w_{\text{prob},d} \cdot (1 - P_{d,t} \cdot x_{d,t}) \right]$$':
            '$$\min \sum_{t=0}^{T-1} \left[ p_t \cdot \left( \sum_{d \in D} c_{d,t} \cdot x_{d,t} + b_t^{+} - b_t^{-} - s_t \right) + p^{\text{degradation}} \cdot (b_t^{+} + b_t^{-}) + \sum_{d \in D} w_{\text{prob},d} \cdot (1 - P_{d,t} \cdot x_{d,t}) \right]$$'
        }
        
        for old, new in replacements.items():
            if old in formula:
                formula = formula.replace(old, new)
        
        return formula
    
    # Final comprehensive fixes for all formulas
    def final_pattern_fixes(formula):
        """Apply final pattern fixes to ensure subscript-before-superscript pattern across all formulas"""
        # The most critical pattern: b^+_t → b_t^+ (superscript-before-subscript to subscript-before-superscript)
        # This covers the main issue in the report
        
        # Handle common battery/grid variables: b^+_t, g^+_t, etc.
        for var in ['b', 'g', 'e', 'η', 'p']:
            # Find any remaining instances of var^+_t or var^-_t and fix them
            formula = re.sub(fr'{var}\^\{{\+\}}_([a-zA-Z0-9]+)', fr'{var}_\1^{{+}}', formula)
            formula = re.sub(fr'{var}\^\{{-\}}_([a-zA-Z0-9]+)', fr'{var}_\1^{{-}}', formula)
            formula = re.sub(fr'{var}\^\+_([a-zA-Z0-9]+)', fr'{var}_\1^{{+}}', formula)
            formula = re.sub(fr'{var}\^-_([a-zA-Z0-9]+)', fr'{var}_\1^{{-}}', formula)
        
        # Fix variables with 'max' or other text in superscript followed by subscripts
        formula = re.sub(r'([A-Za-z])\^\{([^}]+)\}_([a-zA-Z0-9]+)', r'\1_\3^{\2}', formula)
        
        # Fix any other variable with superscript first, subscript second pattern
        formula = re.sub(r'([A-Za-z0-9])\^([^{}_\s])_([a-zA-Z0-9]+)', r'\1_\3^{\2}', formula)
        
        # Fix E^{text{req}}_d pattern to E_d^{text{req}}
        formula = re.sub(r'([A-Za-z])\^\{\\text\{([^}]+)\}\}_([a-zA-Z0-9]+)', r'\1_\3^{\\text{\2}}', formula)
        
        return formula
        
    # Fix and restore math blocks
    cleaned_inline = [final_pattern_fixes(fix_math_formula(f)) for f in inline_math_blocks]
    cleaned_display = [final_pattern_fixes(fix_math_formula(f)) for f in display_math_blocks]
    
    # Restore the formulas with their fixed versions
    for i, formula in enumerate(cleaned_inline):
        content = content.replace(f"INLINE_MATH_PLACEHOLDER_{i}_", f"${formula}$")
    
    for i, formula in enumerate(cleaned_display):
        content = content.replace(f"DISPLAY_MATH_PLACEHOLDER_{i}_", f"$${formula}$$")
    
    # Fix "Subject to:" and "Where:" sections that are outside math mode
    content = re.sub(
        r'Subject to:\s*\n((?:\s*-[^\n]+\n)*)', 
        lambda m: '**Subject to:**\n\n' + m.group(1), 
        content
    )
    content = re.sub(
        r'Where:\s*\n((?:\s*-[^\n]+\n)*)', 
        lambda m: '**Where:**\n\n' + m.group(1), 
        content
    )
    
    return content

def create_html_with_mathjax(md_path, html_path):
    """Convert Markdown to HTML with MathJax support."""
    # Read markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
        
    # Clean up math formulas for proper rendering
    md_content = process_math_formulas(md_content)
    
    # Convert markdown to HTML
    extensions = [
        'tables',       # For tables
        'fenced_code',  # For code blocks
        'codehilite',   # For syntax highlighting
    ]
    
    html_body = markdown.markdown(md_content, extensions=extensions)
    
    # Complete HTML document with MathJax
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EMS Technical Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            padding: 20px;
            margin: 0 auto;
            max-width: 900px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #333;
            margin-top: 24px;
            margin-bottom: 16px;
        }}
        h1 {{ font-size: 28px; }}
        h2 {{ font-size: 24px; border-bottom: 1px solid #eaecef; padding-bottom: 8px; }}
        h3 {{ font-size: 20px; }}
        h4 {{ font-size: 18px; }}
        h5, h6 {{ font-size: 16px; }}
        code {{
            background-color: #f6f8fa;
            padding: 3px 6px;
            border-radius: 3px;
            font-family: Consolas, monospace;
        }}
        pre {{
            background-color: #f6f8fa;
            padding: 16px;
            overflow: auto;
            border-radius: 3px;
        }}
        blockquote {{
            border-left: 4px solid #ddd;
            padding-left: 16px;
            color: #666;
            margin-left: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        table, th, td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        th {{
            background-color: #f6f8fa;
            text-align: left;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        @media print {{
            @page {{
                margin: 1cm;
                size: A4;
            }}
            body {{
                font-size: 12pt;
            }}
            a {{
                text-decoration: none;
                color: black;
            }}
            a[href]::after {{
                content: " (" attr(href) ")";
                font-size: 90%;
                color: #555;
            }}
            pre, code {{
                background-color: #f6f8fa !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
        .print-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }}
        .print-button:hover {{
            background-color: #45a049;
        }}
        @media print {{
            .print-button {{
                display: none;
            }}
        }}
    </style>
    <!-- MathJax for formula rendering -->
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true,
                packages: ['base', 'ams', 'noerrors', 'noundefined', 'color']
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                ignoreHtmlClass: 'tex2jax_ignore',
                processHtmlClass: 'tex2jax_process'
            }},
            svg: {{
                fontCache: 'global',
                scale: 1,
                minScale: .5,
                mtextInheritFont: false,
                merrorInheritFont: true,
                mathmlSpacing: false,
                skipAttributes: {{}},
                exFactor: .5,
                displayAlign: 'center',
                displayIndent: '0'
            }}
        }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    <script>
        window.addEventListener('load', function() {{
            MathJax.typeset();
        }});
    </script>
</head>
<body>
    <button class="print-button" onclick="window.print()">Print to PDF</button>
    <h1>EMS Technical Report</h1>
    {html_body}
</body>
</html>
"""
    
    # Write HTML to file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    return html_path

def main():
    # Default file paths
    md_path = "EMS_Technical_Report.md"
    html_path = "EMS_Technical_Report.html"
    
    # Allow command-line arguments to override defaults
    if len(sys.argv) > 1:
        md_path = sys.argv[1]
    if len(sys.argv) > 2:
        html_path = sys.argv[2]
    
    # Check if markdown file exists
    if not os.path.exists(md_path):
        print(f"Error: Markdown file not found: {md_path}")
        return
    
    print(f"Converting {md_path} to HTML with MathJax...")
    
    # Convert markdown to HTML
    create_html_with_mathjax(md_path, html_path)
    
    # Get full path for HTML file
    html_full_path = os.path.abspath(html_path)
    
    # Open HTML file in the default browser
    print(f"Opening {html_path} in your browser...")
    webbrowser.open('file://' + html_full_path)
    
    print("\nFixed the specific broken mathematical formulas:")
    print("✓ Removed spaces from superscript braces: b^{ + } → b^{+}")
    print("✓ Fixed missing subscript markers: g^{+}t → g^{+}_t") 
    print("✓ Fixed malformed max subscripts: G\\max → G_{\\max}")
    print("✓ Fixed sum notation: \\sum{...} → \\sum_{...}")
    print("✓ Improved Subject to: and Where: section formatting")
    
    print("\nInstructions to create PDF:")
    print("1. Click the green 'Print to PDF' button or use Ctrl+P")
    print("2. Select 'Save as PDF' as the destination")
    print("3. Click 'Save' and choose a location for your PDF file")
    print(f"\nHTML file saved to: {html_full_path}")

if __name__ == "__main__":
    main()