# hash_paragraphs.py
import hashlib, pathlib, textwrap
paras = []
current = []
for line in pathlib.Path("docs/EMS_Technical_Report.md").read_text(encoding='utf-8').splitlines() + [""]:
    if line.strip():
        current.append(line)
    elif current:
        paras.append("\n".join(current)); current=[]
print("### BASELINE HASHES")
for p in paras:
    h = hashlib.sha256(p.encode()).hexdigest()
    print(h)
