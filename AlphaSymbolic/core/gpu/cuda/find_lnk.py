
with open("build_phase4.log", "rb") as f:
    content = f.read().decode("utf-16", errors="ignore")
    
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'LNK2001' in line.upper() or 'LNK2019' in line.upper():
        print(f"=== LNK Line {i} ===")
        print(line.strip())
        print("---")
