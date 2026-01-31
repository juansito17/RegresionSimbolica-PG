
with open("build_phase4.log", "r", errors="ignore") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "unresolved external symbol" in line.lower():
            print(f"Line {i}: {line.strip()}")
            # The symbol is usually in the same line or next line depending on format
            # LNK2001: unresolved external symbol "void __cdecl foo(...)" 

