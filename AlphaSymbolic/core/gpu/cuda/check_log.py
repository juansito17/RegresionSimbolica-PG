
with open("compile_errors_no_ninja.log", "r", errors="ignore") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if ": error" in line.lower() or " error " in line.lower():
            print(f"Line {i}: {line.strip()}")
            # Print minimal context
            for j in range(max(0, i-1), min(len(lines), i+2)):
                if j != i: print(f"    {lines[j].strip()}")
            print("-" * 20)
