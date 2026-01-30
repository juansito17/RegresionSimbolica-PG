
try:
    with open('julio.log', 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            lower_line = line.lower()
            if 'error' in lower_line or ' c2' in lower_line or 'fatal' in lower_line:
                print(line.strip())
            # Also print near the end to capture linker errors
            # implementation detail: just print specific keywords
except Exception as e:
    print(f"Failed to read log: {e}")
