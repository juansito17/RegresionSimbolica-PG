
import pandas as pd
import os
import shutil

ROOT_CSV = "learned_formulas.csv"
TARGET_CSV = os.path.join("results", "learned_formulas.csv")

def merge_csvs():
    print("--- Merging CSVs ---")
    
    # 1. Load Root CSV
    if os.path.exists(ROOT_CSV):
        try:
            df_root = pd.read_csv(ROOT_CSV)
            print(f"Loaded Root CSV: {len(df_root)} rows")
        except Exception as e:
            print(f"Error loading Root CSV: {e}")
            df_root = pd.DataFrame()
    else:
        print("Root CSV not found.")
        df_root = pd.DataFrame()

    # 2. Load Target CSV
    if os.path.exists(TARGET_CSV):
        try:
            df_target = pd.read_csv(TARGET_CSV)
            print(f"Loaded Target CSV: {len(df_target)} rows")
        except Exception as e:
            print(f"Error loading Target CSV: {e}")
            df_target = pd.DataFrame()
    else:
        print("Target CSV not found. Creating new...")
        df_target = pd.DataFrame()
        
    # 3. Concatenate
    if df_root.empty and df_target.empty:
        print("Both empty. Nothing to do.")
        return

    df_combined = pd.concat([df_target, df_root], ignore_index=True)
    
    # 4. Drop Duplicates (based on formula)
    # Ensure 'formula' column exists
    if 'formula' in df_combined.columns:
        before = len(df_combined)
        df_combined.drop_duplicates(subset=['formula'], keep='last', inplace=True)
        after = len(df_combined)
        print(f"Dropped {before - after} duplicates. Total unique: {after}")
    else:
        print("Warning: 'formula' column missing. Skipping duplicate check.")

    # 5. Save to Target
    os.makedirs("results", exist_ok=True)
    df_combined.to_csv(TARGET_CSV, index=False)
    print(f"Saved to {TARGET_CSV}")
    
    # 6. Delete Root
    if os.path.exists(ROOT_CSV):
        os.remove(ROOT_CSV)
        print(f"Deleted {ROOT_CSV}")

if __name__ == "__main__":
    merge_csvs()
