
import pandas as pd
import numpy as np

try:
    df = pd.read_csv("feynman_expanded_results.csv")
    
    print("\n--- GENERAL STATS ---")
    stats = df.groupby('Model').agg(
        Total=('ID', 'count'),
        Solved=('Status', lambda x: (x == '✅ SOLVED').sum()),
        Failed=('Status', lambda x: (x == '❌ FAILED').sum()),
        Avg_Time=('Time', 'mean')
    )
    stats['Success_Rate'] = (stats['Solved'] / stats['Total']) * 100
    print(stats)
    
    print("\n--- FAILURES ANALYSIS ---")
    failures = df[df['Status'] != '✅ SOLVED']
    if not failures.empty:
        print(failures[['Model', 'ID', 'Status', 'Time']].to_string())
    else:
        print("No failures found!")

    # Compare Times for Common Solved
    solved_lite = df[(df['Model'] == 'lite') & (df['Status'] == '✅ SOLVED')]
    solved_pro = df[(df['Model'] == 'pro') & (df['Status'] == '✅ SOLVED')]
    
    common_ids = set(solved_lite['ID']).intersection(set(solved_pro['ID']))
    print(f"\n--- HEAD-TO-HEAD (Common Solved: {len(common_ids)}) ---")
    
    lite_times = df[(df['Model'] == 'lite') & (df['ID'].isin(common_ids))].set_index('ID')['Time']
    pro_times = df[(df['Model'] == 'pro') & (df['ID'].isin(common_ids))].set_index('ID')['Time']
    
    diff = pro_times - lite_times
    print(f"Avg Time Saved by Lite per problem: {diff.mean():.2f}s")
    
except Exception as e:
    print(e)
