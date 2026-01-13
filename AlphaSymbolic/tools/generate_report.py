import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.grammar import ExpressionTree

# --- DATA ---
X_FULL = np.arange(1, 28) # 1 to 27
Y_REAL = np.array([
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 
    14772512, 95815104, 666090624, 4968057848, 39029188884, 314666222712, 
    2691008701644, 2423393768440, 227514171973736, 2207893435808352,
    22317699616364044, 234907967154122528
], dtype=np.float64)

INPUT_CSV = "top_formulas.csv"
OUTPUT_CSV = "top_5_detailed_report.csv"

def evaluate_formula(formula_str, x_vals):
    try:
        tree = ExpressionTree.from_infix(formula_str)
        if not tree.is_valid:
            print(f"Invalid Formula: {formula_str}")
            return np.zeros_like(x_vals)
        return tree.evaluate(x_vals) # Should handle array input
    except Exception as e:
        print(f"Error evaluating {formula_str}: {e}")
        return np.zeros_like(x_vals)

def main():
    print(f"Generating report from {INPUT_CSV}...")
    
    if not os.path.exists(INPUT_CSV):
        print("Input file not found.")
        return

    # Load Top 5
    df_in = pd.read_csv(INPUT_CSV)
    top_5 = df_in.head(5).copy()
    
    # Prepare List of Rows for new DataFrame
    report_rows = []
    
    for idx, row in top_5.iterrows():
        formula = row['formula']
        print(f"Processing #{idx+1}: {formula[:30]}...")
        
        y_pred = evaluate_formula(formula, X_FULL)
        
        # Build Row Dictionary
        new_row = {'Rank': idx + 1, 'Formula': formula}
        
        total_mape = 0
        count = 0
        
        for i, x in enumerate(X_FULL):
            real = Y_REAL[i]
            pred = y_pred[i]
            
            # Handle division by zero if real is 0 (indexes 1 and 2 are 0)
            if real == 0:
                # If real is 0, error is absolute diff? Or undefined %?
                # Usually we define error strictly. If pred is 0, error is 0%.
                # Let's show abs diff for 0, or skip % calc.
                # User asked for "error porcentual".
                # If Real=0, Pred=0 -> 0%
                # If Real=0, Pred=0.1 -> Infinite %.
                # Let's put NaN or a placeholder for 0 values, or just show raw delta.
                # Indices 1, 2 (i=1, i=2 since 0-indexed) are 0.
                if abs(pred) < 1e-9:
                    err_pct = 0.0
                else:
                    err_pct = np.nan # Undefined
            else:
                err_pct = abs((pred - real) / real) * 100
                total_mape += err_pct
                count += 1
            
            # Add columns
            # new_row[f'X_{x}_Real'] = real
            new_row[f'X_{x}_Pred'] = pred
            new_row[f'X_{x}_Err%'] = err_pct
        
        new_row['Mean_Err%'] = total_mape / count if count > 0 else 0
        report_rows.append(new_row)
        
    # Create DF
    # To keep it organized, let's order columns: Rank, Formula, Mean_Err%, then X_1_Pred, X_1_Err%, etc.
    df_out = pd.DataFrame(report_rows)
    
    # Calculate Mean Error Percentage across all points (excluding 0s)
    
    cols = ['Rank', 'Formula', 'Mean_Err%']
    for x in X_FULL:
        cols.append(f'X_{x}_Pred')
        cols.append(f'X_{x}_Err%')
        
    df_out = df_out[cols]
    
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Report saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
