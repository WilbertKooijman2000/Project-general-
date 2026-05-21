import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Load the cached enriched DataFrame
df = pd.read_pickle(r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\model_results_v4\enriched_df_cache.pkl")

# Show what we're working with
print(f"Shape: {df.shape}")
print(f"Columns (first 30): {list(df.columns[:30])}")
print(f"Columns (last 10): {list(df.columns[-10:])}")
print(f"\nDtypes summary:\n{df.dtypes.value_counts()}")

# Try to auto-detect the label column
label_candidates = [c for c in df.columns if any(k in c.lower() 
    for k in ['credibility', 'label', 'class', 'target', 'y_'])]
print(f"\nPossible label columns: {label_candidates}")