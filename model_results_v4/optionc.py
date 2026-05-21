import pickle
with open(r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\model_results_v4\enriched_df_cache.pkl", "rb") as f:
    obj = pickle.load(f)
print(type(obj))