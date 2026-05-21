import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_pickle(r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\model_results_v4\enriched_df_cache.pkl")

non_features = ['credibility_class', 'dataset', 'source_file', 'url', 'source_name', 'bias_rating', 'factual_reporting']
y = df['credibility_class']
X = df.drop(columns=non_features, errors='ignore').select_dtypes(include=['number', 'bool']).fillna(0)

# Replicate the EXACT rule from Model_comparison_v3.py line 993
n_features_to_keep = min(len(X.columns), max(80, int(len(X.columns) * 0.6)))
print(f"Replicating pipeline: {len(X.columns)} -> {n_features_to_keep} features")

selector = SelectKBest(f_classif, k=n_features_to_keep)
selector.fit(X, y)

results = pd.DataFrame({
    'feature_name': X.columns,
    'f_score': selector.scores_,
    'p_value': selector.pvalues_,
    'selected': selector.get_support()
}).sort_values('f_score', ascending=False, na_position='last')

results.to_csv('feature_list_for_appendix.csv', index=False)
print(f"Wrote {len(results)} features ({results['selected'].sum()} selected) to feature_list_for_appendix.csv")