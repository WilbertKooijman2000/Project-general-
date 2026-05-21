import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_pickle(r"C:\Users\Wilbe\OneDrive\Desktop\profiling-data-Copy(1)\thesis_output\model_results_v4\enriched_df_cache.pkl")

# Drop non-feature columns
non_features = ['credibility_class', 'dataset', 'source_file', 'url', 'source_name', 'bias_rating', 'factual_reporting']
y = df['credibility_class']
X = df.drop(columns=non_features, errors='ignore')

# Some columns might still be non-numeric — check
print(f"X shape after dropping metadata: {X.shape}")
print(f"\nNon-numeric columns remaining:")
non_numeric = X.select_dtypes(exclude=['number', 'bool']).columns.tolist()
print(non_numeric)

# Drop those too if they exist (likely strings we need to ignore or one-hot)
if non_numeric:
    print(f"\nSample values from non-numeric columns:")
    for c in non_numeric[:5]:
        print(f"  {c}: {X[c].dropna().unique()[:5]}")
    X = X.select_dtypes(include=['number', 'bool'])
    print(f"\nX shape after keeping only numeric: {X.shape}")

# Show class distribution
print(f"\nClass distribution:\n{y.value_counts()}")

# Now try SelectKBest at the k that matches your thesis
selector = SelectKBest(score_func=f_classif, k='all')  # 'all' = compute scores for everything
selector.fit(X.fillna(0), y)

results = pd.DataFrame({
    'feature_name': X.columns,
    'f_score': selector.scores_,
    'p_value': selector.pvalues_,
}).sort_values('f_score', ascending=False)

# How many pass p < 0.05?
n_significant = (results['p_value'] < 0.05).sum()
print(f"\nFeatures with p < 0.05: {n_significant} of {len(results)}")
print(f"\nTop 10 by F-score:")
print(results.head(10).to_string(index=False))
print(f"\nBottom 5 by F-score (least discriminative):")
print(results.tail(5).to_string(index=False))

# Save the full ranked list
results['selected'] = results['p_value'] < 0.05
results.to_csv('feature_list_for_appendix.csv', index=False)
print(f"\n✓ Wrote {len(results)} features to feature_list_for_appendix.csv")