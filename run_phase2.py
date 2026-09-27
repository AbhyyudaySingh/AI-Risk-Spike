import pandas as pd
import json
import numpy as np

from src.fraud_spike.features.registry import generate_feature_registry
from src.fraud_spike.features.pipeline import FeaturePipeline

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, set): return list(obj)
        return super(NpEncoder, self).default(obj)

def main():
    print("Running Phase 2 Pipeline...")
    results = {}
    
    # 1. Feature Registry
    registry = generate_feature_registry(
        "artifacts/manifests/schema_manifest.json", 
        "artifacts/manifests/feature_manifest.json"
    )
    results["features_excluded_count"] = len(registry.get("excluded", {}))
    results["features_included_count"] = len(registry.get("features", {}))
    
    # Example of excluded feature reason
    sample_excl_key = list(registry.get("excluded", {}).keys())[0]
    results["sample_excluded_feature"] = {sample_excl_key: registry["excluded"][sample_excl_key]}
    
    # 2. Load splits
    print("Loading splits...")
    df_train = pd.read_parquet("data/processed/train/train.parquet")
    df_val = pd.read_parquet("data/processed/validation/validation.parquet")
    df_test = pd.read_parquet("data/processed/test/test.parquet")
    
    # 3. Fit Pipeline
    pipeline = FeaturePipeline("artifacts/manifests/feature_manifest.json")
    pipeline.fit(df_train)
    
    # Record imputation stats to prove it only used train
    results["numerical_imputation_stats"] = pipeline.num_imputer.medians_
    
    # 4. Transform - to ensure history continuity across splits, we transform chronologically sequentially
    df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
    X_all = pipeline.transform(df_all)
    
    # Split back strictly by lengths to maintain boundaries
    len_train = len(df_train)
    len_val = len(df_val)
    
    X_train = X_all.iloc[:len_train].copy()
    X_val = X_all.iloc[len_train:len_train + len_val].copy()
    X_test = X_all.iloc[len_train + len_val:].copy()
    
    # 5. Save transformed
    X_train.to_parquet("data/processed/features/X_train.parquet", index=False)
    X_val.to_parquet("data/processed/features/X_val.parquet", index=False)
    X_test.to_parquet("data/processed/features/X_test.parquet", index=False)
    
    # Save pipeline artifacts
    pipeline.save("artifacts/preprocessing")
    
    # Record shapes
    results["train_transformed_shape"] = X_train.shape
    results["val_transformed_shape"] = X_val.shape
    results["test_transformed_shape"] = X_test.shape
    
    # Missingness indicators selected
    missing_indicators = [c for c in X_train.columns if c.endswith("_missing")]
    results["missingness_indicators_created"] = missing_indicators
    
    # Historical features created
    hist_features = [c for c in X_train.columns if c.startswith("history_")]
    results["past_only_history_features"] = hist_features
    
    # Example of historical feature
    sample_hist = X_train[['TransactionID', 'TransactionDT', 'card1', 'TransactionAmt', 'history_card1_prior_tx_count', 'history_card1_prior_amt_sum']].head(10).to_dict(orient='records')
    results["historical_feature_example"] = sample_hist
    
    with open("phase2_results.json", "w") as f:
        json.dump(results, f, indent=4, cls=NpEncoder)
        
    print("Phase 2 complete. Results saved to phase2_results.json")

if __name__ == "__main__":
    main()
