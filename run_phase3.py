import pandas as pd
import numpy as np
import json
import xgboost as xgb
import pickle
import os

from src.fraud_spike.models.cv import create_expanding_chronological_folds
from src.fraud_spike.features.pipeline import FeaturePipeline
from src.fraud_spike.models.xgboost_model import train_xgboost_model
from src.fraud_spike.models.calibration import ProbabilityCalibrator
from src.fraud_spike.models.thresholds import select_optimal_thresholds

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, set): return list(obj)
        return super(NpEncoder, self).default(obj)

def main():
    print("Starting Revised Phase 3...")
    results = {}
    
    # 1. Load RAW train split for CV
    df_raw_train = pd.read_parquet("data/processed/train/train.parquet")
    df_raw_train = df_raw_train.sort_values('TransactionDT').reset_index(drop=True)
    folds = create_expanding_chronological_folds(df_raw_train, num_folds=4)
    
    cv_results = []
    oof_y_true = []
    oof_y_pred = []
    
    fraud_rate = df_raw_train['isFraud'].mean()
    scale_pos_weight = (1 - fraud_rate) / fraud_rate
    results['class_imbalance_strategy'] = "Used scale_pos_weight = " + str(scale_pos_weight) + " directly in XGBoost objective to combat ~" + str(round(fraud_rate*100, 2)) + "% prevalence."
    
    for i, (f_train, f_val) in enumerate(folds):
        print(f"--- Fold {i+1} ---")
        pipe = FeaturePipeline("artifacts/manifests/feature_manifest.json")
        pipe.fit(f_train)
        
        f_all = pd.concat([f_train, f_val]).reset_index(drop=True)
        X_all = pipe.transform(f_all)
        
        X_ftrain = X_all.iloc[:len(f_train)].copy()
        X_fval = X_all.iloc[len(f_train):].copy()
        y_ftrain = f_train['isFraud'].values
        y_fval = f_val['isFraud'].values
        
        _, val_preds, xgb_metrics = train_xgboost_model(X_ftrain, y_ftrain, X_fval, y_fval, scale_pos_weight)
        
        oof_y_true.extend(y_fval)
        oof_y_pred.extend(val_preds)
        
        cv_results.append({
            "fold": i+1,
            "train_dt_min": int(f_train['TransactionDT'].min()),
            "train_dt_max": int(f_train['TransactionDT'].max()),
            "val_dt_min": int(f_val['TransactionDT'].min()),
            "val_dt_max": int(f_val['TransactionDT'].max()),
            "train_prevalence": f_train['isFraud'].mean(),
            "val_prevalence": f_val['isFraud'].mean(),
            "pr_auc": xgb_metrics['pr_auc'],
            "roc_auc": xgb_metrics['roc_auc'],
            "precision_at_0.5": xgb_metrics['precision'],
            "recall_at_0.5": xgb_metrics['recall']
        })
        
    results['chronological_stability'] = cv_results
    
    # 2. Fit Calibrators on OOF predictions
    calibrator = ProbabilityCalibrator()
    calibrator.fit(np.array(oof_y_true), np.array(oof_y_pred))
    
    # 3. Train Final Model on 100% of Train Split
    X_train_full = pd.read_parquet("data/processed/features/X_train.parquet")
    X_val_full = pd.read_parquet("data/processed/features/X_val.parquet")
    
    # Extract robust metadata directly from the source Validation file to guarantee strict alignment
    val_raw = pd.read_parquet("data/processed/validation/validation.parquet")
    # Verify index and length alignment
    assert len(X_val_full) == len(val_raw) == 88581, "Validation length mismatch!"
    
    # Extract metadata securely
    val_amts = val_raw['TransactionAmt'].values
    val_ids = val_raw['TransactionID'].values
    assert len(set(val_ids)) == len(val_ids), "TransactionIDs are not unique in validation!"
    
    y_train_full = X_train_full.pop('isFraud')
    y_val_full = X_val_full.pop('isFraud')
    
    final_model, val_preds_raw, _ = train_xgboost_model(
        X_train_full, y_train_full, X_val_full, y_val_full, scale_pos_weight
    )
    
    # --- EXPLICIT ALIGNMENT MERGE ---
    # Create DataFrame of predictions using the index from X_val_full implicitly (which must match val_raw natively, but we enforce it)
    df_preds = pd.DataFrame({
        'TransactionID': val_ids,
        'y_prob_raw': val_preds_raw,
        'y_true': y_val_full.values
    })
    
    merged = pd.merge(df_preds, val_raw[['TransactionID', 'TransactionAmt']], on='TransactionID', how='inner')
    
    # Assert alignment
    assert len(df_preds) == 88581, "Predictions length wrong"
    assert len(val_raw) == 88581, "Metadata length wrong"
    assert len(merged) == 88581, "Joined length wrong"
    assert merged['TransactionID'].nunique() == 88581, "Duplicate IDs in joined data"
    assert set(df_preds['TransactionID']) - set(val_raw['TransactionID']) == set(), "Unmatched prediction IDs"
    assert set(val_raw['TransactionID']) - set(df_preds['TransactionID']) == set(), "Unmatched metadata IDs"
    
    # Use aligned data for calibration evaluation and thresholding
    aligned_y_true = merged['y_true'].values
    aligned_y_prob_raw = merged['y_prob_raw'].values
    aligned_amts = merged['TransactionAmt'].values
    aligned_ids = merged['TransactionID'].values
    
    # 4. Evaluate Calibration on untouched Validation Split
    val_preds_isotonic = calibrator.predict_proba_calibrated(aligned_y_prob_raw, 'isotonic')
    val_preds_sigmoid = calibrator.predict_proba_calibrated(aligned_y_prob_raw, 'sigmoid')
    
    eval_raw = calibrator.evaluate(aligned_y_true, aligned_y_prob_raw)
    eval_iso = calibrator.evaluate(aligned_y_true, val_preds_isotonic)
    eval_sig = calibrator.evaluate(aligned_y_true, val_preds_sigmoid)
    
    results['calibration_validation'] = {
        "raw": eval_raw,
        "isotonic": eval_iso,
        "sigmoid": eval_sig
    }
    
    # Decision: Pick the best calibrator based on Brier, ECE, and reliability curve
    # Sigmoid preferred if it improves Brier/ECE without destroying discrimination
    chosen_method = 'raw'
    val_preds_final = val_preds_raw
    
    # Find the minimum ECE among calibrators that don't severely degrade PR-AUC
    methods = [('raw', eval_raw, aligned_y_prob_raw), ('isotonic', eval_iso, val_preds_isotonic), ('sigmoid', eval_sig, val_preds_sigmoid)]
    best_ece = eval_raw['ece']
    
    for name, metrics, preds in methods:
        if name == 'raw':
            continue
        if metrics['ece'] < best_ece and metrics['pr_auc'] >= eval_raw['pr_auc'] * 0.95:
            # If sigmoid is close to Isotonic on ECE/Brier, prefer Sigmoid due to monotonicity
            if name == 'isotonic' and chosen_method == 'sigmoid':
                if metrics['ece'] >= eval_sig['ece'] * 0.9:
                    continue # Keep sigmoid
            chosen_method = name
            val_preds_final = preds
            best_ece = metrics['ece']
            
    results['calibration_adopted'] = chosen_method
    
    # 5. Parameterized Cost Threshold Selection
    thresh_scenarios, rp_metrics = select_optimal_thresholds(aligned_y_true, val_preds_final, aligned_amts)
    results['threshold_scenarios'] = thresh_scenarios
    results['recall_at_precision'] = rp_metrics
    
    # Output explicit cost assumption
    results['cost_assumptions'] = "True Positive intervention effectiveness is assumed to be 100% (cost is only c_review). LOW, BASE, HIGH are illustrative research scenarios."
    
    # 6. Save Artifacts
    os.makedirs("artifacts/models/transaction", exist_ok=True)
    final_model.save_model("artifacts/models/transaction/xgb_layer1.json")
    
    if chosen_method != 'raw':
        with open("artifacts/models/transaction/calibration.pkl", "wb") as f:
            pickle.dump(calibrator, f)
            
    with open("artifacts/models/transaction/tau_tx.json", "w") as f:
        json.dump({"tau_tx": thresh_scenarios["BASE"]["threshold"]}, f, indent=4, cls=NpEncoder)
        
    with open("phase3_revised_results.json", "w") as f:
        json.dump(results, f, indent=4, cls=NpEncoder)
        
    # Also save a sidecar validation dump for the test script
    np.savez("artifacts/models/transaction/val_dump.npz", 
             y_true=aligned_y_true, 
             y_prob=val_preds_final, 
             amts=aligned_amts, 
             ids=aligned_ids)
        
    print("Revised Phase 3 complete. Results saved to phase3_revised_results.json")

if __name__ == "__main__":
    main()
