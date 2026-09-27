import os
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, precision_score, recall_score, confusion_matrix, brier_score_loss
import joblib

from src.fraud_spike.streaming.simulate import FastReplayEngine
from src.fraud_spike.spike.aggregate import build_tumbling_windows
from src.fraud_spike.spike.ground_truth import compute_q10_eligible_volume, create_spike_events
from src.fraud_spike.evaluation.spike_metrics import evaluate_spike_detection

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def verify_upstream_artifacts():
    print("Verifying frozen Phase 1-5 upstream artifacts...")
    artifacts = [
        'artifacts/models/transaction/xgb_layer1.json',
        'artifacts/models/transaction/calibration.pkl',
        'artifacts/models/transaction/tau_tx.json',
        'artifacts/manifests/spike_manifest.json'
    ]
    hashes = {}
    for a in artifacts:
        if os.path.exists(a):
            hashes[a] = compute_sha256(a)
            print(f"Verified {a}: {hashes[a]}")
        else:
            raise FileNotFoundError(f"Missing required artifact: {a}")
    return hashes

def evaluate_layer1(y_true, y_prob, tau_tx):
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    roc_auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    y_pred = (y_prob >= tau_tx).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    alert_volume = fp + tp
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'alert_volume': int(alert_volume),
        'brier_score': brier,
        'mean_predicted_prob': float(np.mean(y_prob)),
        'observed_prevalence': float(np.mean(y_true))
    }

def main():
    hashes_before = verify_upstream_artifacts()
    
    # Load Test Data
    test_df = pd.read_parquet('data/processed/test/test.parquet')
    
    # We must predict on test_df
    from src.fraud_spike.features.pipeline import FeaturePipeline
    print("Running Layer 1 Inference...")
    pipeline = FeaturePipeline('artifacts/manifests/feature_manifest.json').load('artifacts/preprocessing')
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model('artifacts/models/transaction/xgb_layer1.json')
    calibrator = joblib.load('artifacts/models/transaction/calibration.pkl')
    
    X_test = test_df.drop(columns=['isFraud'])
    y_test = test_df['isFraud'].values
    
    X_processed = pipeline.transform(X_test)
    X_for_model = X_processed.drop(columns=['TransactionID', 'TransactionDT'])
    raw_probs = model.predict_proba(X_for_model)[:, 1]
    calib_probs = calibrator.predict_proba_calibrated(raw_probs, method='sigmoid')
    
    # Load tau_tx
    with open('artifacts/models/transaction/tau_tx.json', 'r') as f:
        tau_tx = json.load(f)['tau_tx']
        
    print(f"Frozen tau_tx: {tau_tx}")
    
    # Save probabilities to npz for FastReplayEngine
    np.savez_compressed('artifacts/models/transaction/test_dump.npz', 
                        ids=test_df['TransactionID'].values,
                        y_prob=calib_probs,
                        amts=test_df['TransactionAmt'].values)
    
    l1_metrics = evaluate_layer1(y_test, calib_probs, tau_tx)
    print("\n--- Layer 1 Held-Out Test Metrics ---")
    for k, v in l1_metrics.items():
        print(f"{k}: {v}")
        
    # Proxy Events on Test using frozen 0.1950 threshold
    w_size = 3600
    q10_vol = 16
    p975 = 0.1950 # Frozen
    
    test_preds = pd.DataFrame({
        'TransactionID': test_df['TransactionID'].values,
        'TransactionDT': test_df['TransactionDT'].values,
        'TransactionAmt': test_df['TransactionAmt'].values,
        'card1': test_df['card1'].values,
        'fraud_probability': calib_probs,
        'isFraud': y_test
    })
    
    events_df = create_spike_events(test_preds, w_size, q10_vol, p975)
    print(f"\nGround-Truth Spike Events on Test: {len(events_df)}")
    
    # Layer 2 Streaming Replay
    print("Running Layer 2 Streaming Engine...")
    class TestFastReplayEngine(FastReplayEngine):
        def __init__(self, df):
            super().__init__(df)
            test_dump = np.load('artifacts/models/transaction/test_dump.npz')
            self.tx_probs = dict(zip(test_dump['ids'], test_dump['y_prob']))
            
    engine = TestFastReplayEngine(test_df)
    # The parent constructor loads val_dump, but our child overrode tx_probs safely.
    
    test_df_eval = test_preds.sort_values('TransactionDT').reset_index(drop=True)
    all_windows = []
    for i, row in test_df_eval.iterrows():
        tx_dict = row.to_dict()
        if 'isFraud' in tx_dict:
            del tx_dict['isFraud']
        if 'fraud_probability' in tx_dict:
            del tx_dict['fraud_probability']
        windows = engine.process_transaction(tx_dict)
        all_windows.extend(windows)
        
    all_windows.extend(engine.flush_eos())
    
    df_windows = pd.DataFrame(all_windows)
    df_windows.to_parquet('artifacts/streaming/test_replay_windows.parquet')
    
    alerts = df_windows[df_windows['severity'] == 'HIGH'].copy()
    
    l2_metrics = evaluate_spike_detection(alerts, events_df, df_windows, q10_vol)
    
    print("\n--- Layer 2 Final Metrics ---")
    print(json.dumps(l2_metrics, indent=4))
    
    # Evaluate States
    states = df_windows['incident_state']
    prev_states = states.shift(1, fill_value='RESOLVED')
    open_transitions = ((states == 'OPEN') & (prev_states == 'RESOLVED')).sum()
    distinct_incidents = df_windows['incident_id'].nunique()
    
    print(f"\nPredicted Incidents Opened: {distinct_incidents}")
    
    # Business Simulation
    print("\n--- Business Simulation (Test Set) ---")
    
    # To compute transaction risk costs, we must evaluate the test set TP/FP/FN/TN
    # We already have tn, fp, fn, tp from l1_metrics.
    
    scenarios = {
        "LOW": {"c_rev": 2, "c_fric": 5, "lgf": 0.50, "c_ops": 10},
        "BASE": {"c_rev": 10, "c_fric": 15, "lgf": 0.80, "c_ops": 25},
        "HIGH": {"c_rev": 25, "c_fric": 50, "lgf": 1.00, "c_ops": 50}
    }
    
    # Wait, the amount at risk per transaction is not just count. We need the actual amounts for transaction risk!
    # LGF applies to the amount.
    tp_amts = test_df_eval[(y_test == 1) & (calib_probs >= tau_tx)]['TransactionAmt'].sum()
    fn_amts = test_df_eval[(y_test == 1) & (calib_probs < tau_tx)]['TransactionAmt'].sum()
    
    for name, s in scenarios.items():
        # False spike alert cost
        c_alert_review = 500
        ops_cost = distinct_incidents * c_alert_review
        
        # Tx Risk cost:
        # False Positive friction cost: fp * c_fric
        # True Positive review cost: tp * c_rev
        # False Negative missed fraud cost: fn_amts * lgf
        fric_cost = l1_metrics['fp'] * s['c_fric']
        rev_cost = l1_metrics['tp'] * s['c_rev']
        missed_cost = fn_amts * s['lgf']
        
        tx_risk_cost = fric_cost + rev_cost + missed_cost
        
        print(f"\nScenario {name}:")
        print(f"Transaction-Risk Economic Simulation: ${tx_risk_cost:,.2f}")
        print(f"Spike Incident-Response Operational Cost: ${ops_cost:,.2f} (from {distinct_incidents} predicted incidents)")
        print(f"Note: These are illustrative assumptions, not merchant-provided economics.")
        
    print("\nVerifying artifacts remain unchanged...")
    hashes_after = verify_upstream_artifacts()
    assert hashes_before == hashes_after, "Artifact hash mismatch!"
    
    print("\nGenerating final run manifest...")
    manifest = {
        "dataset_split_identifiers": {
            "test_rows": len(test_df),
            "test_fraud_prevalence": float(np.mean(y_test))
        },
        "model_version": "Layer1-v3.4.1",
        "feature_manifest_version": "v2",
        "calibration_method": "Sigmoid (LogisticRegression)",
        "tau_tx": tau_tx,
        "window_size": w_size,
        "N": 20,
        "Z_threshold": 3.0,
        "Q10": q10_vol,
        "proxy_spike_threshold": p975,
        "seeds": 42,
        "artifact_hashes": hashes_after
    }
    
    with open('artifacts/manifests/final_run_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=4)
        
if __name__ == '__main__':
    main()
