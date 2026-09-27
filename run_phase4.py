import pandas as pd
import numpy as np
import json
import os
import warnings
import hashlib

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

from src.fraud_spike.spike.aggregate import build_tumbling_windows
from src.fraud_spike.spike.rolling_baseline import calculate_rolling_baseline
from src.fraud_spike.spike.zscore import compute_zscore_detector
from src.fraud_spike.spike.ewma import compute_ewma_detector
from src.fraud_spike.spike.cusum import compute_cusum_detector
from src.fraud_spike.spike.ground_truth import compute_q10_eligible_volume, compute_proxy_spike_thresholds, create_spike_events
from src.fraud_spike.evaluation.spike_metrics import evaluate_spike_detection

# Fixed Layer-1 Configurations (from Phase 3)
TAU_TX = 0.1314

def load_data():
    print("Loading data...")
    train_df = pd.read_parquet('data/processed/train/train.parquet', columns=['TransactionID', 'TransactionDT', 'isFraud'])
    val_df = pd.read_parquet('data/processed/validation/validation.parquet', columns=['TransactionID', 'TransactionDT', 'isFraud'])
    
    # Load Phase 3 predictions for validation
    val_dump = np.load('artifacts/models/transaction/val_dump.npz')
    preds_df = pd.DataFrame({
        'TransactionID': val_dump['ids'],
        'fraud_probability': val_dump['y_prob'], # This is the adopted Sigmoid calibrated prob
        'TransactionAmt': val_dump['amts']
    })
    
    # Merge validation DT
    val_preds = pd.merge(preds_df, val_df[['TransactionID', 'TransactionDT']], on='TransactionID', how='inner')
    
    assert len(val_preds) == len(preds_df), "Mismatch during ID merge for DT"
    
    return train_df, val_df, val_preds

def main():
    train_df, val_df, val_preds = load_data()
    
    window_sizes = [900, 1800, 3600, 21600]
    history_lengths = [5, 10, 20]
    
    results = {}
    best_f1 = -1
    best_config = {}
    best_telemetry = None
    
    print("Evaluating Phase 4 Configurations...")
    for w_size in window_sizes:
        print(f"\n--- Window Size: {w_size}s ---")
        
        # 1. Ground Truth Generation
        q10_vol = compute_q10_eligible_volume(train_df, w_size)
        print(f"Q10 Eligible Volume: {q10_vol}")
        
        thresholds = compute_proxy_spike_thresholds(val_df, w_size, q10_vol)
        p975 = thresholds['p975']
        print(f"p975 Threshold: {p975:.4f}")
        
        events_df = create_spike_events(val_df, w_size, q10_vol, p975)
        print(f"Proxy Spike Events found: {len(events_df)}")
        
        # 2. Aggregation
        windows_df = build_tumbling_windows(val_preds, w_size, TAU_TX)
        
        for N in history_lengths:
            # 3. Rolling Baseline
            base_df = calculate_rolling_baseline(windows_df, N, signal_col='suspicious_rate')
            
            # 4. Detectors
            # A. Z-score (Primary)
            z_df = compute_zscore_detector(base_df, 'suspicious_rate', N)
            
            # B. EWMA
            # Just computing scores; not tuning alert threshold explicitly here to keep it simple, 
            # we will evaluate Z-score primarily for alerts.
            ewma_df = compute_ewma_detector(z_df, 'suspicious_rate', N, alpha=0.2)
            
            # C. CUSUM
            cusum_df = compute_cusum_detector(ewma_df, 'suspicious_rate', N, drift=0.01)
            
            # 5. Evaluate Z-score performance (which outputs candidate_alert_flag)
            alerts = cusum_df[cusum_df['candidate_alert_flag'] == 1]
            metrics = evaluate_spike_detection(alerts, events_df, windows_df, q10_vol)
            
            config_key = f"{w_size}s_N{N}"
            
            metrics['q10_volume'] = q10_vol
            metrics['p975_threshold'] = p975
            
            results[config_key] = metrics
            
            print(f"  N={N} | Precision: {metrics['strict']['event_precision']:.3f} | Recall: {metrics['strict']['event_recall']:.3f} | FAR: {metrics['false_alert_window_rate']:.4f}")
            
            # Simple heuristic for "best" configuration: F1 score equivalent of event precision and recall
            prec = metrics['strict']['event_precision']
            rec = metrics['strict']['event_recall']
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            # Tie breaker goes to the primary 3600s
            if f1 > best_f1 or (f1 == best_f1 and w_size == 3600 and N == 10):
                best_f1 = f1
                best_config = {
                    'window_size': w_size,
                    'history_length': N,
                    'precision': prec,
                    'recall': rec,
                    'far': metrics['false_alert_window_rate']
                }
                best_telemetry = cusum_df
                
    # Use 3600 as the definitive fallback if performance is zero everywhere
    if best_config.get('window_size') is None:
        best_config = {'window_size': 3600, 'history_length': 10}
        
    print("\nBest Configuration Selected:")
    print(best_config)
    
    # Save artifacts
    os.makedirs('artifacts/manifests', exist_ok=True)
    os.makedirs('artifacts/windows', exist_ok=True)
    os.makedirs('artifacts/metrics', exist_ok=True)
    
    with open('artifacts/manifests/spike_manifest.json', 'w') as f:
        json.dump({
            'chosen_window_size': best_config['window_size'],
            'chosen_history_length': best_config.get('history_length', 10),
            'detector': 'zscore',
            'warm_up_policy': f"Wait for {best_config.get('history_length', 10)} eligible windows",
            'tau_spike': 3.0, # The default we hardcoded in zscore.py
            'tau_tx_hash': compute_sha256('artifacts/models/transaction/tau_tx.json')
        }, f, indent=4)
        
    with open('artifacts/metrics/phase4_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    if best_telemetry is not None:
        best_telemetry.to_csv('artifacts/windows/validation_spike_telemetry.csv', index=False)
        
    print("Phase 4 Complete. Results and telemetry saved.")

if __name__ == "__main__":
    # Suppress runtime warnings from zero divisions which we handle via epsilon
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
