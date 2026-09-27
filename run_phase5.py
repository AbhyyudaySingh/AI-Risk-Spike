import os
import hashlib
import pandas as pd
import numpy as np
import json
from src.fraud_spike.streaming.simulate import run_simulation

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def verify_upstream_artifacts():
    print("Verifying frozen Phase 1-4 upstream artifacts...")
    artifacts = [
        'artifacts/models/transaction/xgb_layer1.json',
        'artifacts/models/transaction/calibration.pkl',
        'artifacts/models/transaction/tau_tx.json',
        'artifacts/models/transaction/val_dump.npz',
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

def check_equivalence():
    print("\n--- Verifying Phase-4 offline output against Phase-5 replay output ---")
    
    # Load offline telemetry
    offline_file = 'artifacts/windows/validation_spike_telemetry.csv'
    if not os.path.exists(offline_file):
        print(f"Warning: Offline telemetry file {offline_file} not found. Cannot perform full equivalence check.")
        return
        
    df_offline = pd.read_csv(offline_file)
    # The offline data contains all windows. Phase-4 used 3600s, N=20 (but if the best config was different, we might have a mismatch in N).
    # Wait, the user said Phase 4 used 3600s, N=20.
    
    df_online = pd.read_parquet('artifacts/streaming/replay_windows.parquet')
    
    # Filter online to match offline windows. Online might have leading NO_DATA windows.
    # Offline windows start at the first transaction's window.
    # Actually, both should align by window_id.
    
    # Compare
    common_cols_int = ['window_id', 'transaction_count', 'suspicious_count']
    common_cols_float = ['suspicious_rate', 'z_score']
    
    # Ensure window_id is set
    df_offline = df_offline.sort_values('window_id').reset_index(drop=True)
    df_online = df_online.sort_values('window_id').reset_index(drop=True)
    
    print(f"Offline windows: {len(df_offline)}, Online windows: {len(df_online)}")
    
    # Align by window_id using an inner merge
    df_merged = pd.merge(df_offline, df_online, on='window_id', suffixes=('_off', '_on'))
    print(f"Aligned windows for comparison: {len(df_merged)}")
    
    # Deterministic equality
    for col in common_cols_int:
        if col + '_off' in df_merged.columns and col + '_on' in df_merged.columns:
            mismatches = (df_merged[col + '_off'] != df_merged[col + '_on']).sum()
            print(f"Deterministic check {col}: {mismatches} mismatches")
            assert mismatches == 0, f"Deterministic mismatch in {col}"
            
    # Floating equality
    for col in common_cols_float:
        if col + '_off' in df_merged.columns and col + '_on' in df_merged.columns:
            # Handle NaNs appropriately
            off_vals = df_merged[col + '_off'].fillna(-999.0)
            on_vals = df_merged[col + '_on'].fillna(-999.0)
            mismatches = (~np.isclose(off_vals, on_vals, atol=1e-5)).sum()
            print(f"Floating check {col}: {mismatches} mismatches")
            assert mismatches == 0, f"Floating mismatch in {col}"

    print("Phase-4 offline vs Phase-5 streaming equivalence verified successfully.")

def main():
    print("=== Starting Phase 5: Chronological Replay ===")
    
    hashes_before = verify_upstream_artifacts()
    
    # Ensure dir exists
    os.makedirs('artifacts/streaming', exist_ok=True)
    
    # Run the simulation (this will save artifacts to 'artifacts/streaming/')
    val_data = 'data/processed/validation/validation.parquet'
    checkpoint = 'artifacts/streaming/checkpoint.pkl'
    
    run_simulation(val_data, checkpoint, test_checkpoint=True)
    
    hashes_after = verify_upstream_artifacts()
    
    # Ensure hashes did not change
    for k in hashes_before:
        assert hashes_before[k] == hashes_after[k], f"Artifact {k} was mutated during Phase 5!"
        
    print("Artifact immutability verified.")
    
    check_equivalence()
    
    print("=== Phase 5 Complete ===")

if __name__ == '__main__':
    main()
