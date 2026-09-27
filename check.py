import pandas as pd
import json

df = pd.read_parquet('artifacts/streaming/replay_windows.parquet')

high_windows = (df['severity'] == 'HIGH').sum()

# Analyze transitions using states
states = df['incident_state']
prev_states = states.shift(1, fill_value='RESOLVED')

open_transitions = ((states == 'OPEN') & (prev_states == 'RESOLVED')).sum()
# In ReplayEngine, an ACTIVE state means an ongoing HIGH severity alert
active_windows = (states == 'ACTIVE').sum()

resolved_transitions = ((states == 'RESOLVED') & (prev_states.isin(['OPEN', 'ACTIVE']))).sum()

distinct_incidents = df['incident_id'].nunique()
active_at_end = 1 if states.iloc[-1] in ['OPEN', 'ACTIVE'] else 0

print(f'HIGH windows: {high_windows}')
print(f'Distinct incidents opened: {distinct_incidents}')
print(f'OPEN transitions: {open_transitions}')
print(f'ACTIVE continuation HIGH windows: {active_windows}')
print(f'RESOLVED transitions: {resolved_transitions}')
print(f'Incidents still active at end: {active_at_end}')

high_df = df[df['severity'] == 'HIGH'].copy()
if not high_df.empty:
    sample = high_df.iloc[0].to_dict()
    print('\nSample Payload:')
    print(json.dumps(sample, indent=4, default=str))
    
    calc_z = (sample['suspicious_rate'] - sample['baseline_value']) / sample['baseline_std']
    print(f'\nMath Verify:')
    print(f'calc_z = {calc_z}')
    print(f'reported_z = {sample["z_score"]}')
