import argparse
import os
import sys

def verify_artifacts():
    print("Verifying artifacts exist...")
    required = [
        'artifacts/streaming/test_replay_windows.parquet',
        'artifacts/manifests/final_run_manifest.json'
    ]
    for r in required:
        if not os.path.exists(r):
            print(f"Error: Required artifact {r} is missing.")
            print("Please ensure you have run Phase 6 to generate the final test replay.")
            sys.exit(1)
    print("All artifacts verified. Replay data is ready for the demo dashboard.")

def regenerate_replay():
    print("Regenerating replay data (Calling run_phase6.py)...")
    if os.path.exists('run_phase6.py'):
        import subprocess
        subprocess.run(['python', 'run_phase6.py'], check=True)
    else:
        print("Error: run_phase6.py not found.")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Risk Spike Demo Runner")
    parser.add_argument('--replay', action='store_true', help="Regenerate the chronological replay before starting")
    
    args = parser.parse_args()
    
    if args.replay:
        regenerate_replay()
    else:
        verify_artifacts()
        
    print("\nTo launch the dashboard, run:")
    print("streamlit run app.py")
