"""
Baseline MAE measurement script.
Runs the CURRENT vision_engine hyperparameters against the 15-frame ground truth
and logs the MAE. This is NOT a grid search — it evaluates only the production config.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tune_parameters import load_ground_truth, test_configuration

# Current production hyperparameters (from vision_engine.py)
CURRENT_PARAMS = {
    'varThreshold': 16,
    'history': 500,
    'morph_kernel': (7, 50),
    'dilate_kernel': 3,
}

def main():
    video_path, frames_db = load_ground_truth()
    annotated_indices = sorted(list(frames_db.keys()))
    max_frame = annotated_indices[-1]

    print("=" * 60)
    print(" BASELINE MAE MEASUREMENT")
    print("=" * 60)
    print(f"Ground truth frames: {len(frames_db)}")
    print(f"Frame indices: {annotated_indices}")
    print(f"Parameters: {CURRENT_PARAMS}")
    print("-" * 60)

    mae = test_configuration(video_path, frames_db, max_frame, CURRENT_PARAMS)

    print(f"\n  BASELINE MAE = {mae:.2f} people\n")
    print("=" * 60)

    # Write to file
    out_path = os.path.join(os.path.dirname(__file__), 'mae_before.txt')
    with open(out_path, 'w') as f:
        f.write(f"Baseline MAE Measurement\n")
        f.write(f"========================\n")
        f.write(f"Date: 2026-04-25\n")
        f.write(f"Ground truth frames: {len(frames_db)}\n")
        f.write(f"Frame indices: {annotated_indices}\n")
        f.write(f"Parameters:\n")
        for k, v in CURRENT_PARAMS.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nMAE = {mae:.2f} people\n")
    print(f"Result written to: {out_path}")

if __name__ == '__main__':
    main()
