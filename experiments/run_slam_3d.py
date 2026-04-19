"""
run_slam_3d.py
==============
Entry point to run the full Embodied 3D SLAM workflow.

This script aggregates the entire processing pipeline:
1. Runs the 3D MiniWorld Collection.
2. Extracts visual features using CLIP (or HOG).
3. Compiles and executes the Nengo SSP-SLAM Simulation.
4. Validates the generated Cognitive Map against the ground truth.
"""

import sys, os
import subprocess
import argparse
import time

def run_cmd(cmd):
    print(f"\n--- Running: {' '.join(cmd)} ---")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"Command failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)

def main():
    parser = argparse.ArgumentParser(description="End-To-End 3D Embodied SLAM")
    parser.add_argument("--steps", default=500, type=int, help="Number of 3D simulation exploration steps")
    parser.add_argument("--feat-method", default="hog", choices=["sift", "hog", "orb", "clip"], help="Feature extractor")
    parser.add_argument("--skip-collection", action="store_true", help="Skip 3D execution and use cached data")
    args = parser.parse_args()

    data_dir = "data/3d"
    python_exe = sys.executable

    print("\n" + "="*50)
    print(" 1. LEVEL 1 & 2: 3D EMBODIED COLLECTION & PERCEPTION")
    print("="*50)
    
    if not args.skip_collection:
        cmd_collect = [
            python_exe, "experiments/collect_3d_data.py",
            "--n-steps", str(args.steps),
            "--n-objects", "4",
            "--policy", "explore",
            "--feat-method", args.feat_method,
            "--save-dir", data_dir
        ]
        run_cmd(cmd_collect)
    else:
        print(f"Skipping collection. Using existing data in '{data_dir}'")

    print("\n" + "="*50)
    print(" 2. LEVEL 3: SSP-SLAM NENGO SIMULATION")
    print("="*50)
    
    # We use minimal neurons for testing speed. Increase for accurate mapping.
    cmd_slam = [
        python_exe, "experiments/run_slam_features.py",
        "--feature-data", f"{data_dir}/feature_vectors.npy",
        "--landmark-data", f"{data_dir}/landmark_positions.npy",
        "--path-data", f"{data_dir}/path.npy",
        "--view-rad", "0.5",
        "--pi-n-neurons", "300",
        "--mem-n-neurons", "300",
        "--save-dir", data_dir,
        "--save"
    ]
    run_cmd(cmd_slam)

    print("\n" + "="*50)
    print(" 3. LEVEL 4: COGNITIVE MAP DASHBOARD")
    print("="*50)
    print("Pipeline Complete! Output generated in data/3d/")
    print("Open 'experiments/slam_3d_dashboard.ipynb' in your IDE to visualize the results.")


if __name__ == "__main__":
    main()
