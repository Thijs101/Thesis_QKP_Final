import sys
import os
sys.path.insert(0, os.path.abspath('src'))

import os
import pandas as pd
import importlib
import numpy as np
import time

# Import testing logic from other files
from gallo_mountain_curve import collect_gallo_curve_data
from dks_mountain_curve import collect_dks_curve_data
import creating_tables

def run_all():
    """
    Runs all thesis experiments sequentially to generate data and plots.
    """
    print("Starting Thesis Benchmark Suite...")

    # --- 1. Global Settings ---
    num_instances = 30
    gurobi_time_limit = 600
    fennich_max_gallo = 75
    fennich_max_dks = 200

    # --- Step 1: DKS Mountain Curves ---
    print("\nRunning DKS Mountain Curves...")
    n_sizes_dks = [10, 25, 50, 100, 150, 200, 250, 300]
    densities_dks = [0.5, 0.75, 0.9]
    beam_steps_dks = [1, 2, 3, 5, 10, 20, 35, 50]
    
    for n in n_sizes_dks:
        for d in densities_dks:
            print(f"\n--- Testing DKS: N={n}, Density={d} ---")
            collect_dks_curve_data(n, d, beam_steps_dks, num_instances, gurobi_time_limit)

    # --- Step 2: Gallo Mountain Curves ---
    print("\nRunning Gallo Mountain Curves...")
    n_sizes_gallo = [10, 25, 50, 100, 150]
    densities_gallo = [0.25, 0.50, 0.75, 1.0]
    beam_steps_gallo = [1, 2, 3, 5, 10, 20]
    
    for n in n_sizes_gallo:
        for d in densities_gallo:
            print(f"\n--- Testing Gallo: N={n}, Density={d} ---")
            collect_gallo_curve_data(n, d, beam_steps_gallo, num_instances, gurobi_time_limit)

    # --- Step 3: Unified Thesis Raw Data (Fennich + Proposed) ---
    print("\nGenerating Unified Raw Data (Hidden Clique)...")
    n_sizes_hc = [10, 25, 40, 50, 60, 75, 85, 90, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600]
    beam_steps_hc = [1, 2, 3, 5, 10, 25, 35, 40, 50, 60, 75, 85, 100, 110, 125, 150, 175, 200, 225, 250, 275, 300]
    csv_diverse = "data/volledige_onderzoek_results_diverse.csv"
    csv_standard = "data/volledige_onderzoek_results_standard.csv"
    
    print("\nRunning Diverse vs Standard comparison...")
    for n in n_sizes_hc:
        creating_tables.collect_curve_data(n, beam_steps_hc, num_instances, csv_file=csv_diverse, use_diversity=True)
        creating_tables.collect_curve_data(n, beam_steps_hc, num_instances, csv_file=csv_standard, use_diversity=False)
    
    def get_thesis_bw(n, inst_type):
        if inst_type == "Gallo" or inst_type == "DKS": return 5
        if inst_type == "Hidden_Clique":
            a, b, c, d = 0.2102, 201.3693, 216.7704, 73.5567
            bw = a * n + b * np.exp(-((n - c)**2) / (2 * d**2))
            return max(1, int(round(bw)))

    # Hidden Clique (Full Range) — Unbuffered baseline
    for n in n_sizes_hc:
        bw = get_thesis_bw(n, "Hidden_Clique")
        creating_tables.collect_raw_thesis_data(n, 0.0, "Hidden_Clique", bw, num_instances=num_instances)
        
    # --- Step 4: Baseline Buffer Sweep Study ---
    print("\nRunning Baseline Buffer Sweep...")
    from evaluate_buffer_levels import run_buffer_optimization_study
    run_buffer_optimization_study(csv_file="data/buffer_evaluation_results.csv")

    # --- Step 5: Safety Buffer Addition ---
    print("\nApplying 5% Buffer to Raw Data...")
    from src.add_buffered_raw_data import add_buffered_data
    add_buffered_data(raw_file="data/thesis_raw_data.csv", buffer_file="data/buffer_evaluation_results.csv")

    # --- Step 6: Buffer Validation Study (N=700-1200) ---
    print("\nRunning Buffer Validation Study...")
    from evaluate_buffer_levels import run_validation_study
    val_n_sizes = [700, 800, 900, 1000, 1100, 1200]
    val_buffers = [0.0, 0.05]
    run_validation_study(
        csv_file="data/buffer_validation_results.csv",
        n_sizes=val_n_sizes,
        buffer_levels=val_buffers,
    )
    
    # Process DKS instances for Mountain Curves & Impact Analysis
    for n in [10, 25, 50, 100, 150, 200, 250, 300]:
        for d in [0.5, 0.75, 0.9]:
            bw = get_thesis_bw(n, "DKS")
            creating_tables.collect_raw_thesis_data(n, d, "DKS", bw, num_instances=num_instances, fennich_max_n=fennich_max_dks)
            
    # Process Gallo instances for Mountain Curves & Impact Analysis
    for n in [10, 25, 50, 75, 100, 150]:
        for d in [0.25, 0.5, 0.75, 1.0]:
            bw = get_thesis_bw(n, "Gallo")
            creating_tables.collect_raw_thesis_data(n, d, "Gallo", bw, num_instances=num_instances, fennich_max_n=fennich_max_gallo)

    # --- Step 7: Ablation Study ---
    print("\nRunning Ablation Study...")
    from run_ablation_study import run_all_ablations
    
    run_all_ablations(inst_types=["DKS"], n_sizes=[10, 25, 50, 100, 150, 200, 250, 300], num_instances=num_instances, densities=[0.5, 0.75, 0.9])
    run_all_ablations(inst_types=["Gallo"], n_sizes=[10, 25, 50, 75, 100, 150], num_instances=num_instances, densities=[0.25, 0.5, 0.75, 1.0])
    run_all_ablations(inst_types=["Hidden_Clique"], n_sizes=n_sizes_hc, num_instances=num_instances)

    # --- Step 8: Local Search Impact ---
    print("\nMeasuring Local Search Gain...")
    from run_local_search_impact import run_ls_impact_main
    
    run_ls_impact_main(inst_types=["DKS"], n_sizes=[10, 25, 50, 100, 150, 200, 250, 300], num_instances=num_instances, densities=[0.5, 0.75, 0.9])
    run_ls_impact_main(inst_types=["Gallo"], n_sizes=[10, 25, 50, 75, 100, 150], num_instances=num_instances, densities=[0.25, 0.5, 0.75, 1.0])
    run_ls_impact_main(inst_types=["Hidden_Clique"], n_sizes=n_sizes_hc, num_instances=num_instances)
    
    # --- Step 9: Aggregated Anchor Study ---
    print("\nRunning Aggregated Anchor Study...")
    from evaluate_buffer_levels import run_aggregated_anchor_study
    
    # Analyze optimal inclusion percentage on high difficulty instances
    aggregated_anchor_pcts = [0.05, 0.10, 0.15, 0.20, 0.25]
    for pct in aggregated_anchor_pcts:
        run_aggregated_anchor_study(csv_file="data/aggregated_anchor_evaluation_results.csv", aggregated_anchor_pct=pct, n_sizes=[300, 400, 500, 600], num_instances=num_instances)
        
    # Standardize the optimal 15% aggregated anchor rate over the full baseline scope
    chosen_aggregated_anchor_pct = 0.15
    print(f"\nRunning Full Study (Anchor Pct: {chosen_aggregated_anchor_pct*100:.1f}%)")
    run_aggregated_anchor_study(csv_file="data/aggregated_anchor_evaluation_results.csv", aggregated_anchor_pct=chosen_aggregated_anchor_pct, n_sizes=n_sizes_hc, num_instances=num_instances)
    run_aggregated_anchor_study(csv_file="data/aggregated_anchor_evaluation_results.csv", aggregated_anchor_pct=chosen_aggregated_anchor_pct, n_sizes=[700, 800, 900, 1000, 1100, 1200], num_instances=num_instances)
    
    # --- Step 10: Aggregated Anchor Mountain Curves ---
    print("\nGenerating Aggregated Anchor Mountain Curves...")
    n_sizes_aggregated_anchor = [10, 25, 40, 50, 60, 75, 85, 90, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600]
    beam_steps_aggregated_anchor = [1, 2, 3, 5, 10, 25, 35, 40, 50, 60, 75, 85, 100, 110, 125, 150, 175, 200, 225, 250, 275, 300]
    csv_aggregated_anchor = "data/volledige_onderzoek_results_aggregated_anchor.csv"
    
    for n in n_sizes_aggregated_anchor:
        creating_tables.collect_curve_data(n, beam_steps_aggregated_anchor, num_instances, csv_file=csv_aggregated_anchor, use_diversity=True, use_aggregated_anchor=True, aggregated_anchor_pct=0.15)
        
    # --- Step 11: Dual Calibration Buffer Validation ---
    print("\nRunning Dual Calibration Buffer Validation Study...")
    val_n_unbuf = [300, 400, 500, 600]
    val_n_buf = [10, 25, 40, 50, 60, 75, 85, 90, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    
    run_validation_study(csv_file="data/buffer_validation_results_aggregated_anchor.csv", n_sizes=val_n_unbuf, buffer_levels=[0.0], algo="aggregated_anchor")
    run_validation_study(csv_file="data/buffer_validation_results_aggregated_anchor.csv", n_sizes=val_n_buf, buffer_levels=[0.05], algo="aggregated_anchor")
    
    # --- Step 12: Visualizations ---
    print("\nGenerating Visualizations...")
    import subprocess
    import sys
    
    scripts_to_run = [
        "src/analyze_gallo_dks.py",
        "src/plot_raw_data.py",
        "src/generate_thesis_plots.py",
        "src/plot_thesis_ablation_all.py",
        "src/plot_local_search_impact.py",
        "src/fit_confirmed_crossing.py"
    ]
    
    for script in scripts_to_run:
        print(f"Processing {script}...")
        try:
            subprocess.run([sys.executable, script], check=True)
            print(f"Done: {script} visualizations created.")
        except subprocess.CalledProcessError:
            print(f"Failed: {script} encountered an issue.")

    print("\nBenchmarks complete.")

if __name__ == "__main__":
    run_all()
