import time
import os
import json
import numpy as np
import pandas as pd
from Dynamic_Programming_QKP import generate_benchmark_instance, fl_heuristic_enhanced, solve_qkp_gurobi
from Fennich_algorithm import fennich_dp

# Step 1: Set up the Gurobi Cache.
# Caches Gurobi results to avoid re-solving instances across multiple runs.
CACHE_FILE = 'data/dks_gurobi_cache.json'
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        baseline_cache = json.load(f)
else:
    baseline_cache = {}

def run_experiment_DKS(n_val, density, num_instances=30, beam_width=1, gurobi_time_limit=600):
    """
    Solves a set of DKS problems (unit weights) and finds the average profit and time.
    """
    fl_times = []
    fl_values = []
    instance_gaps = []
    gurobi_limit_hits = 0
    
    # Limit Gurobi execution to smaller instances (N <= 30) due to exponential time scaling
    use_gurobi = (n_val <= 30)
    
    for instance_num in range(num_instances):
        seed = instance_num + n_val * 1000 + int(density * 100)
        n_inst, capacity, Q, weights, k = generate_benchmark_instance(n_val, density, seed=seed)
        
        g_val = None
        g_gap = None
        
        if use_gurobi:
            # Check cache for existing solution
            cache_key = f"{n_val}_{density}_{seed}"
            if cache_key not in baseline_cache or 'gurobi_val' not in baseline_cache[cache_key]:
                if cache_key not in baseline_cache: baseline_cache[cache_key] = {}
                print(f"    [Gurobi Solver] Solving N={n_val}, Seed={seed} for the first time...", end=' ', flush=True)
                _, g_val, g_gap = solve_qkp_gurobi(n_inst, capacity, Q, weights, time_limit=gurobi_time_limit)
                
                # Update cache
                baseline_cache[cache_key]['gurobi_val'] = g_val
                baseline_cache[cache_key]['gurobi_gap'] = g_gap
                with open(CACHE_FILE, 'w') as f:
                    json.dump(baseline_cache, f)
                print("Done.")
            
            g_val = baseline_cache[cache_key]['gurobi_val']
            g_gap = baseline_cache[cache_key].get('gurobi_gap', 0.0)
            if g_gap is not None and g_gap > 1e-6:
                gurobi_limit_hits += 1

        # Run heuristic
        start = time.process_time()
        _, fl_value, *_ = fl_heuristic_enhanced(n_inst, capacity, Q, weights, beam_width=beam_width)
        fl_time = time.process_time() - start
        
        fl_values.append(fl_value)
        fl_times.append(fl_time)
        
        # Progress output
        print(f"      Instance {instance_num+1:2d}/{num_instances}: Profit = {fl_value:8.2f} | Time = {fl_time:6.3f}s", end='\r', flush=True)
        
        # Calculate optimality gap if Gurobi optimal is available
        if use_gurobi and g_val is not None and (g_gap is None or g_gap < 1e-6):
            gap = 100 * (g_val - fl_value) / g_val if g_val > 0 else 0
            instance_gaps.append(gap)
            
    # Aggregate results
    res_dict = {
        'n': n_val, 'density': density, 'beam_width': beam_width,
        'avg_objective': np.mean(fl_values), 'avg time (s)': np.mean(fl_times),
        'avg gap (%)': np.mean(instance_gaps) if instance_gaps else None,
        'num_instances': num_instances,
        'gurobi_limit_hits': gurobi_limit_hits if use_gurobi else 0
    }
    return pd.DataFrame([res_dict])

def collect_dks_curve_data(n_val, density, beam_steps, num_instances=30, gurobi_time_limit=600, csv_file="data/dks_mountain_curve.csv"):
    """
    This is the main function called by the benchmark suite to get DKS data.
    """

    # 2. Run our heuristic for different beam widths
    done_configs = set()
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        for _, row in df_existing.iterrows():
            done_configs.add((int(row['n']), float(row['density']), int(row['beam_width'])))

    for b in beam_steps:
        if (n_val, density, b) in done_configs:
            continue
            
        print(f"  Testing Beam Width B={b}...")
        df_temp = run_experiment_DKS(n_val, density, num_instances, b, gurobi_time_limit)
        df_temp.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
        res = df_temp.iloc[0]
        gap_str = f"{res['avg gap (%)']:.4f}%" if pd.notna(res['avg gap (%)']) else "N/A"
        print(f"    -> Profit: {res['avg_objective']:.2f} | Gap: {gap_str} | Time: {res['avg time (s)']:.2f}s")
