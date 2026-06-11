from Dynamic_Programming_QKP import generate_gallo_instance
from Dynamic_Programming_QKP import generate_hidden_clique
from Dynamic_Programming_QKP import solve_qkp_gurobi
from Dynamic_Programming_QKP import fl_heuristic_enhanced
from Dynamic_Programming_QKP import generate_benchmark_instance
from Fennich_algorithm import fennich_dp
import time
import numpy as np
import pandas as pd
import os
import json

def collect_raw_thesis_data(n_val, density, inst_type, beam_width, num_instances=30, csv_file="data/thesis_raw_data.csv", fennich_max_n=None):
    """
    Collects raw performance data for Fennich and the Proposed Heuristic.
    """
    # 1. Load existing IDs once for fast skipping
    done_ids = set()
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        mask = (df_existing['inst_type'] == inst_type) & (df_existing['n'] == n_val) & (df_existing['density'] == density) & (df_existing['beam_width'] == beam_width)
        done_ids = set(df_existing[mask]['instance_id'].unique())

    print(f"Collecting Data: {inst_type} (N={n_val}, D={density}, BW={beam_width})")
    
    for i in range(num_instances):
        if i in done_ids:
            continue
        # 1. Standardized Seed Logic
        if inst_type == "Hidden_Clique":
            seed = i + n_val * 1000
            n, cap, Q, w, clique, optimal = generate_hidden_clique(n_val, seed=seed)
        elif inst_type == "DKS":
            seed = i + n_val * 1000 + int(density * 100)
            n, cap, Q, w, k = generate_benchmark_instance(n_val, density, seed=seed)
            # Find optimal via Gurobi cache or dynamic solve
            optimal = None
            cache_file = "data/dks_gurobi_cache.json"
            cache_key = f"{n_val}_{density}_{seed}"
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache = json.load(f)
                    if cache_key in cache and 'gurobi_val' in cache[cache_key]:
                        optimal = cache[cache_key]['gurobi_val']
                except Exception:
                    pass
            if optimal is None and n_val <= 30:
                try:
                    _, g_val, g_gap = solve_qkp_gurobi(n, cap, Q, w, time_limit=600)
                    if g_val is not None:
                        optimal = g_val
                        # Save to cache
                        cache = {}
                        if os.path.exists(cache_file):
                            with open(cache_file, 'r') as f: cache = json.load(f)
                        cache[cache_key] = {'gurobi_val': g_val, 'gurobi_gap': g_gap}
                        with open(cache_file, 'w') as f: json.dump(cache, f)
                except Exception:
                    pass
        elif inst_type == "Gallo":
            seed = i + n_val * 1000 + int(density * 100)
            n, cap, Q, w = generate_gallo_instance(n_val, density, seed=seed)
            # Find optimal via Gurobi cache or dynamic solve
            optimal = None
            cache_file = "data/gallo_gurobi_cache.json"
            cache_key = f"{n_val}_{density}_{seed}"
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache = json.load(f)
                    if cache_key in cache and 'gurobi_val' in cache[cache_key]:
                        optimal = cache[cache_key]['gurobi_val']
                except Exception:
                    pass
            if optimal is None and n_val <= 50:
                try:
                    _, g_val, g_gap = solve_qkp_gurobi(n, cap, Q, w, time_limit=600)
                    if g_val is not None:
                        optimal = g_val
                        # Save to cache
                        cache = {}
                        if os.path.exists(cache_file):
                            with open(cache_file, 'r') as f: cache = json.load(f)
                        cache[cache_key] = {'gurobi_val': g_val, 'gurobi_gap': g_gap}
                        with open(cache_file, 'w') as f: json.dump(cache, f)
                except Exception:
                    pass

        # 2. Run Fennich
        skip_fennich = False
        if fennich_max_n is not None:
            if n_val > fennich_max_n:
                skip_fennich = True
        else:
            if (inst_type == "Gallo" and n_val > 75) or (inst_type == "DKS" and n_val > 200):
                skip_fennich = True

        if skip_fennich:
            fn_final, fn_dp, fn_time_dp, fn_time_ls, fn_time_total, fn_ls_iters = None, None, None, None, None, None
        else:
            _, fn_final, fn_dp, fn_time_dp, fn_time_ls, fn_ls_iters = fennich_dp(n, cap, w, Q)
            fn_time_total = fn_time_dp + fn_time_ls
        
        # 3. Run Proposed Heuristic
        _, fl_final, _, fl_dp, _, _, fl_ls_iters, fl_uniq_sols, fl_time_dp, fl_time_ls = fl_heuristic_enhanced(
            n, cap, Q, w, use_dynamic=True, beam_width=beam_width, multi_path_ls=True, use_diversity=True
        )

        # 4. Record everything
        res_row = {
            'inst_type': inst_type, 'n': n_val, 'density': density, 'instance_id': i, 'beam_width': beam_width,
            'optimal': optimal,
            'fn_dp_obj': fn_dp, 'fn_final_obj': fn_final, 'fn_time_dp': fn_time_dp, 'fn_time_ls': fn_time_ls, 
            'fn_time_total': fn_time_total, 'fn_ls_iters': fn_ls_iters,
            'fl_dp_obj': fl_dp, 'fl_final_obj': fl_final, 'fl_time_dp': fl_time_dp, 'fl_time_ls': fl_time_ls, 
            'fl_time_total': fl_time_dp + fl_time_ls, 'fl_ls_iters': fl_ls_iters, 'fl_unique_optima': fl_uniq_sols
        }
        
        # 5. Save IMMEDIATELY to disk
        df_row = pd.DataFrame([res_row])
        df_row.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
        
        if fn_final is not None:
            print(f"  Inst {i+1:2d}/{num_instances} | Fn: {fn_final:g} ({fn_time_total:.2f}s) | Our: {fl_final:g} ({fl_time_dp+fl_time_ls:.2f}s)", end='\r', flush=False)
        else:
            print(f"  Inst {i+1:2d}/{num_instances} | Fn: skipped | Our: {fl_final:g} ({fl_time_dp+fl_time_ls:.2f}s)", end='\r', flush=False)

    print(f"Finished {inst_type} (N={n_val})")




def run_experiment_HI(n_values, num_instances=10, use_dynamic=False, beam_width=1, multi_path_ls=False, use_diversity=False, lambda_div=0.5, use_aggregated_anchor=False, aggregated_anchor_pct=0.15):
    results = []
    
    total_configs = len(n_values)
    current_config = 0
    
    for n in n_values:

        current_config += 1
        print(f"[{current_config}/{total_configs}] Testing n={n}")
            

            
        instance_gaps = []
        fl_times = []

            
        for instance_num in range(num_instances):
            print(f"  Instance {instance_num+1}/{num_instances}...", end=' ')
                
                # Generate instance
            n_inst, capacity, Q, weights, clique, optimal = generate_hidden_clique(n, seed=instance_num + n*1000)
                
                # Run F&L
            start = time.process_time()
            fl_items, fl_value, fl_weight,*_ = fl_heuristic_enhanced(n_inst, capacity, Q, weights, use_dynamic=use_dynamic, beam_width=beam_width, multi_path_ls=multi_path_ls, use_diversity=use_diversity, use_aggregated_anchor=use_aggregated_anchor, aggregated_anchor_pct=aggregated_anchor_pct)
            fl_time = time.process_time() - start
                
                # Calculate gap
            gap = 100 * (optimal - fl_value) / optimal if optimal > 0 else 0
                
            instance_gaps.append(gap)
            fl_times.append(fl_time)
            print(f"Gap: {gap:.3f}%, FL: {fl_time:.3f}s")
        instance_gaps_arr=np.array(instance_gaps)
        success_count = np.sum(instance_gaps_arr == 0)
        success_rate = (success_count / num_instances) * 100
        
            # Store results
        if instance_gaps:
            results.append({
                'n': int(n),
                'avg gap (\%)': np.mean(instance_gaps),
                'std gap': np.std(instance_gaps),
                'min gap (\%)': np.min(instance_gaps),
                'max gap (\%)': np.max(instance_gaps),
                'success_rate (\%)': success_rate,
                'avg time (s)': np.mean(fl_times)})
    
    return pd.DataFrame(results)


def collect_curve_data(n_val, beam_steps, num_instances=30, csv_file="data/volledige_onderzoek_results.csv", use_diversity=True, use_aggregated_anchor=False, aggregated_anchor_pct=0.15):
    """
    Collects beam width curve data.
    """
    # 1. Read existing configurations to avoid redundant runs
    done_configs = set()
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        if 'n' in df_existing.columns and 'beam_width' in df_existing.columns:
            for _, row in df_existing.iterrows():
                done_configs.add((int(row['n']), int(row['beam_width'])))

    print(f"Collecting data for curve n={n_val} (Diversity={use_diversity})")
    
    for b in beam_steps:
        # 2. Skip if already exists
        if (n_val, b) in done_configs:
            print(f"Skipping B={b}... (Already in CSV)")
            continue
            
        print(f"Testing B={b}...")
        
        # Use the existing experiment function
        df_temp = run_experiment_HI([n_val], num_instances=num_instances, 
                                     use_dynamic=True, 
                                     beam_width=b, 
                                     multi_path_ls=True, 
                                     use_diversity=use_diversity,
                                     use_aggregated_anchor=use_aggregated_anchor,
                                     aggregated_anchor_pct=aggregated_anchor_pct)
        
        # Extract relevant data row
        res = df_temp.iloc[0]
        
        new_row = {
            'n': n_val,
            'beam_width': b,
            'avg_gap (%)': res['avg gap (\%)'],
            'std_gap': res['std gap'],
            'success_rate (%)': res['success_rate (\%)'],
            'avg_time (s)': res['avg time (s)'],
            'max_gap (%)': res['max gap (\%)']
        }
        
        avg_gap=res['avg gap (\%)']
        succes_rate=res['success_rate (\%)']
        print(f"Gap: {avg_gap:.3f}% | Success: {succes_rate:.1f}%")
        
        # 3. Save directly to CSV immediately 
        # (mode='a' appends, header=True only if file does not exist)
        df_new = pd.DataFrame([new_row])
        df_new.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

