import os
import pandas as pd
import numpy as np


from Dynamic_Programming_QKP import generate_hidden_clique, fl_heuristic_enhanced

def get_thesis_bw(n, algo="diverse"):
    """Calculates the raw predicted beam width based on the Confirmed Crossing formula."""
    if algo == "aggregated_anchor":
        a, b, c, d = 0.4468, 184.5293, 191.9661, 56.8142
    else: # diverse
        a, b, c, d = 0.2102, 201.3693, 216.7704, 73.5567
    bw = a * n + b * np.exp(-((n - c)**2) / (2 * d**2))
    return max(1, int(round(bw)))

def run_buffer_optimization_study(csv_file="buffer_evaluation_results.csv"):
    print("Starting Buffer Optimization Study (Hidden Clique)")
    
    n_sizes = [300, 400, 500, 600]
    buffer_levels = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    num_instances = 30
    
    # Initialize or load CSV
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
    else:
        df_existing = pd.DataFrame(columns=[
            "n", "instance_id", "base_bw", "buffer_pct", "test_bw", 
            "optimal", "fl_dp_obj", "fl_final_obj", "fl_time_dp", "fl_time_ls", "fl_time_total", 
            "fl_ls_iters", "fl_unique_optima", "success"
        ])
        df_existing.to_csv(csv_file, index=False)
        
    for buffer_pct in buffer_levels:
        print(f"Testing buffer level: +{int(buffer_pct*100)}%")
        
        for n in n_sizes:
            if buffer_pct == 0.0 and n != 600:
                print(f"\n  [SKIPPING] N={n}, Buffer=0% - Already covered in main benchmark.")
                continue
                
            base_bw = get_thesis_bw(n)
            test_bw = int(round(base_bw * (1.0 + buffer_pct)))
            print(f"Testing N={n} | Beam Width: {test_bw} (Base: {base_bw})")
            
            for inst_id in range(num_instances):
                # 1. Skip Check (Resumability)
                df_current = pd.read_csv(csv_file)
                if not df_current[(df_current['n'] == n) & 
                                  (df_current['instance_id'] == inst_id) & 
                                  (df_current['buffer_pct'] == buffer_pct)].empty:
                    print(f"  [SKIPPING] N={n}, Inst={inst_id}, Buffer={int(buffer_pct*100)}% - Already in CSV.")
                    continue
                    
                # 2. Generate exact instance
                seed = inst_id + n * 1000
                _, cap, Q, w, clique, optimal = generate_hidden_clique(n, seed=seed)
                
                # 3. Run algorithm
                print(f"  Running Inst {inst_id:2d}... ", end="", flush=True)
                
                (best_set, fl_final, fl_weight, fl_dp, avg_gain, total_gain, 
                 avg_ls_iters, uniq_sols, fl_t_dp, fl_t_ls) = fl_heuristic_enhanced(
                    n, cap, Q, w, use_dynamic=True, beam_width=test_bw, multi_path_ls=True, use_diversity=True
                )
                
                fl_t_tot = fl_t_dp + fl_t_ls
                success = 1 if fl_final == optimal else 0
                gap = optimal - fl_final
                
                if success:
                    print(f"Success! (DP: {fl_t_dp:.1f}s, LS: {fl_t_ls:.1f}s, Total: {fl_t_tot:.1f}s)")
                else:
                    print(f"FAILED (Gap: {gap}, Total: {fl_t_tot:.1f}s)")
                    
                # 4. Save incrementally
                new_row = pd.DataFrame([{
                    "n": n,
                    "instance_id": inst_id,
                    "base_bw": base_bw,
                    "buffer_pct": buffer_pct,
                    "test_bw": test_bw,
                    "optimal": optimal,
                    "fl_dp_obj": fl_dp,
                    "fl_final_obj": fl_final,
                    "fl_time_dp": fl_t_dp,
                    "fl_time_ls": fl_t_ls,
                    "fl_time_total": fl_t_tot,
                    "fl_ls_iters": avg_ls_iters,
                    "fl_unique_optima": uniq_sols,
                    "success": success
                }])
                new_row.to_csv(csv_file, mode='a', header=False, index=False)

def run_validation_study(csv_file="buffer_validation_results.csv", n_sizes=None, buffer_levels=None, algo="diverse"):
    """Validates buffer levels on large out-of-sample sizes. Configurable from the runner."""
    print(f"Starting Buffer Validation Study | Algorithm: {algo}")

    if n_sizes is None:
        n_sizes = [700, 800, 900, 1000]
    if buffer_levels is None:
        buffer_levels = [0.0, 0.05]  # Baseline (0%) and chosen optimal (5%)
    num_instances = 30

    # Initialize or load CSV
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
    else:
        df_existing = pd.DataFrame(columns=[
            "n", "instance_id", "base_bw", "buffer_pct", "test_bw",
            "optimal", "fl_dp_obj", "fl_final_obj", "fl_time_dp", "fl_time_ls", "fl_time_total",
            "fl_ls_iters", "fl_unique_optima", "success"
        ])
        df_existing.to_csv(csv_file, index=False)

    for buffer_pct in buffer_levels:
        print(f"Validating buffer level: +{int(buffer_pct*100)}% | Algo: {algo}")

        for n in n_sizes:
            base_bw = get_thesis_bw(n, algo=algo)
            test_bw = int(round(base_bw * (1.0 + buffer_pct)))
            print(f"Validating N={n} | Beam Width: {test_bw} (Base: {base_bw})")

            for inst_id in range(num_instances):
                # 1. Skip Check (Resumability)
                df_current = pd.read_csv(csv_file)
                if not df_current[(df_current['n'] == n) &
                                  (df_current['instance_id'] == inst_id) &
                                  (df_current['buffer_pct'] == buffer_pct)].empty:
                    print(f"  [SKIPPING] N={n}, Inst={inst_id}, Buffer={int(buffer_pct*100)}% - Already in CSV.")
                    continue

                # 2. Generate exact instance
                seed = inst_id + n * 1000
                _, cap, Q, w, clique, optimal = generate_hidden_clique(n, seed=seed)

                use_agg = (algo == "aggregated_anchor")
                agg_pct = 0.15 if use_agg else 0.0

                (best_set, fl_final, fl_weight, fl_dp, avg_gain, total_gain,
                 avg_ls_iters, uniq_sols, fl_t_dp, fl_t_ls) = fl_heuristic_enhanced(
                    n, cap, Q, w, use_dynamic=True, beam_width=test_bw, multi_path_ls=True, use_diversity=True, use_aggregated_anchor=use_agg, aggregated_anchor_pct=agg_pct
                )

                fl_t_tot = fl_t_dp + fl_t_ls
                success = 1 if fl_final == optimal else 0
                gap = optimal - fl_final

                if success:
                    print(f"Done. (DP: {fl_t_dp:.1f}s, LS: {fl_t_ls:.1f}s, Total: {fl_t_tot:.1f}s)")
                else:
                    print(f"Failed. (Gap: {gap}, Total: {fl_t_tot:.1f}s)")

                # 4. Save incrementally
                new_row = pd.DataFrame([{
                    "n": n,
                    "instance_id": inst_id,
                    "base_bw": base_bw,
                    "buffer_pct": buffer_pct,
                    "test_bw": test_bw,
                    "optimal": optimal,
                    "fl_dp_obj": fl_dp,
                    "fl_final_obj": fl_final,
                    "fl_time_dp": fl_t_dp,
                    "fl_time_ls": fl_t_ls,
                    "fl_time_total": fl_t_tot,
                    "fl_ls_iters": avg_ls_iters,
                    "fl_unique_optima": uniq_sols,
                    "success": success
                }])
                new_row.to_csv(csv_file, mode='a', header=False, index=False)

def run_aggregated_anchor_study(csv_file="aggregated_anchor_evaluation_results.csv", aggregated_anchor_pct=0.15, n_sizes=None, num_instances=30):
    """Benchmarks the Aggregated Anchor steering on the baseline configuration."""
    print(f"Starting Aggregated Anchor Study (Hidden Clique) | Pct: {aggregated_anchor_pct*100}%")
    
    if n_sizes is None:
        n_sizes = [10, 25, 40, 50, 60, 75, 85, 90, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600]
    
    # Initialize or load CSV
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
    else:
        df_existing = pd.DataFrame(columns=[
            "n", "instance_id", "base_bw", "beam_width", "aggregated_anchor_pct",
            "optimal", "fl_dp_obj", "fl_final_obj", "fl_time_dp", "fl_time_ls", "fl_time_total", 
            "fl_ls_iters", "fl_unique_optima", "success"
        ])
        df_existing.to_csv(csv_file, index=False)
        
    for n in n_sizes:
        base_bw = get_thesis_bw(n)
        test_bw = max(1, int(round(base_bw * 1.05)))
        print(f"Testing N={n} | Base BW: {base_bw} | Buffered BW: {test_bw} | Aggregated Anchor Pct: {aggregated_anchor_pct*100}%")
        
        for inst_id in range(num_instances):
            # 1. Skip Check (Resumability)
            df_current = pd.read_csv(csv_file)
            if not df_current[(df_current['n'] == n) & 
                              (df_current['instance_id'] == inst_id) &
                              (df_current['beam_width'] == test_bw) &
                              (df_current['aggregated_anchor_pct'] == aggregated_anchor_pct)].empty:
                print(f"  [SKIPPING] N={n}, Inst={inst_id}, B={test_bw} - Already in CSV.")
                continue
                
            # 2. Generate exact instance
            seed = inst_id + n * 1000
            _, cap, Q, w, clique, optimal = generate_hidden_clique(n, seed=seed)
            
            # 3. Run algorithm
            print(f"  Running Inst {inst_id:2d}... ", end="", flush=True)
            
            (best_set, fl_final, fl_weight, fl_dp, avg_gain, total_gain, 
             avg_ls_iters, uniq_sols, fl_t_dp, fl_t_ls) = fl_heuristic_enhanced(
                n, cap, Q, w, use_dynamic=True, beam_width=test_bw, multi_path_ls=True, use_diversity=True, 
                use_aggregated_anchor=True, aggregated_anchor_pct=aggregated_anchor_pct
            )
            
            fl_t_tot = fl_t_dp + fl_t_ls
            success = 1 if fl_final == optimal else 0
            gap = optimal - fl_final
            
            if success:
                print(f"Done. (DP: {fl_t_dp:.1f}s, LS: {fl_t_ls:.1f}s, Total: {fl_t_tot:.1f}s)")
            else:
                print(f"Failed. (Gap: {gap}, Total: {fl_t_tot:.1f}s)")
                
            # 4. Save incrementally
            new_row = pd.DataFrame([{
                "n": n,
                "instance_id": inst_id,
                "base_bw": base_bw,
                "beam_width": test_bw,
                "aggregated_anchor_pct": aggregated_anchor_pct,
                "optimal": optimal,
                "fl_dp_obj": fl_dp,
                "fl_final_obj": fl_final,
                "fl_time_dp": fl_t_dp,
                "fl_time_ls": fl_t_ls,
                "fl_time_total": fl_t_tot,
                "fl_ls_iters": avg_ls_iters,
                "fl_unique_optima": uniq_sols,
                "success": success
            }])
            new_row.to_csv(csv_file, mode='a', header=False, index=False)
if __name__ == "__main__":
    # Run the default buffer optimization study if executed directly
    run_buffer_optimization_study()
