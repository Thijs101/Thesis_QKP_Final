import os
import pandas as pd
import numpy as np
import time
from Dynamic_Programming_QKP import (
    generate_benchmark_instance, 
    generate_gallo_instance, 
    generate_hidden_clique, 
    fl_heuristic_enhanced
)

def get_beam_width(n, inst_type):
    if inst_type == "Gallo" or inst_type == "DKS": return 5
    if inst_type == "Hidden_Clique":
        a, b, c, d = 0.2102, 201.3693, 216.7704, 73.5567
        b_star = a * n + b * np.exp(-((n - c)**2) / (2 * d**2))
        base_bw = max(1, int(round(b_star)))
        return max(1, int(round(base_bw * 1.05)))  # With 5% Safety Buffer!
    return 10

def run_ablation_variant(name, n_val, density, beam_width, use_dynamic, use_diversity, multi_path_ls, inst_type, num_instances=20, v_num=1):
    """Runs a specific heuristic variant."""
    print(f"  [{v_num}/6] Running Variant: {name} (N={n_val}, B={beam_width}, D={density})")
    vals, times, gaps = [], [], []
    
    for i in range(num_instances):
        if inst_type == "Hidden_Clique": 
            seed = i + n_val * 1000
            n, cap, Q, w, clique, optimal = generate_hidden_clique(n_val, seed=seed)
        else: 
            seed = i + n_val * 1000 + int(density * 100)
            if inst_type == "DKS": n, cap, Q, w, k = generate_benchmark_instance(n_val, density, seed=seed)
            elif inst_type == "Gallo": n, cap, Q, w = generate_gallo_instance(n_val, density, seed=seed)
            
            # Retrieve optimal from cache if available
            optimal = None
            cache_file = "data/dks_gurobi_cache.json" if inst_type == "DKS" else "data/gallo_gurobi_cache.json"
            cache_key = f"{n_val}_{density}_{seed}"
            density_str_alt = str(int(density)) if density.is_integer() else str(density)
            cache_key_alt = f"{n_val}_{density_str_alt}_{seed}"
            
            if os.path.exists(cache_file):
                try:
                    import json
                    with open(cache_file, 'r') as f:
                        cache = json.load(f)
                    if cache_key in cache and 'gurobi_val' in cache[cache_key]:
                        optimal = cache[cache_key]['gurobi_val']
                    elif cache_key_alt in cache and 'gurobi_val' in cache[cache_key_alt]:
                        optimal = cache[cache_key_alt]['gurobi_val']
                except Exception:
                    pass
            
        start = time.process_time()
        _, val, *_ = fl_heuristic_enhanced(n, cap, Q, w, use_dynamic=use_dynamic, beam_width=beam_width, multi_path_ls=multi_path_ls, use_diversity=use_diversity)
        elapsed = time.process_time() - start
        
        vals.append(val); times.append(elapsed)
        if optimal is not None: gaps.append(100 * (optimal - val) / optimal if optimal > 0 else 0)
        print(f"    Instance {i+1:2d}/{num_instances} | Val: {val:g} | Time: {elapsed:.2f}s", end='\r', flush=False)
    print()
    
    g_arr = np.array(gaps) if gaps else np.array([])
    return {
        'variant': name, 'inst_type': inst_type, 'n': n_val, 'density': density, 'beam_width': beam_width,
        'avg_obj': np.mean(vals), 'avg_time': np.mean(times), 
        'avg_gap': np.mean(gaps) if len(gaps) > 0 else None, 'std_gap': np.std(gaps) if len(gaps) > 0 else None,
        'max_gap': np.max(gaps) if len(gaps) > 0 else None,
        'success_rate': (np.sum(g_arr == 0) / len(g_arr) * 100) if len(g_arr) > 0 else None
    }

def run_all_ablations(inst_types=None, n_sizes=None, num_instances=20, densities=None, csv_file="data/ablation_study_results.csv", raw_source="data/thesis_raw_data.csv"):
    if inst_types is None: inst_types = ["DKS", "Gallo", "Hidden_Clique"]
    if n_sizes is None: n_sizes = [50, 100, 200]
    if densities is None: densities = [0.5]
    
    results = []
    if os.path.exists(csv_file): results = pd.read_csv(csv_file).to_dict('records')
    done_configs = {(r['inst_type'], int(r['n']), float(r['density']), r['variant'], int(r['beam_width'])) for r in results}

    df_raw = pd.read_csv(raw_source) if os.path.exists(raw_source) else None
    df_buffer = pd.read_csv("buffer_evaluation_results.csv") if os.path.exists("buffer_evaluation_results.csv") else None

    for inst in inst_types:
        current_densities = densities if inst != "Hidden_Clique" else [0.0]
        for d in current_densities:
            for n in n_sizes:
                print(f"\n--- Ablation: {inst} (N={n}, D={d}) ---")
                
                b = get_beam_width(n, inst)
                
                # 1. Borrow Fennich from Raw Source
                if df_raw is not None and (inst, n, d, 'Fennich_Baseline', 0) not in done_configs:
                    m = df_raw[(df_raw['inst_type']==inst) & (df_raw['n']==n) & (df_raw['density']==d)]
                    if not m.empty:
                        print("  [Baseline] Borrowing Fennich from Step 3.")
                        if not m['optimal'].isna().any() and (m['optimal'] > 0).all():
                            g = 100 * (m['optimal'] - m['fn_final_obj']) / m['optimal']
                        else:
                            g = None
                        results.append({'variant': 'Fennich_Baseline', 'inst_type': inst, 'n': n, 'density': d, 'beam_width': 0,
                                      'avg_obj': m['fn_final_obj'].mean(), 'avg_time': m['fn_time_total'].mean(), 
                                      'avg_gap': g.mean() if g is not None else None, 
                                      'std_gap': g.std() if g is not None else None,
                                      'max_gap': g.max() if g is not None else None,
                                      'success_rate': (sum(g == 0) / len(m) * 100) if g is not None else None})
                        # Add immediately to done_configs to avoid duplicates in subsequent variants check
                        done_configs.add((inst, n, d, 'Fennich_Baseline', 0))

                # 2. Borrow Full Proposed from Raw Source or Buffer Study (only if beam width matches)
                borrowed_v4 = False
                if (inst, n, d, 'V4_Full_Proposed', b) not in done_configs:
                    m = df_raw[(df_raw['inst_type']==inst) & (df_raw['n']==n) & (df_raw['density']==d)] if df_raw is not None else pd.DataFrame()
                    if not m.empty:
                        m_bw = m[m['beam_width'] == b]
                        if not m_bw.empty and len(m_bw) >= num_instances:
                            m = m_bw
                            print("  [V4] Borrowing Proposed from Step 3.")
                            if not m['optimal'].isna().any() and (m['optimal'] > 0).all():
                                g = 100 * (m['optimal'] - m['fl_final_obj']) / m['optimal']
                            else:
                                g = None
                            results.append({'variant': 'V4_Full_Proposed', 'inst_type': inst, 'n': n, 'density': d, 'beam_width': m['beam_width'].iloc[0],
                                          'avg_obj': m['fl_final_obj'].mean(), 'avg_time': m['fl_time_total'].mean(), 
                                          'avg_gap': g.mean() if g is not None else None, 
                                          'std_gap': g.std() if g is not None else None,
                                          'max_gap': g.max() if g is not None else None,
                                          'success_rate': (sum(g == 0) / len(m) * 100) if g is not None else None})
                            done_configs.add((inst, n, d, 'V4_Full_Proposed', b))
                            borrowed_v4 = True
                    elif df_buffer is not None and inst == "Hidden_Clique":
                        # Check if buffer_evaluation_results.csv contains this size with matching beam width
                        mb = df_buffer[(df_buffer['n'] == n) & (df_buffer['test_bw'] == b)]
                        if not mb.empty and len(mb) >= num_instances:
                            print(f"  [V4] Borrowing Proposed from Buffer Optimization Study ({int(round(mb['buffer_pct'].iloc[0]*100))}% buffer).")
                            g = 100*(mb['optimal']-mb['fl_final_obj'])/mb['optimal']
                            results.append({'variant': 'V4_Full_Proposed', 'inst_type': inst, 'n': n, 'density': d, 'beam_width': b,
                                          'avg_obj': mb['fl_final_obj'].mean(), 'avg_time': mb['fl_time_total'].mean(), 
                                          'avg_gap': g.mean(), 
                                          'std_gap': g.std(),
                                          'max_gap': g.max(),
                                          'success_rate': (sum(g == 0) / len(mb) * 100)})
                            done_configs.add((inst, n, d, 'V4_Full_Proposed', b))
                            borrowed_v4 = True

                # 3. Run missing variants
                variants = [
                    ("V1_Static_Greedy", 1, False, False, False, 1), 
                    ("V2_Dynamic_Greedy", 1, True, False, False, 2), 
                    ("V3_Diverse_Beam_Static", b, False, True, True, 3),
                    ("V3b_Standard_Beam", b, True, False, True, 4),
                    ("V3c_Standard_Beam_Static", b, False, False, True, 5)
                ]
                
                # Check if V4 needs to be run dynamically
                has_v4 = borrowed_v4 or (inst, n, d, 'V4_Full_Proposed', b) in done_configs
                if not has_v4:
                    variants.append(("V4_Full_Proposed", b, True, True, True, 6))
                    
                for name, bw, dyn, div, ls, v_num in variants:
                    if (inst, n, d, name, bw) not in done_configs:
                        res = run_ablation_variant(name, n, d, bw, dyn, div, ls, inst, num_instances, v_num=v_num)
                        results.append(res)
                        done_configs.add((inst, n, d, name, bw))
                        # Sort results consistently by variant order
                        df_res = pd.DataFrame(results)
                        v_order = {
                            'Fennich_Baseline': 0, 'V1_Static_Greedy': 1, 'V2_Dynamic_Greedy': 2,
                            'V3_Diverse_Beam_Static': 3, 'V3b_Standard_Beam': 4, 
                            'V3c_Standard_Beam_Static': 5, 'V4_Full_Proposed': 6
                        }
                        df_res['sort_key'] = df_res['variant'].map(v_order).fillna(99)
                        df_res = df_res.sort_values(by=['inst_type', 'n', 'density', 'sort_key']).drop(columns=['sort_key'])
                        df_res.to_csv(csv_file, index=False)

if __name__ == "__main__":
    run_all_ablations()
