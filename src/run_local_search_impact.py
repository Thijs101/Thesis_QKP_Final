import os
import pandas as pd
import numpy as np

def run_ls_impact_summary(raw_source="data/thesis_raw_data.csv", csv_summary="data/local_search_impact_summary.csv"):
    """
    Pure Summarizer: Reads the unified raw data from Step 3 and produces the LS Impact report.
    No solvers are called here.
    """
    if not os.path.exists(raw_source):
        print(f"  [Error] {raw_source} not found. Run Step 3 first!")
        return

    print(f"Summarizing data from {raw_source}...")
    df = pd.read_csv(raw_source)
    
    # For Hidden Clique: only use the 5% buffered beam width rows
    import numpy as np_local
    def _buffered_bw(n):
        a, b, c, d = 0.2102, 201.3693, 216.7704, 73.5567
        base = a * n + b * np_local.exp(-((n - c)**2) / (2 * d**2))
        return max(1, int(round(max(1, int(round(base))) * 1.05)))
    
    hc_mask = df['inst_type'] == 'Hidden_Clique'
    if hc_mask.any():
        # Keep only rows where beam_width matches the 5% buffered value
        buffered_keep = df[hc_mask].apply(lambda r: int(r['beam_width']) == _buffered_bw(int(r['n'])), axis=1)
        df = pd.concat([df[~hc_mask], df[hc_mask][buffered_keep]])
        print(f"  [Filter] Kept {buffered_keep.sum()} Hidden Clique rows with 5% buffered beam width.")
    
    # Calculate Gaps and Gains for every instance
    df['fn_gain_pct'] = (df['fn_final_obj'] - df['fn_dp_obj']) / df['fn_dp_obj'] * 100
    df['fl_gain_pct'] = (df['fl_final_obj'] - df['fl_dp_obj']) / df['fl_dp_obj'] * 100
    
    # Hidden Clique Specifics
    df['fn_gap'] = 100 * (df['optimal'] - df['fn_final_obj']) / df['optimal']
    df['fl_gap'] = 100 * (df['optimal'] - df['fl_final_obj']) / df['optimal']
    
    summary_rows = []
    # Group by instance configuration
    groups = df.groupby(['inst_type', 'n', 'density', 'beam_width'])
    
    for (inst, n, d, bw), batch in groups:
        # Check if we have optimal values for this batch (some or all)
        has_optimal = batch['optimal'].notna().any()
        # Drop rows where optimal is NaN to compute gaps correctly
        valid_batch = batch.dropna(subset=['optimal'])
        num_valid = len(valid_batch)
        
        row = {
            'inst_type': inst, 'n': n, 'density': d, 'beam_width': bw,
            'num_instances': len(batch),
            # Proposed Metrics
            'our_avg_gain_pct': batch['fl_gain_pct'].mean(),
            'our_std_gain_pct': batch['fl_gain_pct'].std(),
            'our_avg_time_total': batch['fl_time_total'].mean(),
            'our_avg_unique_optima': batch['fl_unique_optima'].mean(),
            'our_avg_ls_iters': batch['fl_ls_iters'].mean(),
            'our_success_rate': (sum(valid_batch['fl_gap'] == 0) / num_valid * 100) if has_optimal else None,
            'our_avg_final_gap': valid_batch['fl_gap'].mean() if has_optimal else None,
            'our_max_gap': valid_batch['fl_gap'].max() if has_optimal else None,
            
            # Fennich Metrics
            'fn_avg_gain_pct': batch['fn_gain_pct'].mean(),
            'fn_std_gain_pct': batch['fn_gain_pct'].std(),
            'fn_avg_time_total': batch['fn_time_total'].mean(),
            'fn_avg_ls_iters': batch['fn_ls_iters'].mean(),
            'fn_success_rate': (sum(valid_batch['fn_gap'] == 0) / num_valid * 100) if has_optimal else None,
            'fn_avg_final_gap': valid_batch['fn_gap'].mean() if has_optimal else None,
            'fn_max_gap': valid_batch['fn_gap'].max() if has_optimal else None,
        }
        summary_rows.append(row)
        
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(csv_summary, index=False)
    print(f"  -> Impact Summary saved to {csv_summary}")

def run_ls_impact_main(*args, **kwargs):
    """
    Wrapper to satisfy import in run_thesis_benchmarks.py.
    Calls run_ls_impact_summary to process the raw data.
    """
    run_ls_impact_summary()

if __name__ == "__main__":
    run_ls_impact_summary()
