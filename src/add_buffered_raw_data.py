import os
import pandas as pd
import numpy as np
import sys

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dynamic_Programming_QKP import generate_hidden_clique, fl_heuristic_enhanced

def get_thesis_bw(n):
    a, b, c, d = 0.2102, 201.3693, 216.7704, 73.5567
    bw = a * n + b * np.exp(-((n - c)**2) / (2 * d**2))
    return max(1, int(round(bw)))

def add_buffered_data(raw_file="data/thesis_raw_data.csv", buffer_file="data/buffer_evaluation_results.csv"):
    """
    Appends new rows with buffered beam width to the raw data csv.
    """
    if not os.path.exists(raw_file):
        print(f"[Error] {raw_file} does not exist!")
        return

    print(f"Adding 5% Buffered Data to {raw_file}")

    df_raw = pd.read_csv(raw_file)
    df_buffer = pd.read_csv(buffer_file) if os.path.exists(buffer_file) else None

    # Find all unique HC configurations
    hc = df_raw[df_raw['inst_type'] == 'Hidden_Clique']
    n_sizes = sorted(hc['n'].unique())

    # Build a set of already-done (n, instance_id, beam_width) combos
    done_set = set()
    for _, row in df_raw[df_raw['inst_type'] == 'Hidden_Clique'].iterrows():
        done_set.add((int(row['n']), int(row['instance_id']), int(row['beam_width'])))

    borrowed_count = 0
    run_count = 0
    skipped_count = 0

    for n in n_sizes:
        base_bw = get_thesis_bw(n)
        buffered_bw = max(1, int(round(base_bw * 1.05)))

        if base_bw == buffered_bw:
            print(f"  [SKIP] N={n:3d} | Base BW == Buffered BW ({base_bw}), no new data needed.")
            continue

        # Get instance IDs from the existing unbuffered data
        existing = hc[hc['n'] == n]
        instance_ids = sorted(existing['instance_id'].unique())

        print(f"\n--- N={n:3d} | Base BW: {base_bw} | Buffered BW: {buffered_bw} | Instances: {len(instance_ids)} ---")

        for inst_id in instance_ids:
            # Skip if already done
            if (n, inst_id, buffered_bw) in done_set:
                skipped_count += 1
                continue

            # Try to borrow from buffer_evaluation_results.csv
            borrowed = False
            if df_buffer is not None:
                mb = df_buffer[
                    (df_buffer['n'] == n) &
                    (df_buffer['instance_id'] == inst_id) &
                    (df_buffer['test_bw'] == buffered_bw)
                ]
                if not mb.empty:
                    brow = mb.iloc[0]
                    # We need the Fennich results too — get them from the existing unbuffered row
                    orig = existing[existing['instance_id'] == inst_id].iloc[0]

                    new_row = {
                        'inst_type': 'Hidden_Clique', 'n': n, 'density': 0.0,
                        'instance_id': inst_id, 'beam_width': buffered_bw,
                        'optimal': brow['optimal'],
                        'fn_dp_obj': orig['fn_dp_obj'], 'fn_final_obj': orig['fn_final_obj'],
                        'fn_time_dp': orig['fn_time_dp'], 'fn_time_ls': orig['fn_time_ls'],
                        'fn_time_total': orig['fn_time_total'], 'fn_ls_iters': orig['fn_ls_iters'],
                        'fl_dp_obj': brow['fl_dp_obj'], 'fl_final_obj': brow['fl_final_obj'],
                        'fl_time_dp': brow['fl_time_dp'], 'fl_time_ls': brow['fl_time_ls'],
                        'fl_time_total': brow['fl_time_total'],
                        'fl_ls_iters': brow['fl_ls_iters'], 'fl_unique_optima': brow['fl_unique_optima']
                    }
                    pd.DataFrame([new_row]).to_csv(raw_file, mode='a', header=False, index=False)
                    done_set.add((n, inst_id, buffered_bw))
                    borrowed = True
                    borrowed_count += 1
                    print(f"  [Borrowed] N={n:3d}, Inst={inst_id:2d} | BW={buffered_bw}")
                    continue

            if not borrowed:
                # Generate instance and run
                seed = inst_id + n * 1000
                _, cap, Q, w, clique, optimal = generate_hidden_clique(n, seed=seed)

                # Get Fennich results from existing row (same for any BW)
                orig = existing[existing['instance_id'] == inst_id].iloc[0]

                # Run our heuristic with buffered beam width
                (best_set, fl_final, fl_weight, fl_dp, avg_gain, total_gain,
                 avg_ls_iters, uniq_sols, fl_t_dp, fl_t_ls) = fl_heuristic_enhanced(
                    n, cap, Q, w, use_dynamic=True, beam_width=buffered_bw,
                    multi_path_ls=True, use_diversity=True
                )

                new_row = {
                    'inst_type': 'Hidden_Clique', 'n': n, 'density': 0.0,
                    'instance_id': inst_id, 'beam_width': buffered_bw,
                    'optimal': optimal,
                    'fn_dp_obj': orig['fn_dp_obj'], 'fn_final_obj': orig['fn_final_obj'],
                    'fn_time_dp': orig['fn_time_dp'], 'fn_time_ls': orig['fn_time_ls'],
                    'fn_time_total': orig['fn_time_total'], 'fn_ls_iters': orig['fn_ls_iters'],
                    'fl_dp_obj': fl_dp, 'fl_final_obj': fl_final,
                    'fl_time_dp': fl_t_dp, 'fl_time_ls': fl_t_ls,
                    'fl_time_total': fl_t_dp + fl_t_ls,
                    'fl_ls_iters': avg_ls_iters, 'fl_unique_optima': uniq_sols
                }
                pd.DataFrame([new_row]).to_csv(raw_file, mode='a', header=False, index=False)
                done_set.add((n, inst_id, buffered_bw))
                run_count += 1
                print(f"  [Computed] N={n:3d}, Inst={inst_id:2d} | BW={buffered_bw} | Val={fl_final}")

    print(f"Done. Borrowed: {borrowed_count} | Computed: {run_count} | Skipped: {skipped_count}")

if __name__ == "__main__":
    add_buffered_data()
