import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

plt.style.use('seaborn-v0_8-whitegrid')

def get_relative_convergence_b(df_subset, threshold=0.999):
    """
    Finds the smallest beam width that achieves at least 'threshold' (e.g. 99.9%) 
    of the maximum objective value found across all tested beam widths.
    """
    df_subset = df_subset.sort_values('beam_width')
    beams = df_subset['beam_width'].values
    objs = df_subset['avg_objective'].values
    
    max_obj = np.max(objs)
    target_obj = max_obj * threshold
    
    for i in range(len(beams)):
        if objs[i] >= target_obj:
            return beams[i]
    return beams[-1]

def analyze_and_plot(csv_file, name):
    if not os.path.exists(csv_file):
        print(f"Skipping {name}, {csv_file} not found.")
        return
        
    df = pd.read_csv(csv_file)
    all_ns = sorted(df['n'].unique())
    
    n_vals = []
    b_vals = []
    
    # We will average the B* across all densities for a given N
    # Alternatively, we could plot a separate point per density
    for n in all_ns:
        df_n = df[df['n'] == n]
        densities = df_n['density'].unique()
        b_stars_for_n = []
        for d in densities:
            df_nd = df_n[df_n['density'] == d]
            b_star = get_relative_convergence_b(df_nd, threshold=0.999)
            b_stars_for_n.append(b_star)
        
        # Take the maximum required beam width across all densities to be safe,
        # or the average. Let's use max for conservative heuristic design.
        n_vals.append(n)
        b_vals.append(np.max(b_stars_for_n))
        
    X = np.array(n_vals, dtype=float)
    Y = np.array(b_vals, dtype=float)
    
    print(f"\n--- {name} Relative Convergence Points ---")
    for n, b in zip(X, Y):
        print(f"  N={n:3.0f} | B*={b:3.0f} (Max across densities)")



    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(X, Y, color='black', label=f'{name} B* (Relative Conv. 99.9%)', s=50, zorder=5)
    


    plt.title(f'{name}: Optimal Beam Width vs N (Relative Convergence)', fontsize=14)
    plt.xlabel('Instance Size (N)', fontsize=12)
    plt.ylabel('Required Beam Width (B*)', fontsize=12)
    plt.legend()
    plt.ylim(ymin=0)
    plt.savefig(f'plots/{name.lower()}_fit.png', dpi=300)
    print(f"Saved plot to plots/{name.lower()}_fit.png")

analyze_and_plot('data/gallo_mountain_curve.csv', 'Gallo')
analyze_and_plot('data/dks_mountain_curve.csv', 'DKS')
