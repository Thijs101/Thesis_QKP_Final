import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib

# --- Academic Plot Styling ---
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'text.usetex': False,
    'figure.dpi': 150,
})

def plot_mountain_curve(csv_file, title, output_name, densities_to_plot=None):
    """
    Generates mountain curves showing improvement over the baseline (B=1) 
    as the beam width increases.
    """
    if not os.path.exists(csv_file):
        print(f"Skipping {csv_file} (file not found)")
        return

    df = pd.read_csv(csv_file)
    
    if densities_to_plot is None:
        densities_to_plot = sorted(df['density'].unique())

    for d in densities_to_plot:
        df_d = df[df['density'] == d]
        if df_d.empty: continue

        plt.figure(figsize=(10, 7))
        all_n = sorted(df_d['n'].unique())
        
        # Use a sequential colormap (viridis) to show increasing N
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_n)))

        for n, color in zip(all_n, colors):
            subset = df_d[df_d['n'] == n].sort_values('beam_width')
            if subset.empty: continue
            
            # Use Greedy (B=1) as the baseline for 0% improvement
            greedy_val = subset[subset['beam_width'] == 1]['avg_objective'].values
            if len(greedy_val) == 0: continue
            v0 = greedy_val[0]
            
            # Calculate relative improvement (%)
            improvement = 100 * (subset['avg_objective'] - v0) / v0
            
            # Plot the heuristic mountain curve
            plt.plot(subset['beam_width'], improvement, 
                     marker='o', label=f'N = {n}', color=color, linewidth=2, markersize=6)

        plt.title(f'{title} (Density = {d})', fontweight='bold', pad=15)
        plt.xlabel('Beam Width ($B$)', fontweight='bold')
        plt.ylabel('Improvement over Baseline ($B=1$) (%)', fontweight='bold')
        plt.xscale('log')
        
        # Format the X-axis to show actual numbers instead of scientific notation
        plt.xticks([1, 2, 3, 5, 10, 20, 50, 100, 200], ['1', '2', '3', '5', '10', '20', '50', '100', '200'])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        
        # Keep only the major grid for a cleaner look
        plt.grid(True, which='major', linestyle='-', alpha=0.2, color='gray')
        
        # Style borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Place legend safely
        plt.legend(title='Instance Size ($N$)', loc='best', framealpha=0.95, edgecolor='#cccccc')
        
        fname = f"{output_name}_D{int(d*100)}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {fname}")


def main():
    # 1. DKS Mountain Curves
    plot_mountain_curve('data/dks_mountain_curve.csv', 
                       'Mountain Curve - DKS Instances', 'plots/Plot_DKS')
    
    # 2. Gallo Mountain Curves
    plot_mountain_curve('data/gallo_mountain_curve.csv', 
                       'Mountain Curve - Gallo Instances', 'plots/Plot_Gallo')
    

if __name__ == "__main__":
    main()
