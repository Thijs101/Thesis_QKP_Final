import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_local_search_impact_plot():
    print("Generating Local Search Impact Plot...")
    
    # 1. Load data
    csv_file = 'data/local_search_impact_summary.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found in the current directory.")
        return
        
    df = pd.read_csv(csv_file)
    print(f"Loaded local search impact data from '{csv_file}'.")

    # Filter for Hidden Clique instances
    df_hc = df[df['inst_type'] == 'Hidden_Clique'].copy()
    if df_hc.empty:
        print("Error: No Hidden Clique data found in the impact summary CSV.")
        return
        
    # Sort by size N to ensure lines are plotted chronologically/in order of size
    df_hc = df_hc.sort_values('n')

    # Academic Plot Styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # X-axis sizes and corresponding Y-axis values
    sizes = df_hc['n'].values
    our_gains = df_hc['our_avg_gain_pct'].values
    fn_gains = df_hc['fn_avg_gain_pct'].values

    print("Hidden Clique data points:")
    for sz, o_g, f_g in zip(sizes, our_gains, fn_gains):
        print(f"  N={sz:3d} -> Our Gain: {o_g:5.2f}%, Fennich Gain: {f_g:5.2f}%")

    # 3. Add Shaded Region and Vertical Dashed Lines for Difficulty Peak Zone first
    # This places them in the background (zorder=1)
    ax.axvspan(150, 250, color='#7f8c8d', alpha=0.12, label='Difficulty Peak Zone ($N \\in [150, 250]$)', zorder=1)
    ax.axvline(150, color='#7f8c8d', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)
    ax.axvline(250, color='#7f8c8d', linestyle='--', linewidth=1.2, alpha=0.7, zorder=1)

    # Line 1: Combined Approach
    ax.plot(
        sizes, 
        our_gains, 
        label='Combined Approach', 
        color='#3498db', 
        marker='D', 
        markersize=7, 
        linewidth=2,
        markerfacecolor='#3498db',
        markeredgecolor='white',
        markeredgewidth=0.8,
        zorder=3
    )

    # Line 2: Fennich Baseline
    ax.plot(
        sizes, 
        fn_gains, 
        label='Fennich Baseline', 
        color='#333333', 
        marker='x', 
        markersize=8, 
        linewidth=2,
        markeredgewidth=1.8,
        zorder=3
    )

    # Set Title and Labels
    ax.set_title('Local Search Gain vs Instance Size (Hidden Clique)', fontweight='bold', pad=15)
    ax.set_xlabel('Instance Size ($N$)', fontweight='bold', labelpad=8)
    ax.set_ylabel('Average Local Search Gain (%)', fontweight='bold', labelpad=8)

    # Configure axes limits and ticks
    ax.set_xlim(left=sizes.min() - 10, right=sizes.max() + 10)
    ax.set_ylim(bottom=0, top=12) # Starts strictly at 0, top leaves margin above 10.43%
    
    # Set explicit representative x-ticks
    x_ticks = [10, 50, 100, 150, 200, 250, 300, 400, 500, 600]
    ax.set_xticks(x_ticks)
    
    # Format y-ticks to show percentage symbol or clean integers
    ax.set_yticks(np.arange(0, 13, 2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

    # Grid and Spines styling
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    # Legend styling
    ax.legend(loc='upper left', frameon=True, edgecolor='#cccccc', facecolor='white', framealpha=0.9)

    # Adjust layout
    plt.tight_layout()
    
    # Save the output plot
    output_filename = 'plots/HC_Local_Search_Impact.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to '{output_filename}'.")

if __name__ == '__main__':
    generate_local_search_impact_plot()
