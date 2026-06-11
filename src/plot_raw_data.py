import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_plots(csv_file, output_filename, title_prefix):
    if not os.path.exists(csv_file):
        print(f"Error: Cannot find {csv_file}")
        return

    df = pd.read_csv(csv_file)
    
    all_n = sorted(df['n'].unique())
    representative_ns = [n for n in all_n if n in [25, 50, 100, 150, 200, 300, 400, 500, 600]]
    if not representative_ns:
        representative_ns = all_n # fallback

    # Filter dataframe
    df_plot = df[df['n'].isin(representative_ns)]

    fig, ax1 = plt.subplots(figsize=(10, 7))
    fig.suptitle(f'Empirical Performance Analysis of {title_prefix}', fontsize=18, fontweight='bold', y=1.02)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(representative_ns)))

    # Plot Gap vs Beam Width
    for n, color in zip(representative_ns, colors):
        df_n = df_plot[df_plot['n'] == n].sort_values('beam_width')
        ax1.plot(df_n['beam_width'], df_n['avg_gap (%)'], marker='o', markersize=5, linewidth=2, color=color, label=f'N = {n}')

    ax1.set_title('Optimality Gap vs. Beam Width', fontsize=14)
    ax1.set_xlabel('Beam Width ($B$)', fontsize=12)
    ax1.set_ylabel('Average Gap (%)', fontsize=12)
    ax1.set_xscale('log')
    
    # Custom x-ticks
    ax1.set_xticks([1, 5, 10, 25, 50, 100, 200, 300])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(title='Graph Size (N)', loc='upper right')

    # Final touches
    plt.tight_layout()
    output_filename = output_filename
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")

if __name__ == "__main__":
    generate_plots("data/volledige_onderzoek_results_diverse.csv", "plots/Overview_Diverse.png", "Diverse Beam Search")
    generate_plots("data/volledige_onderzoek_results_standard.csv", "plots/Overview_Standard.png", "Standard Beam Search")
