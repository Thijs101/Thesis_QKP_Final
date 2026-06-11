import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_all_ablation_plots():
    print("Generating Ablation Plots...")
    
    # 1. Load data
    csv_file = 'data/ablation_study_results.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found in the current directory.")
        return
        
    df = pd.read_csv(csv_file)
    print(f"Loaded ablation study results from '{csv_file}'.")

    # 2. Academic Plot Styling
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

    # Define clean labels and colors for the variants
    variants_to_plot = [
        'Fennich_Baseline', 
        'V1_Static_Greedy', 
        'V2_Dynamic_Greedy', 
        'V3_Diverse_Beam_Static', 
        'V4_Full_Proposed'
    ]

    display_labels = {
        'Fennich_Baseline': 'Fennich Baseline',
        'V1_Static_Greedy': 'F&L Baseline',
        'V2_Dynamic_Greedy': 'Dynamic Greedy (DG)',
        'V3_Diverse_Beam_Static': 'Diverse Beam Search (DBS)',
        'V4_Full_Proposed': 'Combined Approach'
    }

    colors = {
        'Fennich_Baseline': '#333333',     # Dark grey/black for baseline
        'V1_Static_Greedy': '#e67e22',     # Orange
        'V2_Dynamic_Greedy': '#f1c40f',    # Yellow/Gold
        'V3_Diverse_Beam_Static': '#2ecc71', # Green
        'V4_Full_Proposed': '#3498db'      # Royal Blue
    }

    markers = {
        'Fennich_Baseline': 'x',
        'V1_Static_Greedy': 'o',
        'V2_Dynamic_Greedy': '^',
        'V3_Diverse_Beam_Static': 's',
        'V4_Full_Proposed': 'D'
    }

    # PART 1: GALLO ABLATION PLOT
    print("\nGenerating Gallo Ablation Analysis plot...")
    df_gallo = df[df['inst_type'] == 'Gallo'].copy()
    
    if not df_gallo.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        # Filter out unused variants (like V3b, V3c) so they don't affect the best_avg_obj calculation
        df_gallo = df_gallo[df_gallo['variant'].isin(variants_to_plot)].copy()
        
        # Calculate heuristic gap relative to the best variant for each N and Density
        df_gallo['best_avg_obj'] = df_gallo.groupby(['n', 'density'])['avg_obj'].transform('max')
        df_gallo['heuristic_gap'] = 100 * (df_gallo['best_avg_obj'] - df_gallo['avg_obj']) / df_gallo['best_avg_obj']

        # Panel A: Average Gaps (%)
        ax_gap = axes[0]
        df_gap = df_gallo.copy()  # Use all N
        
        for var in variants_to_plot:
            sub = df_gap[df_gap['variant'] == var].sort_values('n')
            if sub.empty: continue
            
            # Group by n to take the mean across densities
            grouped = sub.groupby('n')['heuristic_gap'].mean().reset_index()
            
            # For Fennich, only plot up to N=75 (since it timed out beyond that)
            if var == 'Fennich_Baseline':
                grouped = grouped[grouped['n'] <= 75]
            
            ax_gap.plot(
                grouped['n'], 
                grouped['heuristic_gap'], 
                label=display_labels[var], 
                color=colors[var], 
                marker=markers[var], 
                markersize=7, 
                linewidth=2
            )

        ax_gap.set_title('A: Gap to Best Heuristic (%) vs. Size', fontweight='bold', pad=12)
        ax_gap.set_xlabel('Instance Size ($N$)', fontweight='bold')
        ax_gap.set_ylabel('Average Gap (%)', fontweight='bold')
        ax_gap.set_xticks([10, 25, 50, 75, 100, 150])
        ax_gap.grid(True, linestyle='--', alpha=0.5)
        ax_gap.spines['top'].set_visible(False)
        ax_gap.spines['right'].set_visible(False)

        # Panel B: Runtimes (Log scale)
        ax_time = axes[1]
        df_time = df_gallo.copy()
        
        for var in variants_to_plot:
            sub = df_time[df_time['variant'] == var].sort_values('n')
            if sub.empty: continue
            
            grouped = sub.groupby('n')['avg_time'].mean().reset_index()
            
            # For Fennich, only plot up to N=75 (since N=100 and N=150 were skipped due to exponential timeouts)
            if var == 'Fennich_Baseline':
                grouped = grouped[grouped['n'] <= 75]
                
            ax_time.plot(
                grouped['n'], 
                grouped['avg_time'], 
                label=display_labels[var], 
                color=colors[var], 
                marker=markers[var], 
                markersize=7, 
                linewidth=2
            )

        ax_time.set_title('B: Average Computation Time (s) vs. Size', fontweight='bold', pad=12)
        ax_time.set_xlabel('Instance Size ($N$)', fontweight='bold')
        ax_time.set_ylabel('Computation Time (seconds, log scale)', fontweight='bold')
        ax_time.set_yscale('log')
        ax_time.set_xticks([10, 25, 50, 75, 100, 150])
        ax_time.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_time.spines['top'].set_visible(False)
        ax_time.spines['right'].set_visible(False)

        # Place unified legend at the bottom
        handles, labels = ax_gap.get_legend_handles_labels()
        fig.legend(
            handles, 
            labels, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 0.05), 
            ncol=5, 
            frameon=True, 
            edgecolor='#cccccc'
        )

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        
        plt.savefig('plots/Gallo_Ablation_Analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("-> Saved: 'plots/Gallo_Ablation_Analysis.png'")
    else:
        print("-> Warning: No Gallo data found in results. Skipping Gallo plot.")

    # PART 2: DKS ABLATION PLOT
    print("\nGenerating DKS Ablation Analysis plot...")
    df_dks = df[df['inst_type'] == 'DKS'].copy()
    
    if not df_dks.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        # Filter out unused variants (like V3b, V3c) so they don't affect the best_avg_obj calculation
        df_dks = df_dks[df_dks['variant'].isin(variants_to_plot)].copy()
        
        # Calculate heuristic gap relative to the best variant for each N and Density
        df_dks['best_avg_obj'] = df_dks.groupby(['n', 'density'])['avg_obj'].transform('max')
        df_dks['heuristic_gap'] = 100 * (df_dks['best_avg_obj'] - df_dks['avg_obj']) / df_dks['best_avg_obj']

        # DKS Gaps (All N)
        ax_gap = axes[0]
        df_gap = df_dks.copy()
        
        for var in variants_to_plot:
            sub = df_gap[df_gap['variant'] == var].sort_values('n')
            if sub.empty: continue
            grouped = sub.groupby('n')['heuristic_gap'].mean().reset_index()
            
            if var == 'Fennich_Baseline':
                grouped = grouped[grouped['n'] <= 200]  # Skipped after N=200
                
            ax_gap.plot(grouped['n'], grouped['heuristic_gap'], label=display_labels[var], color=colors[var], marker=markers[var], markersize=7, linewidth=2)

        ax_gap.set_title('A: Gap to Best Heuristic (%) vs. Size', fontweight='bold', pad=12)
        ax_gap.set_xlabel('Instance Size ($N$)', fontweight='bold')
        ax_gap.set_ylabel('Average Gap (%)', fontweight='bold')
        ax_gap.set_xticks([10, 25, 50, 100, 150, 200, 250, 300])
        ax_gap.grid(True, linestyle='--', alpha=0.5)
        ax_gap.spines['top'].set_visible(False)
        ax_gap.spines['right'].set_visible(False)

        # DKS Runtimes (N=10 to 300)
        ax_time = axes[1]
        for var in variants_to_plot:
            sub = df_dks[df_dks['variant'] == var].sort_values('n')
            if sub.empty: continue
            grouped = sub.groupby('n')['avg_time'].mean().reset_index()
            if var == 'Fennich_Baseline':
                grouped = grouped[grouped['n'] <= 200]  # Skipped after N=200
            ax_time.plot(grouped['n'], grouped['avg_time'], label=display_labels[var], color=colors[var], marker=markers[var], markersize=7, linewidth=2)

        ax_time.set_title('B: Average Computation Time (s) vs. Size', fontweight='bold', pad=12)
        ax_time.set_xlabel('Instance Size ($N$)', fontweight='bold')
        ax_time.set_ylabel('Time (seconds, log scale)', fontweight='bold')
        ax_time.set_yscale('log')
        ax_time.set_xticks([10, 25, 50, 100, 150, 200, 250, 300])
        ax_time.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_time.spines['top'].set_visible(False)
        ax_time.spines['right'].set_visible(False)

        handles, labels = ax_gap.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=5, frameon=True, edgecolor='#cccccc')
        
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        
        plt.savefig('plots/DKS_Ablation_Analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("-> Saved: 'plots/DKS_Ablation_Analysis.png'")
    else:
        print("-> Warning: No DKS data found in results. Skipping DKS plot.")

    # PART 3: HIDDEN CLIQUE ABLATION PLOT
    print("\nGenerating Hidden Clique Ablation Analysis plot...")
    df_hc = df[df['inst_type'] == 'Hidden_Clique'].copy()
    
    if not df_hc.empty:
        # Target representative sizes for visualization
        target_ns = [50, 100, 150, 200, 250, 300, 400, 500, 600]
        df_hc_sub = df_hc[df_hc['n'].isin(target_ns)].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        
        # HC Gaps
        ax_gap = axes[0]
        for var in variants_to_plot:
            sub = df_hc_sub[df_hc_sub['variant'] == var].sort_values('n')
            if sub.empty: continue
            grouped = sub.groupby('n')['avg_gap'].mean().reset_index()
            ax_gap.plot(grouped['n'], grouped['avg_gap'], label=display_labels[var], color=colors[var], marker=markers[var], markersize=7, linewidth=2)

        ax_gap.set_title('A: Average Optimality Gap (%) vs. Size', fontweight='bold', pad=12)
        ax_gap.set_xlabel('Instance Size ($N$)', fontweight='bold')
        ax_gap.set_ylabel('Average Gap (%)', fontweight='bold')
        ax_gap.set_xticks(target_ns)
        ax_gap.grid(True, linestyle='--', alpha=0.5)
        ax_gap.spines['top'].set_visible(False)
        ax_gap.spines['right'].set_visible(False)

        # HC Runtimes
        ax_time = axes[1]
        for var in variants_to_plot:
            sub = df_hc_sub[df_hc_sub['variant'] == var].sort_values('n')
            if sub.empty: continue
            grouped = sub.groupby('n')['avg_time'].mean().reset_index()
            ax_time.plot(grouped['n'], grouped['avg_time'], label=display_labels[var], color=colors[var], marker=markers[var], markersize=7, linewidth=2)

        ax_time.set_title('B: Average Computation Time (s) vs. Size', fontweight='bold', pad=12)
        ax_time.set_xlabel('Instance Size ($N$)', fontweight='bold')
        ax_time.set_ylabel('Time (seconds, log scale)', fontweight='bold')
        ax_time.set_yscale('log')
        ax_time.set_xticks(target_ns)
        ax_time.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_time.spines['top'].set_visible(False)
        ax_time.spines['right'].set_visible(False)

        handles, labels = ax_gap.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=5, frameon=True, edgecolor='#cccccc')
        
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        
        plt.savefig('plots/HC_Ablation_Analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("-> Saved: 'plots/HC_Ablation_Analysis.png'")
    else:
        print("-> Warning: No Hidden Clique data found in results. Skipping Hidden Clique plot.")

    print("Ablation plots complete.")

if __name__ == "__main__":
    generate_all_ablation_plots()
