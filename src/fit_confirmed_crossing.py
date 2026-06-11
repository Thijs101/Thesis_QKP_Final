import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
import os

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

def hybrid_gaussian_func(n, a, b, c, d):
    """
    A hybrid linear-Gaussian function used to model the relationship 
    between instance size N and the required beam width B*.
    """
    return a * n + b * np.exp(-(n - c)**2 / (2 * d**2))

def get_confirmed_crossing_b(df_n, threshold):
    """
    Finds the smallest beam width where the optimality gap falls below 
    the given threshold and stays below it for subsequent beam widths.
    """
    df_n = df_n.sort_values('beam_width')
    beams = df_n['beam_width'].values
    gaps = df_n['avg_gap (%)'].values
    
    for i in range(len(beams)):
        if gaps[i] <= threshold:
            if i == len(beams) - 1:
                return beams[i], True
            # Look-ahead check: ensure the trend remains stable below threshold
            look_ahead = min(3, len(beams) - i - 1)
            next_avg = np.mean(gaps[i+1 : i+1+look_ahead])
            if next_avg <= threshold:
                return beams[i], True
    
    return beams[np.argmin(gaps)], False

def fit_and_plot(csv_file, output_file, threshold=0.5, show_buffer=False):
    """
    Fits the hybrid Gaussian model to the mountain curve data 
    and generates a visualization of the beam width scaling law.
    """
    if not os.path.exists(csv_file):
        print(f"Skipping fit: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    all_ns = sorted(df['n'].unique())
    
    n_vals, b_vals, is_confirmed = [], [], []
    
    for n in all_ns:
        df_n = df[df['n'] == n].copy()
        b_star, conf = get_confirmed_crossing_b(df_n, threshold)
        n_vals.append(n)
        b_vals.append(b_star)
        is_confirmed.append(conf)
    
    n_vals = np.array(n_vals, dtype=float)
    b_vals = np.array(b_vals, dtype=float)
    is_confirmed = np.array(is_confirmed)
    
    # --- Model Fitting ---
    p0 = [0.1, 200, 200, 60] # Initial guess for parameters
    popt, _ = curve_fit(hybrid_gaussian_func, n_vals, b_vals, p0=p0, maxfev=10000)
    a, b, c, d = popt
    
    # Calculate R-squared to evaluate fit quality
    ss_tot = np.sum((b_vals - np.mean(b_vals))**2)
    r2 = 1 - np.sum((b_vals - hybrid_gaussian_func(n_vals, *popt))**2) / ss_tot
    
    n_line = np.linspace(0, max(n_vals) + 50, 300)
    b_line = hybrid_gaussian_func(n_line, *popt)
    
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    # Fitted trend line (Line of Best Fit)
    ax.plot(n_line, b_line, color='#c0392b', linewidth=2.2, zorder=3,
            label=r'$B^*(N) = %.2fN + %.0f \cdot \exp\left(-\frac{(N - %d)^2}{2 \cdot %d^2}\right)$' % (a, b, int(round(c)), int(round(d))))
            
    if show_buffer:
        # Safety Envelope (Buffer)
        buffer_multiplier = 1.05
        b_line_buffer = b_line * buffer_multiplier
        ax.plot(n_line, b_line_buffer, color='#e67e22', linewidth=2.0, linestyle='--', zorder=3,
                label=f'+{int((buffer_multiplier-1)*100)}% Safety Buffer')
                
        # Shaded region between best fit and safety envelope
        ax.fill_between(n_line, b_line, b_line_buffer, color='#e67e22', alpha=0.15, zorder=2)
    
    # Observed data points
    ax.scatter(n_vals[is_confirmed], b_vals[is_confirmed], 
               color='#2c3e50', s=55, zorder=5, edgecolors='white', linewidths=0.8,
               label='Confirmed crossing')
    if np.any(~is_confirmed):
        # Fix the invalid escape sequence warning by using raw string r''
        ax.scatter(n_vals[~is_confirmed], b_vals[~is_confirmed],
                   color='#2c3e50', s=55, zorder=5, facecolors='none', linewidths=1.5,
                   label=fr'Fallback (not reliably $< {threshold}\%$)')
    
    # R-squared annotation
    ax.text(0.97, 0.03, f'$R^2 = {r2:.3f}$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9))
    
    ax.set_xlabel('Instance Size ($N$)')
    ax.set_ylabel('Required Beam Width ($B^*$)')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, linestyle='-', alpha=0.15, color='gray')
    
    # Styling cleanup
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Fit results saved to {output_file}")

if __name__ == "__main__":
    fit_and_plot("volledige_onderzoek_results_diverse.csv", "plots/Confirmed_Crossing_Diverse_Clean.png", threshold=0.5, show_buffer=False)
    fit_and_plot("volledige_onderzoek_results_diverse.csv", "plots/Confirmed_Crossing_Diverse_Buffered.png", threshold=0.5, show_buffer=True)
    fit_and_plot("volledige_onderzoek_results_aggregated_anchor.csv", "plots/Confirmed_Crossing_Aggregated_Anchor_Clean.png", threshold=0.5, show_buffer=False)
    fit_and_plot("volledige_onderzoek_results_aggregated_anchor.csv", "plots/Confirmed_Crossing_Aggregated_Anchor_Buffered.png", threshold=0.5, show_buffer=True)
