# Quadratic Knapsack Problem: Diverse Beam Search & Safe Scaling

This repository contains the source code, raw data, and visualization scripts for my bachelor's thesis on improving dynamic programming heuristics for the Quadratic Knapsack Problem (QKP).

The codebase implements an enhanced **Dynamic Programming (DP)** heuristic combining **Diverse Beam Search** with a **dynamic item ordering** strategy, evaluated against standard Gallo, Densest k-Subgraph, and hidden clique instances up to N=1200.

## Repository Structure

The repository is organized to ensure full academic reproducibility:

- **`run_thesis_benchmarks.py`**: The main execution pipeline. Running this single file computes all missing data points and generates all final plots and tables used in the thesis.
- **`src/`**: Contains all algorithmic logic and plotting scripts.
  - `Dynamic_Programming_QKP.py`: Core implementation of all heuristic variants.
  - `creating_tables.py`: Functions for parsing and aggregating output data.
  - `evaluate_buffer_levels.py`: Validation logic for the Safe Scaling Rule and Aggregated Anchor.
- **`data/`**: Raw output `.csv` files and Gurobi `.json` caches, ensuring all tables in the thesis are reproducible.
- **`plots/`**: Final `.png` visualizations used in the thesis.

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy gurobipy
```

A valid Gurobi license is required only for exact baseline comparisons on small instances (N ≤ 50). For all larger instances the heuristic runs independently.

## How to Run

1. Clone or download this repository.
2. Open a terminal in the root directory.
3. Run the master pipeline:
```bash
python run_thesis_benchmarks.py
```
The script reads existing `.csv` files in `data/`, skips already computed instances, calculates any missing data points, and outputs updated visualizations to `plots/`.

## Methodology Highlights

- **Diverse Beam Search**: Maintains B candidate solutions per DP state, partitioned into an exploitation group (top profit) and an exploration group (maximally dissimilar by Jaccard distance), preventing premature convergence on structured instances.
- **Dynamic Ordering**: Re-ranks remaining items at each DP stage based on realized synergies with the current best solution, allowing the algorithm to adapt to interaction structure as it is revealed.
- **Safe Scaling Rule**: A Gaussian-linear model fitted to the non-monotone difficulty profile of hidden clique instances, with a +5% safety buffer, used to select appropriate beam widths without manual tuning.
- **Aggregated Anchor**: Averages the dynamic ordering signal across the top 15% of beam states by profit, dampening the reinforcement effect where a single high-profit distractor path would otherwise bias the ordering away from the planted clique.