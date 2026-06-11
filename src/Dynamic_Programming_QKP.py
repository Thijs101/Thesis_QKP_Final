import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import random
def calculate_pi3_tilde(k, capacity, Q, weights):
    """
    Calculates the upper bound (pi3_tilde) for a given item k.
    """
    n = len(weights)
    qkk = Q[k, k]
    remaining = capacity - weights[k]
    
    if remaining <= 0:
        return qkk
    
    interactions = []
    for j in range(n):
        if j != k:
            profit = Q[k,j]
            weight = weights[j]
            if weight > 0 and profit > 0:
                interactions.append((profit/weight, profit, weight))
    
    interactions.sort(reverse=True)
        
    total = 0.0
    cap_left = remaining
    for ratio, profit, weight in interactions:
        if weight <= cap_left:
            total += profit
            cap_left -= weight
        else:
            total += ratio * cap_left
            break
    
    return qkk + total

def get_item_order(n, capacity, Q, weights):
    """
    Sorts items by their profit/weight ratio for the processing order.
    """
    ratios = []
    for k in range(n):
        pi3_tilde = calculate_pi3_tilde(k, capacity, Q, weights)
        ratio = pi3_tilde / weights[k]
        ratios.append((ratio, k))
    
    ratios.sort(reverse=True)
    return [k for _, k in ratios]

def fill_up_and_exchange(n, capacity, Q, weights, initial_items):
    '''
    Local search: 1. Fill up (Greedy) 2. Exchange (First Improvement)
    Restarts the whole process as soon as an improvement is made.
    '''
    current_items = set(initial_items)
    current_weight = sum(weights[i] for i in current_items)
    
    idx = list(current_items)
    # Pre-calculated base value
    current_value = float(np.sum(Q[np.ix_(idx, idx)]))

    improved = True
    iterations = 0
    while improved:
        iterations += 1
        improved = False
        
        # Vectorized margin pre-calculation
        idx = list(current_items)
        node_margin = np.diag(Q).astype(float)
        
        if idx:
            node_margin += np.sum(Q[:, idx], axis=1) + np.sum(Q[idx, :], axis=0)
            node_margin[idx] -= 2 * np.diag(Q)[idx]

        # Step 0: Fill up
        not_in_solution = set(range(n)) - current_items
        for i in not_in_solution:
            if current_weight + weights[i] <= capacity:
                delta_add = node_margin[i]
                
                current_items.add(i)
                current_weight += weights[i]
                current_value += delta_add
                improved = True 
                break 
        
        if improved: continue

        # Step 1 & 2: Exchange
        items_to_remove = list(current_items)
        items_to_add = list(not_in_solution)
        
        # Outer loop: Step 1 (Remove item i)
        for i_remove in items_to_remove:
            # Inner loop: Step 2 (Try all j to add)
            for j_add in items_to_add:
                new_weight = current_weight - weights[i_remove] + weights[j_add]
                
                if new_weight <= capacity:
                    delta_swap = node_margin[j_add] - node_margin[i_remove] - (Q[j_add, i_remove] + Q[i_remove, j_add])
                    
                    if delta_swap > 0:
                        current_items.remove(i_remove)
                        current_items.add(j_add)
                        current_weight = new_weight
                        current_value += delta_swap
                        improved = True
                        break # Exit inner loop
            
            if improved: break # Exit outer loop to restart from Step 0

    return current_items, current_value, iterations

def get_best_dynamic_item(remaining_items, previous_beam_row, Q, weights, capacity, use_aggregated_anchor=False, aggregated_anchor_pct=0.15, beam_width=1):
    """
    Calculates the best next item using dynamic steering anchors.
    """
    all_states = []
    for weight_states in previous_beam_row:
        for profit, item_set in weight_states:
            w_total = sum(weights[i] for i in item_set)
            all_states.append((profit, item_set, w_total))
    
    if not all_states: return -1
    
    # Sort by profit first (descending), and by item set size second (descending) to match original baseline tie-breakers
    all_states.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    
    anchors = []
    if use_aggregated_anchor:
        aggregated_anchor_size = max(1, int(round(beam_width * aggregated_anchor_pct)))
        aggregated_anchor_size = min(len(all_states), aggregated_anchor_size)
        anchors = [(s, w) for p, s, w in all_states[:aggregated_anchor_size]]
    else:
        p_leader_set = all_states[0][1]
        p_leader_weight = all_states[0][2]
        # Single anchor (Profit leader only)
        anchors.append((p_leader_set, p_leader_weight))        
    best_item = -1
    max_avg_priority = -float('inf')
    
    for j in remaining_items:
        wj = weights[j]
        qjj = Q[j, j]
        
        total_priority = 0
        for anchor_set, anchor_weight in anchors:
            fixed_synergy = qjj + 2 * sum(Q[j, k] for k in anchor_set)
            potential_synergy = 0
            rem_cap = capacity - anchor_weight - wj
            
            if rem_cap >= 0:
                for l in remaining_items:
                    if l != j and weights[l] <= rem_cap:
                        if Q[j, l] > 0:
                            potential_synergy += Q[j, l]
            
            priority = float((fixed_synergy + potential_synergy) / wj)
            total_priority += priority
            
            
        avg_priority = total_priority / len(anchors)
        
        if avg_priority > max_avg_priority:
            max_avg_priority = avg_priority
            best_item = j
            
    return best_item



def diverse_prune(candidates, beam_width):
    """
    Retains beam_width states split into two equal groups:
    exploitation (profit) and exploration (structural distance).
    """
    # Deduplicate: keep only the best profit per unique item set
    unique_cands = {}
    for p, s in candidates:
        if s not in unique_cands or p > unique_cands[s]:
            unique_cands[s] = p
    deduplicated = sorted(unique_cands.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    deduplicated = [(p, s) for s, p in deduplicated]

    if len(deduplicated) <= beam_width:
        return deduplicated

    half = beam_width // 2
    group1    = deduplicated[:half]   # exploitation: top B/2 by profit
    remaining = deduplicated[half:]

    def to_bitmask(s):
        b = 0
        for item in s:
            b |= (1 << item)
        return b

    def jaccard_xor(b1, pop1, b2, pop2):
        xor_pop = (b1 ^ b2).bit_count()
        union = (pop1 + pop2 + xor_pop) // 2
        if union == 0:
            return 0.0
        intersection = (pop1 + pop2 - xor_pop) // 2
        return intersection / union

    g1_data = [(to_bitmask(s), to_bitmask(s).bit_count()) for _, s in group1]
    rem_data = [(p, s, to_bitmask(s), to_bitmask(s).bit_count()) for p, s in remaining]

    # Precompute each remaining candidate's max-similarity to group1
    max_sims = [max((jaccard_xor(b, pop, gb, gpop) for gb, gpop in g1_data), default=0.0)
                for _, _, b, pop in rem_data]

    # Greedily select the most-distant candidate; update cache incrementally
    group2 = []
    for _ in range(beam_width - half):
        if not rem_data:
            break
        best_idx = min(range(len(rem_data)), key=lambda i: max_sims[i])
        winner_p, winner_s, winner_b, winner_pop = rem_data.pop(best_idx)
        max_sims.pop(best_idx)
        group2.append((winner_p, winner_s))
        # Update remaining candidates' similarity cache with the new winner
        for i, (_, _, b, pop) in enumerate(rem_data):
            sim = jaccard_xor(b, pop, winner_b, winner_pop)
            if sim > max_sims[i]:
                max_sims[i] = sim

    return group1 + group2



def fl_heuristic_enhanced(n, capacity, Q, weights, use_dynamic=True, beam_width=1, multi_path_ls=True, use_diversity=True, use_aggregated_anchor=False, aggregated_anchor_pct=0.15):
    """Simple beam search + dynamic ordering."""
    static_order = get_item_order(n, capacity, Q, weights)
    
    if use_dynamic:
        remaining_items = set(range(n))
    
    f_beam = [[[] for _ in range(capacity + 1)] for _ in range(n + 1)]
    f_beam[0][0] = [(0, frozenset())]

    start_dp = time.process_time()
    for k in range(1, n + 1):
        if use_dynamic:
            if k == 1:
                orig_k = static_order[0]
            else:
                orig_k = get_best_dynamic_item(
                    remaining_items, f_beam[k-1], Q, weights, capacity, 
                    use_aggregated_anchor=use_aggregated_anchor, 
                    aggregated_anchor_pct=aggregated_anchor_pct, 
                    beam_width=beam_width
                )
            remaining_items.remove(orig_k)
        else:
            orig_k = static_order[k-1]

        wk = int(weights[orig_k])
        qkk = float(Q[orig_k, orig_k])

        for r in range(capacity + 1):
            f_beam[k][r].extend(f_beam[k-1][r])

            if r >= wk:
                for prev_profit, prev_set in f_beam[k-1][r - wk]:
                    synergy = sum(Q[orig_k, i] for i in prev_set)
                    new_profit = float(prev_profit + qkk + 2*synergy)
                    new_set = prev_set | frozenset([orig_k])
                    f_beam[k][r].append((new_profit, new_set))

            if len(f_beam[k][r]) > beam_width:
                if use_diversity and beam_width >= 2:
                    f_beam[k][r] = diverse_prune(f_beam[k][r], beam_width)
                else:
                    f_beam[k][r].sort(key=lambda x: (x[0], len(x[1])), reverse=True)
                    f_beam[k][r] = f_beam[k][r][:beam_width]
    
    
    for r in range(capacity + 1):
        if f_beam[n][r]:
            f_beam[n][r].sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    # Collect exactly `beam_width` solutions: the best across all capacities for each index tier (0 to beam_width-1)
    final_beam = []
    seen_sets = set()
    
    for i in range(beam_width):
        best_profit_for_rank = -float('inf')
        best_set_for_rank = None
        
        for r in range(capacity + 1):
            if len(f_beam[n][r]) > i:
                profit, item_set = f_beam[n][r][i]
                if profit > best_profit_for_rank:
                    best_profit_for_rank = profit
                    best_set_for_rank = item_set
                    
        if best_set_for_rank is not None:
            if best_set_for_rank not in seen_sets:
                seen_sets.add(best_set_for_rank)
                final_beam.append((best_profit_for_rank, best_set_for_rank))
                
    # Sort final_beam so the globally best is at the top [0]
    final_beam.sort(key=lambda x: x[0], reverse=True)
    
    dp_only_best_val = final_beam[0][0] if final_beam else -float('inf')
    dp_time = time.process_time() - start_dp

    start_ls = time.process_time()
    ls_improvements = []
    
    best_final_val = -float('inf')
    best_final_set = set()
    
    total_ls_iters = 0
    unique_ls_sols = set()
    
    if multi_path_ls:
        for profit_dp, item_set_dp in final_beam:
            items_polished, value_polished, iters = fill_up_and_exchange(n, capacity, Q, weights, set(item_set_dp))
            total_ls_iters += iters
            unique_ls_sols.add(frozenset(items_polished))
            ls_improvements.append(value_polished - profit_dp)
            if value_polished > best_final_val:
                best_final_val = value_polished
                best_final_set = items_polished
    else:
        best_dp_set = final_beam[0][1] if final_beam else set()
        best_final_set, best_final_val, iters = fill_up_and_exchange(n, capacity, Q, weights, set(best_dp_set))
        total_ls_iters += iters
        unique_ls_sols.add(frozenset(best_final_set))
        ls_improvements.append(best_final_val - dp_only_best_val)
        
    avg_gain = sum(ls_improvements) / len(ls_improvements) if ls_improvements else 0
    total_gain = best_final_val - dp_only_best_val
    ls_time = time.process_time() - start_ls
      
    final_weight = sum(weights[i] for i in best_final_set)
    
    # Calculate avg iterations per branch
    num_paths = len(final_beam) if multi_path_ls else 1
    avg_ls_iters = total_ls_iters / num_paths if num_paths > 0 else 0
    return best_final_set, int(best_final_val), final_weight, dp_only_best_val, avg_gain, total_gain, avg_ls_iters, len(unique_ls_sols), dp_time, ls_time
    # return best_final_set, int(best_final_val), final_weight

def solve_qkp_gurobi(n, capacity, Q, weights, initial_solution=None, time_limit=3600):
    with gp.Model("QKP") as m:
        m.setParam('OutputFlag', 0)
        if time_limit is not None:
            m.setParam('TimeLimit', time_limit)
            
        x = m.addMVar(n, vtype=GRB.BINARY, name="x")
        
        if initial_solution is not None:
            # Create a vector of zeros
            x_start = np.zeros(n)
            # Use list indexing to set the items from your heuristic to 1
            x_start[list(initial_solution)] = 1.0
            # Assign the whole vector at once
            x.Start = x_start
        m.setObjective(x @ Q @ x, GRB.MAXIMIZE)
        m.addConstr(weights @ x <= capacity)
        
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            return np.where(x.X > 0.5)[0].tolist(), m.objVal, 0.0
        elif m.status == GRB.TIME_LIMIT and m.SolCount > 0:
            return np.where(x.X > 0.5)[0].tolist(), m.objVal, m.MIPGap
        else:
            return None, None, None







def generate_gallo_instance(n, density=0.8,seed=None):
    '''Common Instance generator'''
    if seed is not None:
        np.random.seed(seed)
    weights = np.random.randint(1, 51, size=n)
    
    capacity = np.random.randint(50, int(np.sum(weights)+1))
    
    Q = np.zeros((n, n))
    
    for i in range(n):
        # Diagonal (linear profit)
        if np.random.random() < density:
            Q[i, i] = np.random.randint(1, 101)
        
        # Off-diagonal (interaction profits)
        for j in range(i+1, n):
            if np.random.random() < density:
                val = np.random.randint(1, 101)
                Q[i, j] = val
                Q[j, i] = val
                    
    return n, capacity, Q, weights



def generate_hidden_clique(n,seed=None):
    '''Hidden Clique instance generator'''
    if seed is not None:
        np.random.seed(seed)
    clique_size = int(np.sqrt(n))
    weights = np.ones(n, dtype=int)
    capacity = clique_size
    # random interactions profits
    Q = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if np.random.random() < 0.5:
                Q[i, j] = 1
                Q[j, i] = 1
    # clique interactions profits
    clique = np.random.choice(n, clique_size, replace=False)
    for i in clique:
        for j in clique:
            if i != j:
                Q[i, j] = 1
    
    optimal = clique_size * (clique_size - 1)
    return n, capacity, Q, weights, set(clique), optimal

def generate_benchmark_instance(n, density=0.5, seed=None):
    """
    Matches the 'Densest k Subgraph' logic from Pisinger et al. (2007)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 1. Weights are all 1 for the Densest k-Subgraph problem
    weights = np.ones(n, dtype=int)
    
    # 2. k is randomly selected in [2, n-2] as per the paper
    k = random.randint(2, n - 2)
    capacity = k # Because each node has weight 1, capacity = k nodes
    
    # 3. Profits Matrix Q: Only 0s and 1s
    Q = np.zeros((n, n))
    for i in range(n):
        # Diagonal is 0 as per the image text
        Q[i, i] = 0 
        for j in range(i + 1, n):
            # Edge exists with probability 'density'
            if random.random() < density:
                Q[i, j] = Q[j, i] = 1 # Edge profit is exactly 1
            else:
                Q[i, j] = Q[j, i] = 0
                
    return n, capacity, Q, weights, k



