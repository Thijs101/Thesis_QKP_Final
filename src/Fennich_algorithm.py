import numpy as np
import time

def get_total_profit(item_set, profits):
    """Calculates the total profit for a set of items."""
    idx = list(item_set)
    return sum(profits[i, j] for i in idx for j in idx)

def chaillou_greedy_heuristic(candidate_items, weights, profits, capacity, fixed_items=None):
    """Greedy filling heuristic for the Fennich algorithm."""
    if not candidate_items or capacity <= 0:
        return set(), 0

    current_set = set(candidate_items)
    fixed_set = set(fixed_items) if fixed_items else set()

    fixed_synergies = {i: 2 * sum(profits[i, j] for j in fixed_set) for i in current_set}
    internal_synergies = {i: 2 * sum(profits[i, j] for j in current_set if i != j) for i in current_set}

    while sum(weights[i] for i in current_set) > capacity and current_set:
        ratios = []
        for i in current_set:
            total_contribution = profits[i, i] + internal_synergies[i] + fixed_synergies[i]
            ratios.append((total_contribution / weights[i], i))
        
        _, worst_item = min(ratios)
        current_set.remove(worst_item)

        for i in current_set:
            internal_synergies[i] -= 2 * profits[i, worst_item]
        
        del internal_synergies[worst_item]
        del fixed_synergies[worst_item]

    final_profit = get_total_profit(current_set | fixed_set, profits) - get_total_profit(fixed_set, profits)
    return current_set, final_profit

def propagate_states(stage_k, current_weight, min_weight, dp_table, set_table, weights, profits, tabu_list):
    """Propagates states backwards to fill intermediate gaps in the DP table."""
    if dp_table[stage_k, current_weight] in tabu_list:
        return

    tabu_list.add(dp_table[stage_k, current_weight])

    while current_weight > min_weight:
        current_set = set_table[stage_k][current_weight]
        if not current_set:
            break

        ratios = []
        for i in current_set:
            efficiency = (profits[i, i] + 2 * sum(profits[i, j] for j in current_set if i != j)) / weights[i]
            ratios.append((efficiency, i))
        
        _, worst_item = min(ratios)
        lighter_set = current_set - {worst_item}
        new_profit = get_total_profit(lighter_set, profits)
        reduced_weight = current_weight - weights[worst_item]

        if new_profit > dp_table[stage_k, reduced_weight]:
            dp_table[stage_k, reduced_weight] = new_profit
            set_table[stage_k][reduced_weight] = lighter_set.copy()
            current_weight = reduced_weight
        else:
            break

def fennich_dp(num_items, capacity, weights, profits):
    """Main Fennich Dynamic Programming implementation."""
    start_dp = time.process_time()
    dp_table = np.full((num_items + 1, capacity + 1), -np.inf)
    set_table = [[set() for _ in range(capacity + 1)] for _ in range(num_items + 1)]
    dp_table[0, 0] = 0.0 
    
    min_item_weight = min(weights)
    tabu_list = set()

    for k in range(1, num_items + 1):
        item_idx = k - 1
        item_weight = weights[item_idx]

        for r in range(capacity + 1):
            if dp_table[k - 1, r] > dp_table[k, r]:
                dp_table[k, r] = dp_table[k - 1, r]
                set_table[k][r] = set_table[k - 1][r].copy()

            if r + item_weight <= capacity and dp_table[k - 1, r] != -np.inf:
                current_set = set_table[k - 1][r] | {item_idx}
                current_profit = get_total_profit(current_set, profits)

                if current_profit > dp_table[k, r + item_weight]:
                    dp_table[k, r + item_weight] = current_profit
                    set_table[k][r + item_weight] = current_set.copy()

                if r + item_weight <= capacity - min_item_weight:
                    remaining_candidates = list(range(k, num_items))
                    residual_capacity = capacity - (r + item_weight)

                    fill_up_set, extra_profit = chaillou_greedy_heuristic(
                        remaining_candidates, weights, profits, residual_capacity, fixed_items=current_set
                    )

                    full_set = current_set | fill_up_set
                    total_profit = current_profit + extra_profit
                    total_weight = sum(weights[i] for i in full_set)

                    if total_weight <= capacity and total_profit > dp_table[k, total_weight]:
                        dp_table[k, total_weight] = total_profit
                        set_table[k][total_weight] = full_set.copy()
                        propagate_states(k, total_weight, r, dp_table, set_table, weights, profits, tabu_list)

    best_weight_idx = int(np.argmax(dp_table[num_items, :]))
    best_set = set_table[num_items][best_weight_idx].copy()
    dp_profit = float(dp_table[num_items, best_weight_idx])
    best_profit = dp_profit
    dp_time = time.process_time() - start_dp

    start_ls = time.process_time()
    ls_iters = 0
    for item in list(best_set):
        ls_iters += 1
        base_set = best_set - {item}
        residual_cap = capacity - sum(weights[i] for i in base_set)
        candidates = [i for i in range(num_items) if i not in base_set and weights[i] <= residual_cap]
        new_fill_set, added_profit = chaillou_greedy_heuristic(candidates, weights, profits, residual_cap, fixed_items=base_set)
        lost_profit = profits[item, item] + 2 * sum(profits[item, i] for i in base_set)

        if added_profit > lost_profit:
            best_profit = best_profit + added_profit - lost_profit
            best_set = base_set | new_fill_set
            
    ls_time = time.process_time() - start_ls
    return best_set, best_profit, dp_profit, dp_time, ls_time, ls_iters