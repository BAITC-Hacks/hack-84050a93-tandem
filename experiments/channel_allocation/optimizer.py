"""Experimental fixed-audience optimizer; rejected for production after evaluation."""

def allocate_channels(chosen, options, budget):
    """Exact multiple-choice allocation for a FIXED set of disjoint audiences.

    Each DP layer selects one channel variant for one audience. Only dominated
    (more expensive, no better) states within that layer are discarded. Keeping
    every feasible variant permits jointly downgrading one and upgrading another.
    This optimizes estimated risk-adjusted gain, never hidden realized effects.
    """
    if not chosen:
        return []
    grouped = {}
    for option in options:
        grouped.setdefault((option.mask, option.contacts), []).append(option)
    frontier = [(0.0, 0.0, ())]
    for original in chosen:
        variants = grouped.get((original.mask, original.contacts), [original])
        by_cost = {}
        for cost, gain, path in frontier:
            for option in variants:
                next_cost = cost + option.cost
                if next_cost > budget:
                    continue
                next_gain = gain + option.gain
                incumbent = by_cost.get(next_cost)
                if incumbent is None or next_gain > incumbent[0]:
                    by_cost[next_cost] = (next_gain, path + (option,))
        frontier = []
        best_gain = -float('inf')
        for cost in sorted(by_cost):
            gain, path = by_cost[cost]
            if gain > best_gain:
                frontier.append((cost, gain, path))
                best_gain = gain
        if not frontier:
            return list(chosen)
    _, gain, path = max(frontier, key=lambda state: (state[1], -state[0]))
    # Keep exact ties stable, and retain the feasible incumbent if floating-point
    # accumulation produces a negligible numerical difference.
    if gain <= sum(o.gain for o in chosen) + 1e-8:
        return list(chosen)
    return list(path)

