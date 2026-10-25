"""
policies.py
-----------
Policies as arrays over the enumerated states (action index per state, -1 on goal states), and their evaluation.

greedy_policy: the heuristic used as the baseline in the course: put the game into the storage that already holds
its language; otherwise into an empty storage; otherwise send it around the loop. It never evicts.
"""
from typing import Dict

import numpy as np

from .mdp import A_LOOP, A_S1, A_S2, EMPTY, N_FLUSHES, feasible_actions


def greedy_action(q1, n1, q2, n2, lam):
    if q1 == lam:
        return A_S1
    if q2 == lam:
        return A_S2
    if q1 == EMPTY:
        return A_S1
    if q2 == EMPTY:
        return A_S2
    return A_LOOP


def greedy_policy(states) -> np.ndarray:
    pol = np.array([greedy_action(*s[:5]) if s[5] < N_FLUSHES else -1 for s in states], dtype=np.int32)
    for s, a in zip(states, pol):                 # the heuristic only uses feasible actions
        assert a == -1 or a in feasible_actions(*s[:5])
    return pol


def evaluate_policy(policy: np.ndarray, index: dict, env, n_eval: int = 1_000, max_steps: int = 20_000,
                    progress: bool = False, label: str = "") -> Dict:
    """Run n_eval episodes with a fixed policy; mean/std of the episode cost, steps, loops and evictions, and the
    set of visited states (for occupancy-weighted policy agreement)."""
    costs, lengths, loops_, evicts = [], [], [], []
    visited = np.zeros(len(index), dtype=bool)
    terminated = 0
    it = range(n_eval)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc=f"Evaluating {label}", unit="ep")
    for _ in it:
        state = env.reset()
        total, steps, loops, evictions, done = 0.0, 0, 0, 0, False
        for _ in range(max_steps):
            i = index[state]
            visited[i] = True
            q1, n1, q2, n2, lam, k = state
            a = int(policy[i])
            if a == A_LOOP:
                loops += 1
            elif (a == A_S1 and q1 not in (EMPTY, lam)) or (a == A_S2 and q2 not in (EMPTY, lam)):
                evictions += 1
            state, c, done = env.step(a)
            total += c
            steps += 1
            if done:
                break
        terminated += done
        costs.append(total)
        lengths.append(steps)
        loops_.append(loops)
        evicts.append(evictions)
    return {"mean_cost": float(np.mean(costs)), "std_cost": float(np.std(costs)),
            "mean_length": float(np.mean(lengths)), "mean_loops": float(np.mean(loops_)),
            "mean_evictions": float(np.mean(evicts)), "terminated": terminated / n_eval,
            "all_costs": costs, "visited": visited}
