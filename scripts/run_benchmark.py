"""Benchmark of the pure-Python layers: exact optimum (value iteration), greedy baseline, tabular Q-learning.

    python scripts/run_benchmark.py [--gamma 0.99] [--episodes 5000] [--envs 512] [--seed 42] [--n-eval 1000]

Layer 1: value iteration -> V*, pi*  (exact; gamma = 1 is the SSP)
Layer 2: Q-learning on the Python backend -> pi_QL
Layer 3 (Plant Simulation) uses the same agent with WarehouseEnv("plantsim", client=PlantSimClient(...)).

Exact decomposition of the Q-learning gap, all costs undiscounted:
    J(pi_QL) - V*_1 = Delta_gamma (discounting) + Delta_k (ignoring k) + Delta_alg (learning)
Writes results/benchmark.json, results/training.csv, results/agreement_by_k.csv, results/costs_eval.csv.
"""
import argparse
import json

import numpy as np
import pandas as pd

from _path import ROOT
from pnpssp.environment import WarehouseEnv
from pnpssp.mdp import ARRIVAL_P, C_LIN, C_LOOP, C_ROUTE, N_FLUSHES
from pnpssp.policies import evaluate_policy, greedy_policy
from pnpssp.qlearning import QLearningAgent
from pnpssp.value_iteration import expected_initial_value, policy_value, solve, solve_aggregated

ap = argparse.ArgumentParser()
ap.add_argument("--gamma", type=float, default=0.99)
ap.add_argument("--episodes", type=int, default=5_000)
ap.add_argument("--envs", type=int, default=512)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n-eval", type=int, default=1_000)
a = ap.parse_args()
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
J0 = expected_initial_value

# ---------------------------------------------------------------- layer 1: exact SSP optimum and baseline
ref = solve(gamma=1.0, verbose=False)
states, index, T = ref["states"], ref["index"], ref["tables"]
nongoal = T["K"] < N_FLUSHES
v1 = J0(ref["V"], index)
greedy = greedy_policy(states)
j_greedy = J0(policy_value(greedy, T, 1.0), index)
print(f"{len(states):,} states; V*_1(s0) = {v1:.2f}; greedy baseline J = {j_greedy:.2f} "
      f"(+{(j_greedy / v1 - 1) * 100:.1f} %)")

# the reference for Q-learning: the gamma-discounted optimum without the flush counter k
pi_g = solve(gamma=a.gamma, verbose=False, tables=T, states=states, index=index)["policy"]
d_gamma = J0(policy_value(pi_g, T, 1.0), index) - v1
pi_ref = solve_aggregated(T, index, states, a.gamma)
j_ref = J0(policy_value(pi_ref, T, 1.0), index)
d_k = j_ref - v1 - d_gamma
print(f"gamma = {a.gamma}: Delta_gamma = {d_gamma:.3f}, Delta_k = {d_k:.3f} -> reference J = {j_ref:.2f}")

# ---------------------------------------------------------------- layer 2: Q-learning
agent = QLearningAgent(states, index, alpha=1.0, gamma=a.gamma, eps_start=1.0, eps_decay=0.995, eps_min=0.01,
                       seed=a.seed, tables=T, q_init=1.5 / (1 - a.gamma) if a.gamma < 1 else 0.0,
                       alpha_mode="count", alpha_omega=0.7, aggregate_k=a.gamma < 1)
stats = agent.train_parallel(n_envs=a.envs, n_episodes=a.episodes, verbose=False, seed=a.seed + 1)
pi_ql = agent.get_policy()
j_ql = J0(policy_value(pi_ql, T, 1.0), index)
d_alg = j_ql - j_ref
print(f"Q-learning: {a.episodes} episodes in {stats['runtime']:.0f} s; J(pi_QL) = {j_ql:.2f} (Delta_alg = {d_alg:.2f})")

# ---------------------------------------------------------------- simulated episodes (KPIs beyond the cost)
evals = {}
for name, pol, s in (("optimal", ref["policy"], a.seed), ("greedy", greedy, a.seed), ("q_learning", pi_ql, a.seed)):
    evals[name] = evaluate_policy(pol, index, WarehouseEnv("python", seed=s), n_eval=a.n_eval)
visited = evals["optimal"]["visited"]
agree = pi_ql == ref["policy"]
ks = T["K"]
pd.DataFrame({"k": np.arange(N_FLUSHES),
              "agreement": [agree[(ks == k) & visited].mean() for k in range(N_FLUSHES)]}).to_csv(
    RES / "agreement_by_k.csv", index=False, float_format="%.4f")
pd.DataFrame({"episode_cost": stats["episode_costs"], "episode_length": stats["episode_lengths"],
              "epsilon": stats["epsilon_history"]}).to_csv(RES / "training.csv", index_label="episode",
                                                           float_format="%.4f")
pd.DataFrame({k: v["all_costs"] for k, v in evals.items()}).to_csv(RES / "costs_eval.csv", index=False,
                                                                   float_format="%.2f")

# how the optimal policy differs from the greedy one, on the states the optimal policy visits
pol = ref["policy"]
diff = visited & (pol != greedy) & nongoal
kinds = {}
for i in np.flatnonzero(diff):
    q1, n1, q2, n2, lam, k = states[i]
    g, o = greedy[i], pol[i]
    key = f"greedy {['A', 'B', 'LOOP'][g]} -> optimal {['A', 'B', 'LOOP'][o]}" + \
          (" (evict)" if (o == 0 and q1 not in (0, lam)) or (o == 1 and q2 not in (0, lam)) else "")
    kinds[key] = kinds.get(key, 0) + 1

out = {
    "parameters": {"arrival_p": dict(zip(["DE", "EN", "ES"], map(float, ARRIVAL_P))), "capacity": 10,
                   "flushes": N_FLUSHES, "c_route": C_ROUTE, "c_loop": C_LOOP, "C_lin": C_LIN, "gamma": a.gamma,
                   "episodes": a.episodes, "envs": a.envs, "seed": a.seed, "n_eval": a.n_eval},
    "n_states": len(states), "V_star": v1, "J_greedy": j_greedy, "J_ref": j_ref, "J_ql": j_ql,
    "delta_gamma": d_gamma, "delta_k": d_k, "delta_alg": d_alg, "ql_runtime_s": stats["runtime"],
    "agreement_all": float(agree[nongoal].mean()), "agreement_visited": float(agree[visited].mean()),
    "differences_optimal_vs_greedy_visited_states": dict(sorted(kinds.items(), key=lambda x: -x[1])),
    "simulated": {k: {m: v[m] for m in ("mean_cost", "std_cost", "mean_length", "mean_loops", "mean_evictions",
                                       "terminated")} for k, v in evals.items()},
}
(RES / "benchmark.json").write_text(json.dumps(out, indent=2))
for k, v in out["simulated"].items():
    print(f"  {k:10s} cost {v['mean_cost']:8.1f} +- {v['std_cost']:5.1f}  steps {v['mean_length']:7.1f}  "
          f"loops {v['mean_loops']:6.1f}  evictions {v['mean_evictions']:5.1f}")
print(json.dumps(out["differences_optimal_vs_greedy_visited_states"], indent=1))
