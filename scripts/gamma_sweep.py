"""Price of discounting: solve the gamma-discounted problem exactly, evaluate its policy exactly on the
undiscounted SSP, Delta_gamma = J_1(pi_gamma) - V*_1(s0) >= 0.

    python scripts/gamma_sweep.py          # -> results/gamma_sweep.csv
"""
import numpy as np
import pandas as pd

from _path import ROOT
from pnpssp.environment import WarehouseEnv
from pnpssp.mdp import N_FLUSHES
from pnpssp.policies import evaluate_policy
from pnpssp.value_iteration import expected_initial_value as J0, policy_value, solve

GAMMAS = [0.9, 0.95, 0.99, 0.999, 1.0]
N_EVAL, SEED = 300, 42

ref = solve(gamma=1.0, verbose=False)
states, index, T = ref["states"], ref["index"], ref["tables"]
v1, pi_star = J0(ref["V"], index), ref["policy"]
nongoal = T["K"] < N_FLUSHES
visited = evaluate_policy(pi_star, index, WarehouseEnv("python", seed=SEED), n_eval=N_EVAL)["visited"]
rows = []
for g in GAMMAS:
    r = solve(gamma=g, verbose=False, tables=T, states=states, index=index)
    j1 = J0(policy_value(r["policy"], T, 1.0), index)
    m = evaluate_policy(r["policy"], index, WarehouseEnv("python", seed=SEED), n_eval=N_EVAL)
    rows.append(dict(gamma=g, V_gamma=J0(r["V"], index), J1=j1, delta=j1 - v1, delta_pct=(j1 / v1 - 1) * 100,
                     loops=m["mean_loops"], evictions=m["mean_evictions"], steps=m["mean_length"],
                     agree_all=np.mean(r["policy"][nongoal] == pi_star[nongoal]),
                     agree_visited=np.mean(r["policy"][visited] == pi_star[visited])))
    print({k: round(v, 4) for k, v in rows[-1].items()})
pd.DataFrame(rows).to_csv(ROOT / "results" / "gamma_sweep.csv", index=False, float_format="%.4f")
