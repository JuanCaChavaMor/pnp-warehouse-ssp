"""How much the language shares matter: optimum and greedy baseline for the shares used in the different versions
of the project (course task 4, first formulation, Plant Simulation model) and for equal shares.

    python scripts/share_sensitivity.py     # -> results/share_sensitivity.csv
"""
import numpy as np
import pandas as pd

from _path import ROOT
import pnpssp.value_iteration as vi
from pnpssp.policies import greedy_policy

CASES = {"course task 4 (0.2/0.5/0.3)": (0.2, 0.5, 0.3), "first formulation (0.4/0.3/0.3)": (0.4, 0.3, 0.3),
         "Plant Simulation model (0.4/0.35/0.25)": (0.4, 0.35, 0.25), "equal (1/3 each)": (1 / 3, 1 / 3, 1 / 3)}
base = vi.solve(verbose=False)
states, index, T = base["states"], base["index"], base["tables"]
greedy = greedy_policy(states)
rows = []
for name, p in CASES.items():
    vi.ARRIVAL_P = np.array(p)                      # value_iteration reads the shares from this module variable
    r = vi.solve(verbose=False, tables=T, states=states, index=index)
    v = sum(pi * r["V"][index[(0, 0, 0, 0, l, 0)]] for pi, l in zip(p, (1, 2, 3)))
    jg = vi.policy_value(greedy, T, 1.0)
    g = sum(pi * jg[index[(0, 0, 0, 0, l, 0)]] for pi, l in zip(p, (1, 2, 3)))
    rows.append(dict(case=name, p_DE=p[0], p_EN=p[1], p_ES=p[2], V_star=v, J_greedy=g, saving_pct=(1 - v / g) * 100))
    print(rows[-1])
pd.DataFrame(rows).to_csv(ROOT / "results" / "share_sensitivity.csv", index=False, float_format="%.4f")
