"""The SSP with returning games (pnpssp.recirculation): solve it for the loop of the plant model and test the
policy on the mock of the plant, against the greedy heuristic and the optimum of the SSP without return.

For each case (round trip d in decisions, merge rule) the model is solved by relative value iteration and its policy
is run through the COM client on MockPlantSim(loop_delay=d, merge), where games return after exactly d decisions
(FIFO) instead of the fluid approximation of the model. The d = 20 policy is also run on the other loops, to see how
much the policy depends on knowing d.

    python scripts/recirculation_ssp.py     # -> results/recirculation_ssp.csv  (about 30 min on 2 cores)
"""
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _path import ROOT
from pnpssp.environment import WarehouseEnv
from pnpssp.mdp import A_LOOP, feasible_actions
from pnpssp.mock_plantsim import MockPlantSim
from pnpssp.plantsim_com import PlantSimClient
from pnpssp.policies import greedy_action
from pnpssp.recirculation import RecirculationModel
from pnpssp.value_iteration import solve

L_MAX, N_EP, SEED = 20, 30, 1000
CASES = [(20, "alternate"), (5, "alternate"), (60, "alternate")]
# loop_first is only tested (with the d = 20 policy): when returning games always go first, the robot can be
# starved of new games, the model becomes multichain and relative value iteration does not converge
CACHE = ROOT / "results" / "policies"
TESTS = [(5, "alternate"), (20, "alternate"), (60, "alternate"), (20, "loop_first"), (60, "loop_first")]


def run(choose, d, merge, max_dec=6000):
    env = WarehouseEnv("plantsim", client=PlantSimClient(com=MockPlantSim(loop_delay=d, merge=merge), poll=0),
                       seed=SEED)
    dec, loops, ev, lmax, stuck = [], [], [], 0, 0
    for _ in range(N_EP):
        s, done, n, lp, e = env.reset(), False, 0, 0, 0
        while not done and n < max_dec:
            s, _, done = env.step(choose(s, env.info))
            n += 1
            lp += env.info["did_loop"]
            e += env.info["n_evicted"] > 0
            lmax = max(lmax, env.info["len_loop"])
        if done:
            dec.append(env.info["sim_time"])
            loops.append(lp)
            ev.append(e)
        else:
            stuck += 1
    return dict(finished=N_EP - stuck, decisions=np.mean(dec) if dec else np.nan,
                decisions_sd=np.std(dec) if dec else np.nan, loops=np.mean(loops) if loops else np.nan,
                evictions=np.mean(ev) if ev else np.nan, max_loop=lmax)


def solve_case(case):
    """Solve one case, or load its policy from results/policies (int8 array over the states, ~3 MB)."""
    d, merge = case
    f = CACHE / f"recirc_d{d}_{merge}_L{L_MAX}.npz"
    if f.exists():
        z = np.load(f)
        return case, dict(rate=float(z["rate"]), policy=z["policy"], iterations=int(z["iterations"]))
    t0 = time.time()
    m = RecirculationModel(d=d, merge=merge, L_max=L_MAX)
    r = m.solve(verbose=False, max_iter=5000)
    print(f"model d={d} {merge}: {m.N:,} states, rate {r['rate']:.4f} ({1000 / r['rate']:.0f} decisions), "
          f"{r['iterations']} iterations, {time.time() - t0:.0f} s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, rate=r["rate"], policy=r["policy"], iterations=r["iterations"])
    return case, dict(rate=r["rate"], policy=r["policy"], iterations=r["iterations"])


def policy_of(m, pol):
    def choose(s, info):
        a = int(pol[m.index(*s[:5], info["loop"])])
        return a if a in feasible_actions(*s[:5]) else A_LOOP
    return choose


if __name__ == "__main__":
    with Pool(2) as pool:
        solved = pool.map(solve_case, CASES)
    index_model = RecirculationModel(d=20, L_max=L_MAX)        # state indexing is the same for every case
    models = {c: (index_model, r) for c, r in solved}
    vi = solve(verbose=False)
    pol_ssp, index = vi["policy"], vi["index"]
    base = {"greedy": lambda s, i: greedy_action(*s[:5]),
            "SSP optimum without return": lambda s, i: int(pol_ssp[index[s]])}
    rows = []
    for d, merge in TESTS:
        pols = dict(base)
        if (d, merge) in models:
            m, r = models[(d, merge)]
            pols[f"SSP with return (d = {d}, {merge.replace('_', ' ')})"] = policy_of(m, r["policy"])
        if (d, merge) != (20, "alternate"):
            m, r = models[(20, "alternate")]
            pols["SSP with return (d = 20, alternate)"] = policy_of(m, r["policy"])
        for name, ch in pols.items():
            res = run(ch, d, merge)
            model_dec = np.nan
            if name.startswith("SSP with return") and f"d = {d}, {merge.replace('_', ' ')}" in name:
                model_dec = 1000 / models[(d, merge)][1]["rate"]
            rows.append(dict(loop_delay=d, merge=merge, policy=name, model_decisions=model_dec, **res))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "recirculation_ssp.csv", index=False, float_format="%.3f")
