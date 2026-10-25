"""What the MDP leaves out: looped and evicted games come back to the robot.

The MDP treats a looped or evicted game as gone (the next game is a fresh draw from the source). In the plant it
returns to the robot after a trip around the loop. With the mock model (pnpssp.mock_plantsim) driven through the
same COM client as Plant Simulation, compare the optimal MDP policy and the greedy baseline when games return after
d decisions, for two merge rules where the loop joins the main conveyor: alternate (loop and source in turn) and
loop_first (returning games always first). Reported: decisions needed for 1000 games (a proxy of throughput time),
games in the loop, and episodes that do not finish within MAX_DEC decisions (livelock).

    python scripts/recirculation.py          # -> results/recirculation.csv
"""
import numpy as np
import pandas as pd

from _path import ROOT
from pnpssp.environment import WarehouseEnv
from pnpssp.mock_plantsim import MockPlantSim
from pnpssp.plantsim_com import PlantSimClient
from pnpssp.policies import greedy_policy
from pnpssp.value_iteration import solve

CASES = [(None, "alternate"), (5, "alternate"), (20, "alternate"), (60, "alternate"),
         (5, "loop_first"), (20, "loop_first"), (60, "loop_first")]
N_EP, SEED, MAX_DEC = 30, 7, 6000
vi = solve(verbose=False)
policies = {"optimal (MDP)": vi["policy"], "greedy": greedy_policy(vi["states"])}
index = vi["index"]
rows = []
for d, merge in CASES:
    for name, pol in policies.items():
        dec, loop_mean, stuck = [], [], 0
        env = WarehouseEnv("plantsim", client=PlantSimClient(com=MockPlantSim(loop_delay=d, merge=merge), poll=0),
                           seed=SEED)
        for _ in range(N_EP):
            s, done, L, n = env.reset(), False, [], 0
            while not done and n < MAX_DEC:
                s, _, done = env.step(int(pol[index[s]]))
                L.append(env.info["len_loop"])
                n += 1
            if done:
                dec.append(env.info["sim_time"])
                loop_mean.append(np.mean(L))
            else:
                stuck += 1
        rows.append(dict(loop_delay=d if d is not None else "never (MDP)", merge=merge if d is not None else "-",
                         policy=name, finished=N_EP - stuck, episodes=N_EP,
                         decisions=np.mean(dec) if dec else np.nan,
                         decisions_se=np.std(dec) / np.sqrt(len(dec)) if dec else np.nan,
                         loop_mean=np.mean(loop_mean) if loop_mean else np.nan))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "recirculation.csv", index=False, float_format="%.3f")
