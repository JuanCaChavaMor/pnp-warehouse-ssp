"""Train on the plant, not on the MDP: tabular Q-learning through the COM client on the mock model with
recirculation (looped and evicted games come back after `DELAY` decisions, alternate merge).

The state seen by the agent is the MDP state without the flush counter k, (q1, n1, q2, n2, lam), discounted with
GAMMA: the contents of the loop are not observed, so the problem is only approximately Markov. Two starts:
    * from scratch (pessimistic initial values: looping forever),
    * warm start from the MDP solution (the k-free discounted optimum of mdp.py), i.e. learn on the plant what the
      model got wrong.
Each policy is then evaluated greedily, against the greedy baseline and the optimal MDP policy, on the same model.

    python scripts/train_on_plant_model.py   # -> results/plant_training.csv, results/plant_policies.csv
"""
import time

import numpy as np
import pandas as pd

from _path import ROOT
from pnpssp.environment import WarehouseEnv
from pnpssp.mdp import ARRIVAL_P, C_LOOP, N_FLUSHES, feasible_actions
from pnpssp.mock_plantsim import MockPlantSim
from pnpssp.plantsim_com import PlantSimClient
from pnpssp.policies import greedy_action
from pnpssp.value_iteration import solve, solve_aggregated

DELAY, MERGE, GAMMA = 20, "alternate", 0.99
EPISODES, EVAL_EP, SEED = 1500, 30, 3
ALPHA0, OMEGA, EPS0, EPS_MIN, EPS_DECAY = 1.0, 0.7, 0.3, 0.01, 0.997

vi = solve(verbose=False)
states, index, T = vi["states"], vi["index"], vi["tables"]
cfg = sorted({s[:5] for s in states})                          # (q1, n1, q2, n2, lam)
ci = {c: i for i, c in enumerate(cfg)}
FEAS = np.zeros((len(cfg), 3), bool)
for c, i in ci.items():
    FEAS[i, feasible_actions(*c)] = True

# k-free discounted MDP optimum (exact), used for the warm start and as a policy of its own
pol_agg = solve_aggregated(T, index, states, GAMMA)
k0 = {s[:5]: index[s] for s in states if s[5] == 0}
p = np.asarray(ARRIVAL_P)


def q_from_mdp():
    """Q(c, a) = c(s, a) + gamma * E V(f(s, a, l)) on the k = 0 layer of the discounted problem."""
    V = np.zeros(len(states))
    Ck0 = np.array([T["C"][k0[c]] for c in cfg])
    NX = np.array([T["NXT"][k0[c]] for c in cfg])               # (n_cfg, 3, 3) next state indices
    nxt_cfg = np.vectorize(lambda i: ci[states[i][:5]])(NX)
    Vc = np.zeros(len(cfg))
    for _ in range(5000):                                        # discounted VI on configurations (flush stays)
        Q = np.where(FEAS, Ck0 + GAMMA * np.einsum("l,cal->ca", p, Vc[nxt_cfg]), np.inf)
        Vn = Q.min(1)
        if np.max(np.abs(Vn - Vc)) < 1e-9:
            break
        Vc = Vn
    return np.where(FEAS, Q, np.inf)


def make_env(seed):
    return WarehouseEnv("plantsim", client=PlantSimClient(com=MockPlantSim(loop_delay=DELAY, merge=MERGE), poll=0),
                        seed=seed)


def train(Q, seed, label):
    rng = np.random.default_rng(seed)
    N = np.zeros_like(Q)
    env, eps, log = make_env(seed), EPS0, []
    for ep in range(EPISODES):
        s, done, tot, n = env.reset(), False, 0.0, 0
        while not done:
            c = ci[s[:5]]
            feas = np.flatnonzero(FEAS[c])
            a = int(rng.choice(feas)) if rng.random() < eps else int(feas[np.argmin(Q[c, feas])])
            s2, cost, done = env.step(a)
            N[c, a] += 1
            alpha = max(1e-3, ALPHA0 / N[c, a] ** OMEGA)
            target = cost if done else cost + GAMMA * Q[ci[s2[:5]], FEAS[ci[s2[:5]]]].min()
            Q[c, a] += alpha * (target - Q[c, a])
            s, tot, n = s2, tot + cost, n + 1
        eps = max(EPS_MIN, eps * EPS_DECAY)
        log.append(dict(start=label, episode=ep, cost=tot, decisions=env.info["sim_time"], epsilon=eps))
        if ep % 250 == 0:
            print(label, ep, round(tot, 1), env.info["sim_time"], round(eps, 3), flush=True)
    return Q, log


def evaluate(choose, label, seed=1000):
    env, rows = make_env(seed), []
    for _ in range(EVAL_EP):
        s, done, tot, loops, ev = env.reset(), False, 0.0, 0, 0
        while not done:
            s, cost, done = env.step(choose(s))
            tot += cost
            loops += env.info.get("did_loop", 0)
            ev += env.info.get("n_evicted", 0) > 0
        rows.append(dict(policy=label, cost=tot, decisions=env.info["sim_time"], loops=loops, evictions=ev))
    return rows


t0 = time.time()
Q_pess = np.where(FEAS, C_LOOP / (1 - GAMMA), np.inf)
Q_scratch, log1 = train(Q_pess.copy(), SEED, "from scratch")
Q_warm, log2 = train(q_from_mdp(), SEED, "warm start from the MDP")
print(f"training {time.time() - t0:.0f} s")


def from_q(Q):
    return lambda s: int(np.argmin(Q[ci[s[:5]]]))


Q_mdp = q_from_mdp()
ev = []
ev += evaluate(lambda s: greedy_action(*s[:5]), "greedy")
ev += evaluate(lambda s: int(vi["policy"][index[s]]), "optimal MDP policy (gamma = 1)")
ev += evaluate(from_q(Q_mdp), "k-free discounted MDP policy (warm-start point)")
ev += evaluate(from_q(Q_scratch), "Q-learning on the plant model, from scratch")
ev += evaluate(from_q(Q_warm), "Q-learning on the plant model, warm start")
E = pd.DataFrame(ev)
summary = E.groupby("policy", sort=False).agg(decisions=("decisions", "mean"), decisions_sd=("decisions", "std"),
                                               cost=("cost", "mean"), loops=("loops", "mean"),
                                               evictions=("evictions", "mean"))
print(summary.round(1))
summary.to_csv(ROOT / "results" / "plant_policies.csv", float_format="%.2f")
pd.DataFrame(log1 + log2).to_csv(ROOT / "results" / "plant_training.csv", index=False, float_format="%.2f")
