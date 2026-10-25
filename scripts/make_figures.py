"""Figures of the report and the README from results/.

    python scripts/make_figures.py
"""
import json

import numpy as np
import pandas as pd

from _path import ROOT
from pnpssp.plotting import C, INK, MUTED, plt, save

RES, FIG = ROOT / "results", ROOT / "figures"
B = json.loads((RES / "benchmark.json").read_text())

# ---------------------------------------------------------------- 1. benchmark: cost per episode, training curve
ev = pd.read_csv(RES / "costs_eval.csv")
tr = pd.read_csv(RES / "training.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
bins = np.linspace(ev.values.min(), ev.values.max(), 45)
for k, (col, lab) in enumerate((("greedy", "greedy baseline"), ("optimal", "optimal (value iteration)"),
                                 ("q_learning", "Q-learning"))):
    a1.hist(ev[col], bins=bins, color=[C[1], C[0], C[2]][k], alpha=0.55 if k else 0.7,
            label=f"{lab}: {ev[col].mean():.0f}")
a1.set_xlabel("cost of an episode (1000 games)")
a1.set_ylabel("episodes")
a1.legend(fontsize=7)
roll = tr.episode_cost.rolling(50, min_periods=1).mean()
a2.plot(tr.episode, roll, color=C[2], label="Q-learning, mean of 50 episodes")
a2.axhline(B["V_star"], color=C[0], ls="--", lw=1, label=f"optimum V* = {B['V_star']:.0f}")
a2.axhline(B["J_greedy"], color=C[1], ls=":", lw=1, label=f"greedy = {B['J_greedy']:.0f}")
a2.set_yscale("log")
a2.set_xlabel("training episode")
a2.set_ylabel("episode cost")
a2.legend(fontsize=7)
save(fig, FIG / "benchmark.png")

# ---------------------------------------------------------------- 2. discount factor
g = pd.read_csv(RES / "gamma_sweep.csv")
fig, ax = plt.subplots(figsize=(5.2, 3.2))
gg = g[g.gamma < 1]
ax.plot(1 - gg.gamma, gg.delta_pct, "o-", color=C[0])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("1 - gamma")
ax.set_ylabel("extra cost of the discounted policy [%]")
save(fig, FIG / "gamma.png")

# ---------------------------------------------------------------- 3. recirculation and learning on the plant model
r = pd.read_csv(RES / "recirculation.csv")
pp = pd.read_csv(RES / "plant_policies.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), width_ratios=[1.1, 1])
alt = r[r["merge"].isin(["-", "alternate"])]
lab = alt.loop_delay.astype(str).str.replace("never (MDP)", "never\n(MDP)", regex=False).unique()
x = np.arange(len(lab))
for k, (pol, col) in enumerate((("optimal (MDP)", C[0]), ("greedy", C[1]))):
    d = alt[alt.policy == pol]
    a1.bar(x + (k - 0.5) * 0.38, d.decisions, 0.38, yerr=1.96 * d.decisions_se, color=col, label=pol)
a1.set_xticks(x, lab)
a1.set_xlabel("decisions until a looped or evicted game is back at the robot")
a1.set_ylabel("robot decisions for 1000 games")
a1.set_ylim(1200, None)
a1.legend(fontsize=7, loc="upper right")
a1.grid(axis="x", visible=False)
short = {"greedy": "greedy", "optimal MDP policy (gamma = 1)": "MDP optimum",
         "k-free discounted MDP policy (warm-start point)": "MDP optimum,\nk-free, gamma 0.99",
         "Q-learning on the plant model, from scratch": "Q-learning on\nplant model",
         "Q-learning on the plant model, warm start": "Q-learning on\nplant model,\nwarm start"}
cols = [C[1], C[0], C[0], C[2], C[2]]
y = np.arange(len(pp))
a2.barh(y, pp.decisions, xerr=1.96 * pp.decisions_sd / np.sqrt(30), color=cols)
a2.set_yticks(y, [short[p] for p in pp.policy], fontsize=7)
a2.invert_yaxis()
a2.set_xlim(1200, None)
a2.set_xlabel("robot decisions for 1000 games (loop delay 20)")
for i, (v, sd) in enumerate(zip(pp.decisions, pp.decisions_sd)):
    a2.text(v + 1.96 * sd / np.sqrt(30) + 12, i, f"{v:.0f}", va="center", fontsize=7, color=INK)
a2.grid(axis="y", visible=False)
save(fig, FIG / "recirculation.png")
print("figures written to", FIG)

# ---------------------------------------------------------------- 4. the SSP with returning games on the plant model
rs = pd.read_csv(RES / "recirculation_ssp.csv")
cases = list(dict.fromkeys(zip(rs.loop_delay, rs["merge"])))
fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(len(cases))
w = 0.27
for k, (lab, col) in enumerate((("greedy", C[1]), ("SSP optimum without return", MUTED),
                                 ("SSP with return", C[0]))):
    vals, errs, fin = [], [], []
    for d, mg in cases:
        g = rs[(rs.loop_delay == d) & (rs["merge"] == mg)]
        if lab == "SSP with return":
            own = g[g.policy == f"SSP with return (d = {d}, {mg.replace('_', ' ')})"]
            g = own if len(own) else g[g.policy == "SSP with return (d = 20, alternate)"]
        else:
            g = g[g.policy == lab]
        r0 = g.iloc[0]
        vals.append(r0.decisions)
        errs.append(1.96 * r0.decisions_sd / np.sqrt(max(r0.finished, 1)))
        fin.append(int(r0.finished))
    ax.bar(x + (k - 1) * w, vals, w, yerr=errs, color=col, label=lab if lab != "SSP with return" else
           "SSP with return (this model)")
    for xi, v, e, f in zip(x + (k - 1) * w, vals, errs, fin):
        if f < 30:
            ax.text(xi, v + e + 12, f"{f}/30\nfinish", ha="center", fontsize=6, color=INK)
ax.set_xticks(x, [f"d = {d}\n{mg.replace('_', ' ')}" for d, mg in cases])
ax.set_ylim(1200, None)
ax.set_ylabel("robot decisions for 1000 games")
ax.set_xlabel("plant model: round trip of the loop (decisions) and merge rule")
ax.legend(fontsize=7, loc="upper right", ncol=3)
ax.grid(axis="x", visible=False)
save(fig, FIG / "recirculation_ssp.png")
