"""LaTeX tables of the report from results/.

    python report/make_tables.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)


def tab(spec, header, rows):
    return (f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


B = json.loads((RES / "benchmark.json").read_text())
S = B["simulated"]
rows = []
for k, lab, J in (("optimal", "optimal (value iteration)", B["V_star"]), ("q_learning", "Q-learning", B["J_ql"]),
                  ("greedy", "greedy baseline", B["J_greedy"])):
    v = S[k]
    rows.append(f"{lab} & {J:.1f} & {v['mean_cost']:.1f} $\\pm$ {v['std_cost']:.1f} & {v['mean_length']:.0f} & "
                f"{v['mean_loops']:.0f} & {v['mean_evictions']:.1f} \\\\")
(OUT / "benchmark.tex").write_text(tab("lrrrrr", "policy & exact $J$ & simulated cost & decisions & loops & "
                                                "evictions", rows))

sh = pd.read_csv(RES / "share_sensitivity.csv")
rows = [f"{r.case} & {r.V_star:.0f} & {r.J_greedy:.0f} & {r.saving_pct:.1f}\\% \\\\" for r in sh.itertuples()]
(OUT / "shares.tex").write_text(tab("lrrr", "shares DE/EN/ES & optimum $V^*$ & greedy & saving", rows))

r = pd.read_csv(RES / "recirculation.csv")
rows = []
for (d, m), g in r.groupby(["loop_delay", "merge"], sort=False):
    o, gr = g[g.policy.str.startswith("optimal")].iloc[0], g[g.policy == "greedy"].iloc[0]

    def cell(x):
        if x.finished == 0:
            return "--- (none finished)"
        f = "" if x.finished == x.episodes else f" ({x.finished}/{x.episodes} finished)"
        return f"{x.decisions:.0f}{f}"
    rows.append(f"{d} & {m.replace('_', ' ')} & {cell(o)} & {cell(gr)} \\\\")
(OUT / "recirculation.tex").write_text(tab("llrr", "games return after & merge & MDP optimum & greedy", rows))

pp = pd.read_csv(RES / "plant_policies.csv")
rows = [f"{x.policy} & {x.decisions:.0f} $\\pm$ {x.decisions_sd:.0f} & {x.loops:.0f} & {x.evictions:.0f} \\\\"
        for x in pp.itertuples()]
(OUT / "plant.tex").write_text(tab("lrrr", "policy & decisions for 1000 games & loops & evictions", rows))
print("tables written")

rs = pd.read_csv(RES / "recirculation_ssp.csv")
rows = []
for x in rs.itertuples():
    pol = x.policy.replace("SSP with return", "with return").replace("SSP optimum without return",
                                                                     "SSP optimum without return")
    fin = "" if x.finished == 30 else f" ({x.finished}/30)"
    model = "" if pd.isna(x.model_decisions) else f"{x.model_decisions:.0f}"
    rows.append(f"{x.loop_delay} & {x.merge.replace('_', ' ')} & {pol} & {x.decisions:.0f} $\\pm$ "
                f"{x.decisions_sd:.0f}{fin} & {x.loops:.0f} & {x.evictions:.0f} & {model} \\\\")
(OUT / "recirculation_ssp.tex").write_text(tab("rllrrrr", "$d$ & merge & policy & decisions & loops & evictions & "
                                               "model", rows))
