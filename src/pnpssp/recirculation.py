"""
recirculation.py
----------------
The warehouse decision problem with games that come back, as in the Plant Simulation model.

In mdp.py a looped or evicted game disappears and the next game at the robot is a fresh draw from the source: LOOP
re-rolls the game, an eviction throws games away. In the plant (plantsim/methods/AIControl) both send the games around
the loop Return_to_mainCon, and they reach the robot again. This model adds the contents of the loop to the state:

    s = (q1, n1, q2, n2, lam, m),   m = (m_DE, m_EN, m_ES) games going around, m_DE + m_EN + m_ES <= L_max

Transitions. The action acts on the storages as in mdp.py, and LOOP or an eviction adds the games to m. The next
game at the robot comes from the loop with probability rho(L) (L = games in the loop after the action) and is then of
language j with probability m_j / L; otherwise it is a new game from the source (shares p). With a mean round trip of
d decisions, about L / d games come back per decision (fluid approximation of the loop), and the merge with the main
conveyor caps the share of returning games:
    merge = "alternate"   rho(L) = min(L / d, 1/2)   (loop and source in turn)
    merge = "loop_first"  rho(L) = min(L / d, 1)     (returning games always first)
The loop holds at most L_max games: actions that would overfill it are infeasible (the conveyor would block); if no
action is left, LOOP is allowed and the loop count saturates at L_max (a boundary case, rare for L_max >= 24).

Objective. Every decision takes one robot cycle; the loop and the evictions no longer need the proxy costs c_loop
and C_lin of mdp.py, because their consequence, games that have to be handled again, is now part of the
dynamics. For 1000 games the goal is the policy that stores games at the highest rate per decision: an average-reward
problem (reward 10 per completed batch), solved by relative value iteration. Decisions for 1000 games ~ 1000 / rate.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from .mdp import A_LOOP, A_S1, A_S2, ARRIVAL_P, CAP, EMPTY, TYPES, feasible_actions


def multisets(L_max: int):
    """All (m_DE, m_EN, m_ES) with sum <= L_max and a dense lookup array."""
    ms = [m for m in itertools.product(range(L_max + 1), repeat=3) if sum(m) <= L_max]
    look = -np.ones((L_max + 1,) * 3, dtype=np.int64)
    for i, m in enumerate(ms):
        look[m] = i
    return np.array(ms, dtype=np.int64), look


def buffer_configs():
    out = []
    for q1, n1, q2, n2 in itertools.product(range(4), range(CAP), range(4), range(CAP)):
        if (q1 == EMPTY) != (n1 == 0) or (q2 == EMPTY) != (n2 == 0):
            continue
        if q1 != EMPTY and q1 == q2:
            continue
        out.append((q1, n1, q2, n2))
    return out


def apply_action(q1, n1, q2, n2, lam, a):
    """Storage update of one action: (new storages, language sent around, number sent around, batch completed)."""
    if a == A_LOOP:
        return (q1, n1, q2, n2), lam, 1, False
    q, n = (q1, n1) if a == A_S1 else (q2, n2)
    sent_lang, sent = 0, 0
    if q not in (EMPTY, lam):                    # eviction: the partial batch goes around
        sent_lang, sent, n = q, n, 0
    n += 1
    done = n == CAP
    new = (EMPTY, 0) if done else (lam, n)
    b = (new + (q2, n2)) if a == A_S1 else ((q1, n1) + new)
    return b, sent_lang, sent, done


class RecirculationModel:
    def __init__(self, d: float = 20.0, merge: str = "alternate", L_max: int = 24, p=ARRIVAL_P):
        if merge not in ("alternate", "loop_first"):
            raise ValueError("merge must be 'alternate' or 'loop_first'")
        self.d, self.merge, self.L_max, self.p = float(d), merge, L_max, np.asarray(p, float)
        self.cap_rho = 0.5 if merge == "alternate" else 1.0
        self.B = buffer_configs()
        self.bi = {b: i for i, b in enumerate(self.B)}
        self.MS, self.ms_look = multisets(L_max)
        self.nB, self.nM = len(self.B), len(self.MS)
        self.N = self.nB * 3 * self.nM
        self._build()

    # state index = ((b * 3) + lam - 1) * nM + m
    def index(self, q1, n1, q2, n2, lam, m):
        m = tuple(int(x) for x in m)
        if sum(m) > self.L_max:                   # beyond the model: saturate the largest counts
            m = list(m)
            while sum(m) > self.L_max:
                m[int(np.argmax(m))] -= 1
            m = tuple(m)
        return (self.bi[(q1, n1, q2, n2)] * 3 + lam - 1) * self.nM + int(self.ms_look[m])

    def rho(self, L):
        return np.minimum(L / self.d, self.cap_rho)

    def _build(self):
        t0 = time.time()
        nB, nM, MS, look, Lm = self.nB, self.nM, self.MS, self.ms_look, self.L_max
        N = self.N
        NXT = np.zeros((N, 3, 6), dtype=np.int32)
        P = np.zeros((N, 3, 6), dtype=np.float32)
        R = np.zeros((N, 3), dtype=np.float32)
        FEAS = np.zeros((N, 3), dtype=bool)
        Lm_arr = MS.sum(1)
        for bi, (q1, n1, q2, n2) in enumerate(self.B):
            for lam in TYPES:
                rows = np.arange(nM) + (bi * 3 + lam - 1) * nM
                feas = feasible_actions(q1, n1, q2, n2, lam)
                for a in feas:
                    b2, sl, sn, done = apply_action(q1, n1, q2, n2, lam, a)
                    add = np.zeros(3, np.int64)
                    if sn:
                        add[sl - 1] = sn
                    M2 = MS + add
                    fits = M2.sum(1) <= Lm
                    FEAS[rows[fits], a] = True
                    # saturate (only used where no action fits, see below)
                    M2s = M2.copy()
                    over = M2s.sum(1) - Lm
                    for _ in range(int(over.max()) if over.max() > 0 else 0):
                        o = M2s.sum(1) > Lm
                        j = np.argmax(M2s, 1)
                        M2s[o, j[o]] -= 1
                    L2 = M2s.sum(1).astype(float)
                    rho = self.rho(L2)
                    base = self.bi[b2] * 3
                    for j in range(3):
                        # from the loop: language j leaves the loop
                        Mj = M2s.copy()
                        Mj[:, j] = np.maximum(Mj[:, j] - 1, 0)
                        pj = np.where(L2 > 0, rho * M2s[:, j] / np.maximum(L2, 1), 0.0)
                        NXT[rows, a, j] = (base + j) * nM + look[Mj[:, 0], Mj[:, 1], Mj[:, 2]]
                        P[rows, a, j] = pj
                        # from the source
                        NXT[rows, a, 3 + j] = (base + j) * nM + look[M2s[:, 0], M2s[:, 1], M2s[:, 2]]
                        P[rows, a, 3 + j] = (1 - np.where(L2 > 0, rho, 0.0)) * self.p[j]
                    R[rows, a] = CAP if done else 0.0
                # no feasible action fits into the loop: allow LOOP with saturation
                none = ~FEAS[rows].any(1)
                FEAS[rows[none], A_LOOP] = True
        self.NXT, self.P, self.R, self.FEAS = NXT, P, R, FEAS
        self.build_time = time.time() - t0

    def solve(self, tol=1e-7, max_iter=20_000, tau=0.5, verbose=True):
        """Relative value iteration (aperiodicity transform tau). Returns rate (games per decision), h, policy."""
        h = np.zeros(self.N)
        ref = self.index(0, 0, 0, 0, 1, (0, 0, 0))
        t0 = time.time()
        for it in range(max_iter):
            Q = self.R + np.einsum("sao,sao->sa", self.P, h[self.NXT])
            Q = np.where(self.FEAS, Q, -np.inf)
            Th = Q.max(1)
            g = Th[ref] - h[ref]
            h_new = (1 - tau) * h + tau * (Th - Th[ref])
            span = np.max(Th - h) - np.min(Th - h)
            h = h_new
            if verbose and it % 200 == 0:
                print(f"  RVI {it}: rate {g:.6f}, span {span:.2e}", flush=True)
            if span < tol:
                break
        pol = Q.argmax(1).astype(np.int8)
        return dict(rate=float(g), h=h, policy=pol, iterations=it + 1, runtime=time.time() - t0)

    def policy_rate(self, pol, tol=1e-8, max_iter=50_000, tau=0.5):
        """Long-run games per decision of a fixed policy (relative value iteration on the policy)."""
        h = np.zeros(self.N)
        ref = self.index(0, 0, 0, 0, 1, (0, 0, 0))
        r = np.arange(self.N)
        Rp, Pp, Np = self.R[r, pol], self.P[r, pol], self.NXT[r, pol]
        for _ in range(max_iter):
            Th = Rp + np.einsum("so,so->s", Pp, h[Np])
            g = Th[ref] - h[ref]
            span = np.max(Th - h) - np.min(Th - h)
            h = (1 - tau) * h + tau * (Th - Th[ref])
            if span < tol:
                break
        return float(g)
