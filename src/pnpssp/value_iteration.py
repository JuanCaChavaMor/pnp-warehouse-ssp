"""
value_iteration.py
------------------
Exact SSP solver via Value Iteration.  Produces V*(s) and pi*(s).

Two modes:
    'layered' (default) — backward induction over the flush counter k.
        k never decreases, and transitions from layer k go only to layers
        k or k+1.  So V* on layer k+1 is *fixed* input when solving layer k:
        the 200k-state SSP decomposes into N_FLUSHES tiny SSPs (~1.6k states
        each), each of which converges in a few dozen sweeps.  Exact.
    'full' — plain synchronous VI over the whole space (vectorised).
        Slower (information must travel ~10 sweeps per flush, ~1300 sweeps
        total) but useful as a cross-check.

Both are fully vectorised with numpy using the tables from mdp.build_tables.
"""

import numpy as np
import time
from .mdp import enumerate_states, build_index, build_tables, ARRIVAL_P, N_FLUSHES


def _bellman(V, C, NXT, p, gamma=1.0):
    """
    One Bellman backup for the rows described by (C, NXT):
        Q[s,a] = C[s,a] + gamma * sum_l p_l V[NXT[s,a,l]]
    Returns (Q, V_new, argmin).
    """
    EV = np.einsum('l,sal->sa', p, V[NXT])       # expected next value
    Q  = C + gamma * EV                          # inf where infeasible
    return Q, Q.min(axis=1), Q.argmin(axis=1)


def solve(epsilon: float = 1e-6,
          max_iter: int = 10_000,
          method: str = "layered",
          gamma: float = 1.0,
          verbose: bool = True,
          tables: dict = None,
          states: list = None,
          index: dict = None) -> dict:
    """
    Run Value Iteration to convergence.
    gamma < 1 solves the *discounted* problem (same layered scheme applies:
    k still never decreases).  gamma = 1 is the undiscounted SSP.

    Returns dict with keys
        'V', 'policy', 'states', 'index', 'tables', 'n_iter', 'runtime'
    """
    t0 = time.time()

    if states is None:
        if verbose:
            print("Enumerating states...", end=" ", flush=True)
        states = enumerate_states()
        index  = build_index(states)
        if verbose:
            print(f"{len(states):,} states found.")
    if tables is None:
        if verbose:
            print("Building transition tables...", end=" ", flush=True)
        tables = build_tables(states, index)
        if verbose:
            print(f"done ({time.time()-t0:.1f}s).")

    C, NXT, K = tables["C"], tables["NXT"], tables["K"]
    p = np.asarray(ARRIVAL_P, dtype=np.float64)
    N = len(states)

    V      = np.zeros(N, dtype=np.float64)       # goal layer stays 0
    policy = np.full(N, -1, dtype=np.int32)
    total_iter = 0

    if method == "layered":
        # ── Backward induction over k = N_FLUSHES-1 ... 0 ─────────────────────
        for k in range(N_FLUSHES - 1, -1, -1):
            rows = np.flatnonzero(K == k)
            Ck, NXTk = C[rows], NXT[rows]
            # Warm start: V on layer k ≈ V on layer k+1 (same storage config).
            # enumerate_states() has k innermost, so rows align element-wise.
            V[rows] = V[np.flatnonzero(K == k + 1)]
            for it in range(1, max_iter + 1):
                _, Vk_new, _ = _bellman(V, Ck, NXTk, p, gamma)
                delta = np.max(np.abs(Vk_new - V[rows]))
                V[rows] = Vk_new
                if delta < epsilon:
                    break
            total_iter += it
            if verbose and (k % 20 == 0 or k == N_FLUSHES - 1):
                print(f"  layer k={k:3d} | {it:3d} sweeps | "
                      f"V*(empty,DE,k) = {V[index[(0,0,0,0,1,k)]]:9.3f}")
        # Final greedy extraction over the whole space
        _, _, policy_all = _bellman(V, C, NXT, p, gamma)
        nongoal = K < N_FLUSHES
        policy[nongoal] = policy_all[nongoal]

    elif method == "full":
        nongoal = K < N_FLUSHES
        rows    = np.flatnonzero(nongoal)
        Cr, NXTr = C[rows], NXT[rows]
        for it in range(1, max_iter + 1):
            _, V_new, pol = _bellman(V, Cr, NXTr, p, gamma)
            delta   = np.max(np.abs(V_new - V[rows]))
            V[rows] = V_new
            if verbose and (it % 100 == 0 or it == 1):
                print(f"  Iter {it:4d} | Δ = {delta:.2e}")
            if delta < epsilon:
                break
        policy[rows] = pol
        total_iter = it
    else:
        raise ValueError("method must be 'layered' or 'full'")

    runtime = time.time() - t0
    if verbose:
        print(f"✅ VI ({method}) converged, {total_iter} total sweeps, "
              f"⏱ {runtime:.2f}s")

    return {
        "V"       : V,
        "policy"  : policy,
        "states"  : states,
        "index"   : index,
        "tables"  : tables,
        "n_iter"  : total_iter,
        "runtime" : runtime,
    }


def solve_aggregated(tables: dict, index: dict, states: list, gamma: float,
                     epsilon: float = 1e-8, max_iter: int = 100_000) -> np.ndarray:
    """
    Optimal policy of the k-free, infinite-horizon discounted problem
    (flush = stay in the same layer), broadcast over all k.  This is the exact
    target of a Q-Learning agent with aggregate_k=True, and policy_value(., 1.0)
    of it gives the exact price of ignoring k on the true SSP.
    """
    from .mdp import encode
    C, NXT, K = tables["C"], tables["NXT"], tables["K"]
    p = np.asarray(ARRIVAL_P, dtype=np.float64)
    st  = np.array(states, dtype=np.int64)
    k0  = tables["code2idx"][encode(st[:,0], st[:,1], st[:,2], st[:,3], st[:,4], np.zeros(len(st), dtype=np.int64))]
    rows = np.flatnonzero(K == 0)
    C0, NXT0 = C[rows], k0[NXT[rows]]                 # successors remapped to layer 0
    V = np.zeros(len(states))
    for _ in range(max_iter):
        _, Vn, pol = _bellman(V, C0, NXT0, p, gamma)
        d = np.max(np.abs(Vn - V[rows])); V[rows] = Vn
        if d < epsilon: break
    policy = np.full(len(states), -1, dtype=np.int32)
    policy[rows] = pol
    policy = policy[k0]                                # broadcast over k
    policy[K >= N_FLUSHES] = -1
    return policy


def policy_value(policy: np.ndarray, tables: dict, gamma: float = 1.0,
                 epsilon: float = 1e-6, max_iter: int = 10_000) -> np.ndarray:
    """
    Exact policy evaluation: J^pi(s) = expected (gamma-discounted) cost-to-go
    of following the FIXED policy pi.  Same layered scheme as solve().
    Use with gamma=1 to get the true throughput KPI of any policy — e.g. the
    undiscounted cost of a policy that was optimised with gamma < 1.
    """
    C, NXT, K = tables["C"], tables["NXT"], tables["K"]
    p = np.asarray(ARRIVAL_P, dtype=np.float64)
    J = np.zeros(len(policy), dtype=np.float64)
    for k in range(N_FLUSHES - 1, -1, -1):
        rows = np.flatnonzero(K == k)
        a    = policy[rows]
        Ck   = C[rows, a]                                   # (n_k,)
        NXTk = NXT[rows, a]                                 # (n_k, 3)
        assert np.all(np.isfinite(Ck)), "policy uses an infeasible action"
        J[rows] = J[np.flatnonzero(K == k + 1)]             # warm start
        for _ in range(max_iter):
            Jk = Ck + gamma * (J[NXTk] @ p)
            delta = np.max(np.abs(Jk - J[rows]))
            J[rows] = Jk
            if delta < epsilon:
                break
    return J


def expected_initial_value(V: np.ndarray, index: dict) -> float:
    """E_lambda[ V*(empty, empty, lambda, k=0) ] — the true V*(s0)."""
    return float(sum(p * V[index[(0, 0, 0, 0, l, 0)]]
                     for p, l in zip(ARRIVAL_P, (1, 2, 3))))
