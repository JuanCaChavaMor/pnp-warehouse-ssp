"""
mdp.py
------
The warehouse sorting problem as a stochastic shortest-path (SSP) problem: state encoding, feasible actions,
transition f(s, a, l) and stage cost c(s, a).

State tuple: s = (q1, n1, q2, n2, lam, k)
    q_i  in {0,1,2,3}   type stored in storage i: 0 = empty, 1 = DE, 2 = EN, 3 = ES
    n_i  in {0,...,9}   units in storage i (the 10th unit triggers the flush, so 10 is never stored)
    lam  in {1,2,3}     type of the game waiting at the pick-and-place robot
    k    in {0,...,100} completed flushes (10 games each); k = 100 <=> 1000 games stored

Actions: 0 = A (bufferID1), 1 = B (bufferID2), 2 = LOOP (Return_to_mainCon)

Names and parameters follow the Plant Simulation model (plantsim/): the language of a game is the Name of its
manual ("DE" | "EN" | "ES"), the shares are those of the SystemLoad table, and the action strings are the values
Python writes to ActionControl["action",1].
"""

import numpy as np
from typing import List, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────
N_TYPES     = 3          # DE, EN, ES
CAP         = 10         # flush threshold (Plant Simulation: buffer capacity 10)
N_FLUSHES   = 100        # goal: 100 flush events  ->  1000 games (Plant Simulation: GoalGames = 1000)
ARRIVAL_P   = np.array([0.4, 0.35, 0.25])   # p(DE), p(EN), p(ES)  (SystemLoad: Haeufigkeit)

# Identities in the Plant Simulation model
TYPE_NAMES    = {0: "", 1: "DE", 2: "EN", 3: "ES"}             # Name of the game (= its manual)
TYPE_CODES    = {v: k for k, v in TYPE_NAMES.items()}
STORAGE_NAMES = ("bufferID1", "bufferID2")                        # intermediate storages S1, S2

# Cost parameters
C_ROUTE = 1.0
C_LOOP  = 2.0
C_LIN   = 1.5

# Actions
A_S1   = 0
A_S2   = 1
A_LOOP = 2
ACTIONS = [A_S1, A_S2, A_LOOP]
ACTION_NAMES = {A_S1: "A", A_S2: "B", A_LOOP: "LOOP"}            # ActionControl["action",1]

# Product types
EMPTY = 0                # q_i = 0  means storage is empty
TYPES = [1, 2, 3]        # DE=1, EN=2, ES=3


# ── State validation ───────────────────────────────────────────────────────────
def is_valid_state(q1: int, n1: int, q2: int, n2: int,
                   lam: int, k: int) -> bool:
    """Return True iff the state satisfies all structural constraints."""
    # Consistency: empty ↔ zero units
    if (q1 == EMPTY) != (n1 == 0):
        return False
    if (q2 == EMPTY) != (n2 == 0):
        return False
    # Unit counts within capacity (flush resets to 0 before state is stored)
    if not (0 <= n1 <= CAP - 1 or (q1 == EMPTY and n1 == 0)):
        return False
    if not (0 <= n2 <= CAP - 1 or (q2 == EMPTY and n2 == 0)):
        return False
    # Exclusivity: both occupied → different types
    if q1 != EMPTY and q2 != EMPTY and q1 == q2:
        return False
    # Valid type values
    if q1 not in [EMPTY] + TYPES:
        return False
    if q2 not in [EMPTY] + TYPES:
        return False
    if lam not in TYPES:
        return False
    # Flush counter
    if not (0 <= k <= N_FLUSHES):
        return False
    return True


# ── State enumeration & indexing ───────────────────────────────────────────────
def enumerate_states() -> List[Tuple]:
    """
    Return a list of all valid state tuples (q1,n1,q2,n2,lam,k).
    Also builds and returns the index lookup dict.
    """
    states = []
    for q1 in range(4):           # 0..3
        for n1 in range(CAP):     # 0..9  (n=10 triggers flush → reset to 0)
            for q2 in range(4):
                for n2 in range(CAP):
                    for lam in TYPES:
                        for k in range(N_FLUSHES + 1):
                            if is_valid_state(q1, n1, q2, n2, lam, k):
                                states.append((q1, n1, q2, n2, lam, k))
    return states


def build_index(states: List[Tuple]) -> dict:
    """Return {state_tuple: integer_index} mapping."""
    return {s: i for i, s in enumerate(states)}


# ── Feasible actions ───────────────────────────────────────────────────────────
def feasible_actions(q1: int, n1: int, q2: int, n2: int,
                     lam: int) -> List[int]:
    """
    Return list of feasible actions for the given storage configuration
    and incoming product type lam.

    Rules:
        Route to S_i is feasible iff:
        (a) q_i == lam                          → append to matching type
        (b) q_i == EMPTY AND q_j != lam         → use empty slot only if
                                                   other side doesn't hold lam
        (c) q_i != EMPTY AND q_i != lam
            AND q_j != lam                      → eviction safe only if
                                                   other side doesn't hold lam
        Loop is ALWAYS feasible.
    """
    actions = []

    # ── S1 ─────────────────────────────────────────────────────────────────────
    if q1 == lam:
        actions.append(A_S1)                        # (a)
    elif q1 == EMPTY and q2 != lam:
        actions.append(A_S1)                        # (b)
    elif q1 != EMPTY and q1 != lam and q2 != lam:
        actions.append(A_S1)                        # (c)

    # ── S2 ─────────────────────────────────────────────────────────────────────
    if q2 == lam:
        actions.append(A_S2)                        # (a)
    elif q2 == EMPTY and q1 != lam:
        actions.append(A_S2)                        # (b)
    elif q2 != EMPTY and q2 != lam and q1 != lam:
        actions.append(A_S2)                        # (c)

    # ── Loop (always feasible) ─────────────────────────────────────────────────
    actions.append(A_LOOP)

    return actions



# ── Transition function ────────────────────────────────────────────────────────
    
def transition(q1: int, n1: int, q2: int, n2: int,
               lam: int, k: int,
               action: int,
               next_lam: int) -> Tuple:
    """
        Deterministic state update f(s, a) followed by stochastic next arrival.
    
        Returns next state tuple (q1', n1', q2', n2', next_lam, k').
    
        Cases implemented per Section 2.4 of the formulation:
            Case 1 — route to S_i, no eviction
            Case 2 — route to S_i, eviction triggered
            Case 3 — loop
    """

    q1_, n1_, q2_, n2_, k_ = q1, n1, q2, n2, k

    if action == A_LOOP:
        pass

    elif action == A_S1:
        if q1 == lam or q1 == EMPTY:
            # Case 1: no eviction
            if n1 + 1 < CAP:
                q1_ = lam
                n1_ = n1 + 1
            else:
                # Flush
                q1_ = EMPTY
                n1_ = 0
                k_  = k + 1
        else:
            # Case 2: eviction in S1
            q1_ = lam
            n1_ = 1

    elif action == A_S2:
        if q2 == lam or q2 == EMPTY:
            # Case 1: no eviction
            if n2 + 1 < CAP:
                q2_ = lam
                n2_ = n2 + 1
            else:
                # Flush
                q2_ = EMPTY
                n2_ = 0
                k_  = k + 1
        else:
            # Case 2: eviction in S2
            q2_ = lam
            n2_ = 1

    # ── Safety assertion: exclusivity must hold in next state ─────────────────
    assert not (q1_ != EMPTY and q2_ != EMPTY and q1_ == q2_), (
        f"Exclusivity violated in transition!\n"
        f"  Input:  q1={q1}, n1={n1}, q2={q2}, n2={n2}, "
        f"lam={lam}, action={action}\n"
        f"  Output: q1_={q1_}, q2_={q2_}"
    )

    return (q1_, n1_, q2_, n2_, next_lam, k_)



# ── Cost function ──────────────────────────────────────────────────────────────
def cost(q1: int, n1: int, q2: int, n2: int,
         lam: int, action: int) -> float:
    """
    Stage cost c(s, a):
        c_route  (always)
      + c_loop   (if action == loop)
      + C_lin * n_i  (if eviction triggered in S_i)
    """
    c = C_ROUTE

    if action == A_LOOP:
        c += C_LOOP

    elif action == A_S1:
        # Eviction: S1 is occupied by a different type
        if q1 != EMPTY and q1 != lam:
            c += C_LIN * n1

    elif action == A_S2:
        # Eviction: S2 is occupied by a different type
        if q2 != EMPTY and q2 != lam:
            c += C_LIN * n2

    return c


# ── Goal check ─────────────────────────────────────────────────────────────────
def is_goal(k: int) -> bool:
    """True iff the flush counter has reached the target."""
    return k >= N_FLUSHES


# ── Integer encoding (for O(1) vectorised lookups) ─────────────────────────────
# Mixed-radix code over the *superset* box  4 x CAP x 4 x CAP x 3 x (N_FLUSHES+1).
# Invalid combinations simply map to -1 in code2idx.
N_CODES = 4 * CAP * 4 * CAP * N_TYPES * (N_FLUSHES + 1)


def encode(q1, n1, q2, n2, lam, k):
    """State tuple -> integer code in [0, N_CODES). Works on ints or arrays."""
    return ((((q1 * CAP + n1) * 4 + q2) * CAP + n2) * N_TYPES + (lam - 1)) \
        * (N_FLUSHES + 1) + k


# ── Precomputed tables for vectorised solvers ──────────────────────────────────
def build_tables(states: List[Tuple], index: dict) -> dict:
    """
    Precompute, once, everything VI and Q-Learning need per (state, action):

        C[s, a]        stage cost c(s,a); +inf if a infeasible in s
        NXT[s, a, l]   index of f(s, a, l) (next arrival l = DE, EN, ES);
                       0 for infeasible (masked by C = inf anyway)
        FEAS[s, a]     bool feasibility mask
        K[s]           flush counter of state s (for layered VI)
        code2idx       int array of size N_CODES, -1 for invalid codes

    Cost: |S| * 3 * 3 transition calls, a few seconds, done once.
    """
    N    = len(states)
    C    = np.full((N, 3), np.inf, dtype=np.float64)
    NXT  = np.zeros((N, 3, N_TYPES), dtype=np.int64)
    FEAS = np.zeros((N, 3), dtype=bool)
    K    = np.array([s[5] for s in states], dtype=np.int64)

    for i, (q1, n1, q2, n2, lam, k) in enumerate(states):
        if is_goal(k):
            continue                      # absorbing: no actions needed
        for a in feasible_actions(q1, n1, q2, n2, lam):
            FEAS[i, a] = True
            C[i, a]    = cost(q1, n1, q2, n2, lam, a)
            for li, l in enumerate(TYPES):
                NXT[i, a, li] = index[transition(q1, n1, q2, n2, lam, k, a, l)]

    code2idx = np.full(N_CODES, -1, dtype=np.int64)
    codes    = encode(*np.array(states, dtype=np.int64).T)
    code2idx[codes] = np.arange(N)

    return {"C": C, "NXT": NXT, "FEAS": FEAS, "K": K, "code2idx": code2idx}

