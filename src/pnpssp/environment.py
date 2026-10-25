"""
environment.py
--------------
WarehouseEnv: one Gym-like interface for the warehouse SSP with two backends.

    'python'     the transition function of mdp.py (fast, exact; arrivals i.i.d. with ARRIVAL_P)
    'plantsim'   the Plant Simulation model through COM (plantsim_com.PlantSimClient), following the protocol of
                 the SimTalk methods in plantsim/methods: reset -> decision -> AIControl -> decision ...

Both return states in the MDP encoding (q1, n1, q2, n2, lam, k) and the SSP stage cost c(s, a), so a policy or a
Q-table trained on one backend can be evaluated on the other. The Plant Simulation backend also reports the
simulated time (`info["sim_time"]`, seconds), which is the throughput time the project asks to minimise.
"""

from typing import List, Optional, Tuple

import numpy as np

from .mdp import (ACTION_NAMES, ARRIVAL_P, EMPTY, N_FLUSHES, TYPES, TYPE_NAMES, cost, feasible_actions, is_goal,
                  transition)


class WarehouseEnv:
    """
    reset()               -> s0
    step(action)          -> (s', stage_cost, done); details of the last step in self.info
    feasible_actions(s)   -> feasible action indices
    """

    def __init__(self, backend: str = "python", arrival_probs: Optional[np.ndarray] = None,
                 seed: Optional[int] = None, client=None):
        """
        backend       : 'python' or 'plantsim'
        arrival_probs : python backend only; default ARRIVAL_P = (0.4, 0.35, 0.25) for (DE, EN, ES)
        seed          : python backend: numpy seed; plantsim backend: written to SeedRL before each reset
        client        : plantsim backend: a connected plantsim_com.PlantSimClient (or a compatible test double)
        """
        assert backend in ("python", "plantsim"), "backend must be 'python' or 'plantsim'"
        if backend == "plantsim" and client is None:
            raise ValueError("backend='plantsim' needs a PlantSimClient (see plantsim_com.py)")
        self.backend = backend
        self.arrival_probs = arrival_probs if arrival_probs is not None else ARRIVAL_P.copy()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.client = client
        self.q1 = self.n1 = self.q2 = self.n2 = 0
        self.lam, self.k = 1, 0
        self.info = {}
        self._episode = 0

    # ---------------------------------------------------------------- public interface
    def reset(self) -> Tuple:
        """New episode: both storages empty, k = 0, first game at the robot."""
        self.info = {}
        if self.backend == "plantsim":
            seed = None if self.seed is None else self.seed + self._episode
            self._episode += 1
            self._set(self.client.reset(seed))
        else:
            self.q1 = self.n1 = self.q2 = self.n2 = 0
            self.k = 0
            self.lam = self._sample_arrival()
        return self._state()

    def step(self, action: int) -> Tuple[Tuple, float, bool]:
        """Apply action (0 = A, 1 = B, 2 = LOOP); return (s', c(s, a), done)."""
        s = self._state()
        if action not in feasible_actions(*s[:5]):
            raise ValueError(f"infeasible action {ACTION_NAMES.get(action, action)} in state {s}")
        stage_cost = cost(*s[:5], action)
        if self.backend == "python":
            self._set(transition(*s, action, self._sample_arrival()))
        else:
            row = self.client.step(ACTION_NAMES[action])
            self.info = row
            self._set(row["state"])
        return self._state(), stage_cost, is_goal(self.k)

    def feasible_actions(self, state: Optional[Tuple] = None) -> List[int]:
        q1, n1, q2, n2, lam, _ = self._state() if state is None else state
        return feasible_actions(q1, n1, q2, n2, lam)

    def render(self) -> str:
        name = {**TYPE_NAMES, 0: "-"}
        return (f"bufferID1=[{name[self.q1]}, {self.n1}]  bufferID2=[{name[self.q2]}, {self.n2}]  "
                f"robot={name[self.lam]}  flushes={self.k}/{N_FLUSHES}")

    # ---------------------------------------------------------------- helpers
    def _set(self, s):
        if isinstance(s, dict):
            s = s["state"]
        self.q1, self.n1, self.q2, self.n2, self.lam, self.k = s

    def _state(self) -> Tuple:
        return (self.q1, self.n1, self.q2, self.n2, self.lam, self.k)

    def _sample_arrival(self) -> int:
        return int(self.rng.choice(TYPES, p=self.arrival_probs))
