"""
plantsim_com.py
---------------
Python side of the decision protocol implemented by the SimTalk methods in plantsim/methods (Windows, pywin32).

Protocol (one decision = one game ready to leave the PickAndPlace robot):
    1. pnpDecision (exit control of PickAndPlace) calls GetCurrentState, increments CurrentState["event_id",1]
       and stops the event controller. The game waits in the robot.
    2. Python reads the row of CurrentState, writes ActionControl["action",1] := "A" | "B" | "LOOP" and executes
       AIControl.
    3. AIControl applies the action (eviction of a partial batch of another language back to the loop, flush of
       10 equal games to the sink), writes did_loop / n_evicted / completed_batch and restarts the event
       controller; the simulation runs until the next game reaches the robot -> 1.
    reset sets the seed (variable SeedRL), resets the model and runs it to the first decision.

The client converts the row into the MDP state (q1, n1, q2, n2, lam, k) of mdp.py:
    q1 = tipo_A (bufferID1), n1 = ocup_A, q2 = tipo_B (bufferID2), n2 = ocup_B, lam = tipo_actual,
    k  = completed batches = floor(sink count / 10).

Untested against a live Plant Simulation in this repository (no Windows licence in CI); the protocol is tested with
a simulated COM object in tests/test_pnpssp.py.
"""
from __future__ import annotations

import math
import time

from .mdp import TYPE_CODES

FIELDS = ("tipo_actual", "ocup_A", "tipo_A", "ocup_B", "tipo_B", "len_loop", "goal_state", "sim_time", "event_id",
          "did_loop", "n_evicted", "blocked", "completed_batch", "batches")


class PlantSimTimeout(RuntimeError):
    pass


class PlantSimClient:
    """
    PlantSimClient(model_path)                     open Plant Simulation through COM and load the model
    PlantSimClient(com=<object>)                   use an existing COM object (or a test double)

    frame : path of the frame that holds the methods and tables (German installation: ".Modelle.Modell";
            English: ".Models.Model")
    prog_id : COM ProgID; the version-independent "Tecnomatix.PlantSimulation.RemoteControl" picks the
              installed version, or give one explicitly, e.g. "Tecnomatix.PlantSimulation.RemoteControl.2404"
    """

    def __init__(self, model_path: str | None = None, frame: str = ".Modelle.Modell", com=None,
                 prog_id: str = "Tecnomatix.PlantSimulation.RemoteControl", visible: bool = False,
                 timeout: float = 60.0, poll: float = 0.005):
        self.frame, self.timeout, self.poll = frame, timeout, poll
        if com is None:
            import win32com.client                       # pywin32, Windows only
            com = win32com.client.Dispatch(prog_id)
            com.SetVisible(visible)
            if model_path:
                com.LoadModel(model_path)
        self.com = com
        self.event_id = 0

    # ---------------------------------------------------------------- low level
    def _p(self, name: str) -> str:
        return f"{self.frame}.{name}"

    def get(self, table: str, column: str):
        return self.com.GetValue(self._p(f'{table}["{column}",1]'))

    def set(self, path: str, value):
        self.com.SetValue(self._p(path), value)

    def run_method(self, name: str):
        self.com.ExecuteSimTalk(self._p(name))

    def read_row(self) -> dict:
        row = {f: self.get("CurrentState", f) for f in FIELDS}
        row["state"] = self.to_mdp_state(row)
        return row

    @staticmethod
    def to_mdp_state(row: dict) -> tuple:
        def code(name):
            name = (name or "").strip()
            if name not in TYPE_CODES:
                raise ValueError(f"unknown language '{name}' in CurrentState (expected DE, EN, ES)")
            return TYPE_CODES[name]
        k = int(math.floor(float(row["batches"]) + 1e-9))
        return (code(row["tipo_A"]), int(row["ocup_A"]), code(row["tipo_B"]), int(row["ocup_B"]),
                code(row["tipo_actual"]), k)

    def _wait_for_decision(self, previous: int) -> dict:
        """Wait until pnpDecision has written a new event (event_id > previous)."""
        t0 = time.monotonic()
        while True:
            eid = int(self.get("CurrentState", "event_id"))
            if eid > previous:
                self.event_id = eid
                return self.read_row()
            if time.monotonic() - t0 > self.timeout:
                raise PlantSimTimeout(f"no decision event after {self.timeout:.0f} s (event_id = {eid}); "
                                      "check the Console of Plant Simulation (AIControl prints invalid actions)")
            time.sleep(self.poll)

    # ---------------------------------------------------------------- protocol
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.set("SeedRL", int(seed))
        self.run_method("reset")                         # resets tables: event_id starts again from 0
        return self._wait_for_decision(0)

    def step(self, action: str) -> dict:
        if action not in ("A", "B", "LOOP"):
            raise ValueError(f"action must be A, B or LOOP, not {action!r}")
        self.com.SetValue(self._p('ActionControl["action",1]'), action)
        self.run_method("AIControl")
        return self._wait_for_decision(self.event_id)

    def close(self):
        try:
            self.com.CloseModel()
        finally:
            self.com.Quit()
