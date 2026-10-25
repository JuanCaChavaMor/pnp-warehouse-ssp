"""
mock_plantsim.py
----------------
A stand-in for the Plant Simulation model behind the COM interface, for tests without Windows and for a first look
at what the MDP leaves out.

It answers the COM calls PlantSimClient makes (GetValue, SetValue, ExecuteSimTalk, ...) and reproduces the logic of
the SimTalk methods reset, pnpDecision/GetCurrentState and AIControl at the level of decisions:
    * the next game at the robot comes from the source (shares p) or, if one is due, from the loop; when both are
      available, merge="alternate" (default) takes them in turn, merge="loop_first" always serves the loop first
      (then the source can be starved: once as many games circulate as the loop delay, the robot sees only
      returning games);
    * LOOP sends the game around the loop, an eviction sends the whole partial batch around the loop;
    * games return to the robot `loop_delay` decisions after they were sent (FIFO);
    * 10 equal games in a buffer go to the sink.
The MDP of mdp.py is the special case in which looped and evicted games never come back (loop_delay = None).
Time is counted in robot decisions (sim_time = number of decisions so far).
"""
from __future__ import annotations

import re
from collections import deque

import numpy as np

NAMES = ("DE", "EN", "ES")


class MockPlantSim:
    def __init__(self, p=(0.4, 0.35, 0.25), loop_delay: int | None = 8, frame: str = ".Modelle.Modell",
                 merge: str = "alternate"):
        if merge not in ("alternate", "loop_first"):
            raise ValueError("merge must be 'alternate' or 'loop_first'")
        self.p, self.loop_delay, self.frame, self.merge = np.asarray(p, float), loop_delay, frame, merge
        self.values = {"SeedRL": 1, "GoalGames": 1000}
        self.state, self.action = {}, ""
        self.calls = []

    # ---------------------------------------------------------------- COM surface used by PlantSimClient
    def _name(self, path):
        assert path.startswith(self.frame + "."), f"path outside the frame: {path}"
        return path[len(self.frame) + 1:]

    def GetValue(self, path):
        m = re.fullmatch(r'(\w+)\["(\w+)",1\]', self._name(path))
        if m:
            table, col = m.groups()
            return self.state[col] if table == "CurrentState" else self.action
        return self.values[self._name(path)]

    def SetValue(self, path, value):
        name = self._name(path)
        if name == 'ActionControl["action",1]':
            self.action = value
        else:
            self.values[name] = value

    def ExecuteSimTalk(self, code):
        name = self._name(code)
        self.calls.append(name)
        {"reset": self._reset, "AIControl": self._ai_control}[name]()

    def SetVisible(self, v):
        pass

    def CloseModel(self):
        pass

    def Quit(self):
        pass

    # ---------------------------------------------------------------- model logic (mirrors the SimTalk methods)
    def _reset(self):
        self.rng = np.random.default_rng(int(self.values["SeedRL"]))
        self.buf = {"A": [], "B": []}
        self.loop = deque()                  # (due decision, language)
        self.sink, self.t, self.robot, self.last_from_loop = 0, 0, None, False
        self.state = dict(tipo_actual="", ocup_A=0, tipo_A="", ocup_B=0, tipo_B="", len_loop=0, goal_state=0,
                          sim_time=0, event_id=0, did_loop=0, n_evicted=0, blocked=0, completed_batch=0, batches=0,
                          loop_DE=0, loop_EN=0, loop_ES=0)
        self.action = ""
        self._next_decision()

    def _send_around(self, games):
        if self.loop_delay is not None:
            for g in games:
                self.loop.append((self.t + self.loop_delay, g))

    def _next_decision(self):
        self.t += 1
        due = bool(self.loop) and self.loop[0][0] <= self.t
        if due and (self.merge == "loop_first" or not self.last_from_loop):
            self.robot = self.loop.popleft()[1]
            self.last_from_loop = True
        else:
            self.robot = NAMES[self.rng.choice(3, p=self.p)]
            self.last_from_loop = False
        self._get_current_state()
        self.state["event_id"] += 1                                  # pnpDecision

    def _get_current_state(self):
        s = self.state
        s["tipo_actual"] = self.robot or ""
        for b in ("A", "B"):
            s[f"ocup_{b}"] = len(self.buf[b])
            s[f"tipo_{b}"] = self.buf[b][0] if self.buf[b] else ""
        s["len_loop"] = len(self.loop)
        for name in NAMES:
            s[f"loop_{name}"] = sum(g == name for _, g in self.loop)
        s["batches"] = self.sink / 10
        s["goal_state"] = int(self.sink >= self.values["GoalGames"])
        s["sim_time"] = self.t

    def _ai_control(self):
        act, game, s = self.action, self.robot, self.state
        for f in ("did_loop", "n_evicted", "blocked", "completed_batch"):
            s[f] = 0
        if game is None or act not in ("A", "B", "LOOP"):
            return                                                   # like AIControl: no restart
        if act == "LOOP":
            s["did_loop"] = 1
            self._send_around([game])
        else:
            buf = self.buf[act]
            if buf and buf[0] != game:
                s["n_evicted"] = len(buf)
                self._send_around(buf)
                buf.clear()
            buf.append(game)
            if len(buf) >= 10:
                s["completed_batch"] = 1
                self.sink += len(buf)
                buf.clear()
        self.action, self.robot = "", None
        self._next_decision()
