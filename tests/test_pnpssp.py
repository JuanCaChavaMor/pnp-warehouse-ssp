import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnpssp.environment import WarehouseEnv
from pnpssp.mdp import (A_LOOP, A_S1, A_S2, ACTION_NAMES, N_FLUSHES, TYPE_NAMES, build_index, cost, enumerate_states,
                        feasible_actions, transition)
from pnpssp.mock_plantsim import MockPlantSim
from pnpssp.plantsim_com import PlantSimClient
from pnpssp.policies import evaluate_policy, greedy_policy
from pnpssp.value_iteration import expected_initial_value, policy_value, solve


@pytest.fixture(scope="module")
def vi():
    return solve(verbose=False)


def test_identities_match_plant_simulation():
    assert TYPE_NAMES == {0: "", 1: "DE", 2: "EN", 3: "ES"}
    assert ACTION_NAMES == {A_S1: "A", A_S2: "B", A_LOOP: "LOOP"}


def test_state_space_size():
    # storage configurations: both empty + one occupied (2 sides x 3 types x 9 counts) + both occupied, different
    # types (3 x 9 x 2 x 9) = 1 + 54 + 486 = 541; times 3 arrivals and 101 values of k
    assert len(enumerate_states()) == 541 * 3 * 101 == 163_923


def test_transitions_and_costs():
    # eviction: S1 holds 4 EN, a DE arrives and goes to S1 -> 4 games back to the loop, S1 = [DE, 1]
    assert transition(2, 4, 0, 0, 1, 7, A_S1, 3) == (1, 1, 0, 0, 3, 7)
    assert cost(2, 4, 0, 0, 1, A_S1) == 1.0 + 1.5 * 4
    # flush: the 10th DE completes the batch
    assert transition(1, 9, 2, 3, 1, 7, A_S1, 2) == (0, 0, 2, 3, 2, 8)
    assert cost(1, 9, 2, 3, 1, A_LOOP) == 3.0
    # exclusivity: an empty S2 may not receive DE while S1 holds DE
    assert A_S2 not in feasible_actions(1, 3, 0, 0, 1)


def test_value_iteration_layered_equals_full_and_beats_greedy(vi):
    full = solve(method="full", verbose=False, tables=vi["tables"], states=vi["states"], index=vi["index"])
    assert np.max(np.abs(vi["V"] - full["V"])) < 1e-4
    T, idx = vi["tables"], vi["index"]
    v = expected_initial_value(vi["V"], idx)
    g = expected_initial_value(policy_value(greedy_policy(vi["states"]), T, 1.0), idx)
    assert v < g
    assert abs(expected_initial_value(policy_value(vi["policy"], T, 1.0), idx) - v) < 1e-3


def test_policy_evaluation_matches_simulation(vi):
    pol, idx = vi["policy"], vi["index"]
    m = evaluate_policy(pol, idx, WarehouseEnv("python", seed=3), n_eval=150)
    v = expected_initial_value(vi["V"], idx)
    assert abs(m["mean_cost"] - v) < 4 * m["std_cost"] / np.sqrt(150)


def test_com_protocol_with_mock_reproduces_the_mdp(vi):
    """Without recirculation the mock behaves like the MDP: same transitions, same flush counter."""
    mock = MockPlantSim(loop_delay=None)
    env = WarehouseEnv("plantsim", client=PlantSimClient(com=mock, poll=0), seed=11)
    s = env.reset()
    assert mock.calls == ["reset"] and s[:4] == (0, 0, 0, 0)
    pol, idx = vi["policy"], vi["index"]
    steps, done = 0, False
    while not done:
        q1, n1, q2, n2, lam, k = s
        a = int(pol[idx[s]])
        s2, c, done = env.step(a)
        # the deterministic part of the transition must agree with mdp.transition
        assert transition(q1, n1, q2, n2, lam, k, a, s2[4]) == s2
        assert env.info["did_loop"] == (a == A_LOOP)
        s, steps = s2, steps + 1
    assert s[5] == N_FLUSHES and mock.sink == 1000 and env.info["goal_state"] == 1
    assert mock.calls.count("AIControl") == steps


def test_client_rejects_bad_input():
    mock = MockPlantSim()
    cl = PlantSimClient(com=mock, poll=0)
    cl.reset(5)
    with pytest.raises(ValueError):
        cl.step("C")
    assert PlantSimClient.to_mdp_state(dict(tipo_A="DE", ocup_A=3, tipo_B="", ocup_B=0, tipo_actual="ES",
                                            batches=4.0)) == (1, 3, 0, 0, 3, 4)
    with pytest.raises(ValueError):
        PlantSimClient.to_mdp_state(dict(tipo_A="FR", ocup_A=1, tipo_B="", ocup_B=0, tipo_actual="DE", batches=0))


def test_mock_recirculation_returns_looped_games():
    mock = MockPlantSim(loop_delay=2)
    cl = PlantSimClient(com=mock, poll=0)
    first = cl.reset(1)["tipo_actual"]
    cl.step("LOOP")
    cl.step("LOOP")
    assert cl.read_row()["tipo_actual"] == first          # back at the robot two decisions later
