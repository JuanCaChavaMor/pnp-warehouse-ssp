"""
qlearning.py
------------
Tabular Q-Learning agent with optional epsilon-greedy exploration.
Backend-agnostic: works with any WarehouseEnv backend (sequential mode)
or with the vectorised PyTorch environment (parallel mode).

Design notes (vs. the previous version)
---------------------------------------
* Feasibility is enforced EVERYWHERE via a precomputed (|S|,3) bool mask:
    - greedy / exploratory action selection
    - the bootstrap target  min_{a' feasible} Q(s',a')
    - get_policy()
  Without this, infeasible actions keep Q=0 forever and the target collapses
  to c + 0, i.e. the agent only learns stage costs (never long-run values),
  and get_policy() can return infeasible actions (-> exclusivity assertion).
* State -> row index is an integer-code lookup tensor (mdp.encode) on the
  device.  No Python dict lookups, no per-env Python loops in the hot path.
"""

import numpy as np
import torch
import time
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm

from .mdp import (
    ACTIONS, N_FLUSHES, CAP, N_TYPES, EMPTY, ARRIVAL_P,
    C_ROUTE, C_LOOP, C_LIN, A_S1, A_S2, A_LOOP,
    encode, build_tables,
)
from .environment import WarehouseEnv


# ── Device selection ───────────────────────────────────────────────────────────
def get_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        print(f"🖥️  GPU detected: {props.name} "
              f"({props.total_memory / 1e9:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("🖥️  Using CPU")
    return device


# ══════════════════════════════════════════════════════════════════════════════
# VECTORISED PARALLEL ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class VectorisedWarehouseEnv:
    """
    N_envs independent warehouse episodes as one (N_envs, 6) long tensor
    with columns [q1, n1, q2, n2, lam, k].  Mirrors mdp.transition / mdp.cost
    exactly (same constants).  Assumes the caller only submits feasible actions.
    """

    def __init__(self,
                 n_envs: int,
                 arrival_probs: Optional[np.ndarray] = None,
                 device: Optional[torch.device] = None,
                 seed: Optional[int] = None):
        self.n_envs = n_envs
        self.device = device or get_device()
        self.probs  = torch.tensor(
            arrival_probs if arrival_probs is not None else ARRIVAL_P,
            dtype=torch.float32, device=self.device)
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.states = torch.zeros((n_envs, 6), dtype=torch.long,
                                  device=self.device)

    def reset(self) -> torch.Tensor:
        self.states.zero_()
        self.states[:, 4] = self._sample_arrivals()
        return self.states.clone()

    def reset_done(self, done_mask: torch.Tensor) -> None:
        """Reset only the environments where done=True."""
        n_done = int(done_mask.sum().item())
        if n_done == 0:
            return
        self.states[done_mask] = 0
        self.states[done_mask, 4] = self._sample_arrivals(n_done)

    def step(self, actions: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (next_states (N,6), costs (N,), done (N,) bool)."""
        q1, n1, q2, n2, lam, k = self.states.unbind(dim=1)
        q1_, n1_, q2_, n2_, k_ = q1.clone(), n1.clone(), q2.clone(), \
                                 n2.clone(), k.clone()

        costs = torch.full((self.n_envs,), C_ROUTE,
                           dtype=torch.float32, device=self.device)
        costs[actions == A_LOOP] += C_LOOP

        for a, q, n, q_, n_ in ((A_S1, q1, n1, q1_, n1_),
                                (A_S2, q2, n2, q2_, n2_)):
            sel      = (actions == a)
            no_evict = sel & ((q == lam) | (q == EMPTY))
            add      = no_evict & ((n + 1) < CAP)
            flush    = no_evict & ((n + 1) == CAP)
            evict    = sel & (q != lam) & (q != EMPTY)

            q_[add]   = lam[add];  n_[add] = n[add] + 1
            q_[flush] = EMPTY;     n_[flush] = 0;  k_[flush] = k[flush] + 1
            q_[evict] = lam[evict]; n_[evict] = 1
            costs[evict] += C_LIN * n[evict].float()

        self.states = torch.stack([q1_, n1_, q2_, n2_,
                                   self._sample_arrivals(), k_], dim=1)
        done = (k_ >= N_FLUSHES)
        return self.states.clone(), costs, done

    def _sample_arrivals(self, n: Optional[int] = None) -> torch.Tensor:
        size = n if n is not None else self.n_envs
        return torch.multinomial(self.probs.unsqueeze(0).expand(size, -1),
                                 num_samples=1, generator=self.gen
                                 ).squeeze(1) + 1     # 0,1,2 -> 1,2,3


# ══════════════════════════════════════════════════════════════════════════════
# TABULAR Q-LEARNING AGENT
# ══════════════════════════════════════════════════════════════════════════════

class QLearningAgent:
    """
    Tabular Q-Learning agent (cost minimisation, gamma = 1 by default).
    Q-table lives on the device as a (|S|, 3) float tensor.
    """

    def __init__(self,
                 states: list,
                 index: dict,
                 alpha: float = 0.1,
                 gamma: float = 1.0,
                 eps_start: float = 1.0,
                 eps_decay: float = 0.995,
                 eps_min: float = 0.01,
                 use_epsilon: bool = True,
                 device: Optional[torch.device] = None,
                 seed: Optional[int] = None,
                 tables: Optional[dict] = None,
                 q_init: float = 0.0,
                 alpha_mode: str = "constant",
                 alpha_omega: float = 0.7,
                 alpha_min: float = 1e-3,
                 aggregate_k: bool = False):
        """
        alpha_mode : 'constant'  -> alpha fixed
                     'count'     -> alpha(s,a) = max(alpha_min, alpha / N(s,a)^omega)
                                    per state-action visit count (Robbins–Monro
                                    per pair; omega in (0.5, 1]).  A *global*
                                    per-episode decay would starve rarely
                                    visited states, which are the majority here.
        aggregate_k : learn Q over the storage configuration only
                      (q1,n1,q2,n2,lam), ignoring the flush counter k.  Only
                      sensible with gamma < 1 (k is then irrelevant except in
                      the last few layers).  ~100x more samples per (s,a).
                      Goal transitions are treated as continuing (infinite-
                      horizon discounted problem); the resulting stationary
                      policy is broadcast over all k for evaluation on the SSP.
        """

        self.states      = states
        self.index       = index
        self.n_states    = len(states)
        self.n_actions   = len(ACTIONS)
        self.alpha       = alpha
        self.gamma       = gamma
        self.eps         = eps_start
        self.eps_decay   = eps_decay
        self.eps_min     = eps_min
        self.use_epsilon = use_epsilon
        self.q_init      = q_init
        self.alpha_mode  = alpha_mode
        self.alpha_omega = alpha_omega
        self.alpha_min   = alpha_min
        self.aggregate_k = aggregate_k
        self.device      = device or get_device()
        self.rng         = np.random.default_rng(seed)
        self.gen         = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)

        # Precomputed tables (feasibility mask + code->index lookup)
        tables = tables if tables is not None else build_tables(states, index)
        self.FEAS     = torch.tensor(tables["FEAS"], device=self.device)      # (|S|,3) bool
        self.code2idx = torch.tensor(tables["code2idx"], device=self.device)  # (N_CODES,)
        self._goal    = torch.tensor(tables["K"] >= N_FLUSHES, device=self.device)

        # Q-table; infeasible actions are never read thanks to masking.
        # q_init = 0 is OPTIMISTIC for cost minimisation (untried actions look
        # free) -> greedy evaluation drifts into never-corrected loops.
        # A pessimistic q_init (e.g. C_LOOP/(1-gamma), the cost of looping
        # forever) makes greedy avoid untried actions.
        self.Q = torch.full((self.n_states, self.n_actions), float(q_init),
                            dtype=torch.float32, device=self.device)
        self.Q[self._goal] = 0.0
        # Row of the k=0 twin of every state (identity if not aggregating)
        Kt = torch.tensor(tables["K"], device=self.device)
        st = torch.tensor(np.array(states, dtype=np.int64), device=self.device)
        self._k0 = self.code2idx[encode(st[:,0], st[:,1], st[:,2], st[:,3], st[:,4],
                                        torch.zeros_like(st[:,5]))]
        self.N = torch.zeros((self.n_states, self.n_actions),
                             dtype=torch.float32, device=self.device)  # visit counts

        self.episode_costs:   List[float] = []
        self.episode_lengths: List[int]   = []
        self.epsilon_history: List[float] = []

    # ── Indexing ───────────────────────────────────────────────────────────────

    def state_to_idx(self, state_batch: torch.Tensor) -> torch.Tensor:
        """(N,6) long tensor -> (N,) Q-table row indices, all on device."""
        s = state_batch
        k = torch.zeros_like(s[:, 5]) if self.aggregate_k else s[:, 5]
        codes = encode(s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], k)
        return self.code2idx[codes]

    # ── Masked Q helpers ───────────────────────────────────────────────────────

    def _masked_Q(self, idx: torch.Tensor) -> torch.Tensor:
        """Q rows with infeasible actions set to +inf (cost minimisation)."""
        return self.Q[idx].masked_fill(~self.FEAS[idx], float('inf'))

    # ── Action selection ───────────────────────────────────────────────────────

    def select_actions_batch(self, idx_batch: torch.Tensor) -> torch.Tensor:
        """Vectorised epsilon-greedy over feasible actions. Returns (N,)."""
        greedy = self._masked_Q(idx_batch).argmin(dim=1)
        if not (self.use_epsilon and self.eps > 0):
            return greedy
        N       = idx_batch.shape[0]
        fmask   = self.FEAS[idx_batch]
        # Uniform random feasible action: random scores, infeasible -> -inf
        scores  = torch.rand((N, self.n_actions), device=self.device,
                             generator=self.gen).masked_fill(~fmask, -1.0)
        random_a = scores.argmax(dim=1)
        explore  = torch.rand(N, device=self.device, generator=self.gen) < self.eps
        return torch.where(explore, random_a, greedy)

    def select_action(self, state: Tuple, env: WarehouseEnv) -> int:
        """Single-state action selection (sequential mode)."""
        feasible = env.feasible_actions(state)
        if self.use_epsilon and self.rng.random() < self.eps:
            return int(self.rng.choice(feasible))
        idx    = self.index[state]
        q_vals = [(self.Q[idx, a].item(), a) for a in feasible]
        return min(q_vals, key=lambda x: x[0])[1]

    # ── Q update ───────────────────────────────────────────────────────────────

    def update_batch(self, idx, actions, costs, next_idx, done):
        """
        Q(s,a) <- Q(s,a) + alpha [ c + gamma * min_{a' feasible} Q(s',a') - Q(s,a) ]
        with V(goal) = 0.
        """
        with torch.no_grad():
            current = self.Q[idx, actions]
            next_v  = self._masked_Q(next_idx).min(dim=1).values
            if not self.aggregate_k:          # SSP: V(goal) = 0
                next_v = torch.where(done, torch.zeros_like(next_v), next_v)
            target  = costs + self.gamma * next_v
            if self.alpha_mode == "count":
                self.N.index_put_((idx, actions), torch.ones_like(costs), accumulate=True)
                alpha = (self.alpha / self.N[idx, actions] ** self.alpha_omega
                         ).clamp_(min=self.alpha_min)
            else:
                alpha = self.alpha
            self.Q[idx, actions] = current + alpha * (target - current)

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    # ── Parallel training ──────────────────────────────────────────────────────

    def train_parallel(self,
                       n_envs: int = 512,
                       n_episodes: int = 5_000,
                       verbose: bool = True,
                       seed: int = 42) -> Dict:
        """
        Parallel training with n_envs simultaneous episodes.  One 'episode'
        = one env reaching k = N_FLUSHES; epsilon decays per completed episode.
        """
        print(f"🚀 Parallel training: {n_envs} envs on {self.device}")
        vec_env  = VectorisedWarehouseEnv(n_envs, device=self.device, seed=seed)
        states   = vec_env.reset()
        idx      = self.state_to_idx(states)
        ep_costs = torch.zeros(n_envs, device=self.device)
        ep_steps = torch.zeros(n_envs, dtype=torch.long, device=self.device)

        completed = 0
        t0   = time.time()
        pbar = tqdm(total=n_episodes, desc="Parallel Training",
                    unit="ep") if verbose else None

        while completed < n_episodes:
            actions = self.select_actions_batch(idx)
            next_states, step_costs, done = vec_env.step(actions)
            next_idx = self.state_to_idx(next_states)
            self.update_batch(idx, actions, step_costs, next_idx, done)

            ep_costs += step_costs
            ep_steps += 1

            if bool(done.any()):
                for c, s in zip(ep_costs[done].tolist(), ep_steps[done].tolist()):
                    self.episode_costs.append(float(c))
                    self.episode_lengths.append(int(s))
                    self.epsilon_history.append(self.eps)
                    self.decay_epsilon()
                    completed += 1
                if pbar:
                    pbar.update(int(done.sum().item()))
                    pbar.set_postfix(cost=f"{ep_costs[done].mean().item():.1f}",
                                     eps=f"{self.eps:.3f}")
                ep_costs[done] = 0
                ep_steps[done] = 0
                vec_env.reset_done(done)
                next_states = vec_env.states
                next_idx    = self.state_to_idx(next_states)

            states, idx = next_states, next_idx

        if pbar:
            pbar.close()
        runtime = time.time() - t0
        if verbose:
            print(f"✅ Parallel training complete — {completed} episodes "
                  f"in {runtime:.1f}s ({completed/runtime:.1f} ep/s)")
        return self._stats(runtime)

    # ── Sequential training (any backend, incl. Plant Simulation) ──────────────

    def train(self, env: WarehouseEnv, n_episodes: int = 5_000,
              max_steps: int = 50_000, verbose: bool = True) -> Dict:
        iterator = tqdm(range(n_episodes), desc="Training",
                        unit="ep") if verbose else range(n_episodes)
        t0 = time.time()
        for _ in iterator:
            state, total_cost, steps = env.reset(), 0.0, 0
            for _ in range(max_steps):
                action = self.select_action(state, env)
                next_state, stage_cost, done = env.step(action)
                self.update_batch(
                    torch.tensor([self.index[state]],      device=self.device),
                    torch.tensor([action],                 device=self.device),
                    torch.tensor([stage_cost], dtype=torch.float32,
                                 device=self.device),
                    torch.tensor([self.index[next_state]], device=self.device),
                    torch.tensor([done],                   device=self.device),
                )
                total_cost += stage_cost
                steps      += 1
                state       = next_state
                if done:
                    break
            self.episode_costs.append(total_cost)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.eps)
            self.decay_epsilon()
            if verbose:
                iterator.set_postfix(cost=f"{total_cost:.1f}",
                                     eps=f"{self.eps:.3f}", steps=steps)
        return self._stats(time.time() - t0)

    def _stats(self, runtime: float) -> Dict:
        return {"episode_costs"  : self.episode_costs,
                "episode_lengths": self.episode_lengths,
                "epsilon_history": self.epsilon_history,
                "runtime"        : runtime}

    # ── Policy & value extraction ──────────────────────────────────────────────

    def get_policy(self) -> np.ndarray:
        """Greedy policy over FEASIBLE actions; -1 on goal states."""
        all_idx = self._k0 if self.aggregate_k else torch.arange(self.n_states, device=self.device)
        pol = self._masked_Q(all_idx).argmin(dim=1)   # broadcast over k if aggregating
        pol[self._goal] = -1
        return pol.cpu().numpy().astype(np.int32)

    def get_value_function(self) -> np.ndarray:
        all_idx = torch.arange(self.n_states, device=self.device)
        V = self._masked_Q(all_idx).min(dim=1).values
        V[self._goal] = 0.0
        return V.cpu().numpy()

    def visit_counts(self) -> np.ndarray:
        """Rows whose Q was ever updated (Q != 0 on some feasible action)."""
        return ((self.Q != self.q_init) & self.FEAS).any(dim=1).cpu().numpy()

    def warm_start(self, V_star: np.ndarray, tables: dict):
        """
        Initialise Q with the TRUE action values
            Q*(s,a) = c(s,a) + sum_l p_l V*(f(s,a,l)).
        (Setting Q(s,a)=V*(s) for all a is wrong: it makes 'loop' look as
        good as the optimal action, and greedy evaluation then drifts into
        loops that are never corrected.)
        """
        p  = np.asarray(ARRIVAL_P, dtype=np.float64)
        Qs = tables["C"] + np.einsum('l,sal->sa', p, V_star[tables["NXT"]])
        Qs = np.where(np.isfinite(Qs), Qs, 0.0)
        self.Q[:] = torch.tensor(Qs, dtype=torch.float32, device=self.device)
        print("✅ Q-table warm-started from Q* = c + p·V*(f).")
