from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Iterable
import numpy as np

Action = int  # 0: up, 1: down, 2: left, 3: right
ACTIONS: Tuple[Tuple[int,int], ...] = ((-1,0),(1,0),(0,-1),(0,1))

@dataclass(frozen=True)
class GridSpec:
    H: int
    W: int
    terminal: Tuple[Tuple[int,int], ...]
    walls: Tuple[Tuple[int,int], ...] = tuple()
    reward_step: float = -1.0
    reward_terminal: float = 0.0
    gamma: float = 1.0

class Gridworld:
    """
    Deterministic 4-direction gridworld.
    Rewards: -1 per step; 0 when entering terminal (Example 4.1).
    """
    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.H, self.W = spec.H, spec.W
        self.terminal = set(spec.terminal)
        self.walls = set(spec.walls)
        self.gamma = spec.gamma
        self.states: List[Tuple[int,int]] = [
            (r,c) for r in range(self.H) for c in range(self.W) if (r,c) not in self.walls
        ]
        self.start = next(s for s in self.states if s not in self.terminal)

    def in_bounds(self, r:int, c:int) -> bool:
        return 0 <= r < self.H and 0 <= c < self.W

    def step(self, s:Tuple[int,int], a:Action):
        if s in self.terminal:
            return s, 0.0, True
        dr, dc = ACTIONS[a]
        r, c = s
        nr, nc = r + dr, c + dc
        if not self.in_bounds(nr,nc) or (nr,nc) in self.walls:
            nr, nc = r, c
        ns = (nr, nc)
        if ns in self.terminal:
            return ns, self.spec.reward_terminal, True
        return ns, self.spec.reward_step, False

    def transitions(self, s:Tuple[int,int], a:Action) -> Dict[Tuple[int,int], float]:
        ns, _, _ = self.step(s,a)
        return {ns: 1.0}

    def all_states(self) -> List[Tuple[int,int]]:
        return [s for s in self.states]

    def nonterminal_states(self) -> List[Tuple[int,int]]:
        return [s for s in self.states if s not in self.terminal]

    def actions(self, s:Tuple[int,int]) -> Iterable[Action]:
        if s in self.terminal: return []
        return range(4)

    def index_map(self) -> Dict[Tuple[int,int], int]:
        return {s:i for i,s in enumerate(self.states)}

    def to_value_grid(self, v:Dict[Tuple[int,int], float]) -> np.ndarray:
        arr = np.zeros((self.H,self.W), dtype=float)
        for r in range(self.H):
            for c in range(self.W):
                if (r,c) in self.walls:
                    arr[r,c] = np.nan
                else:
                    arr[r,c] = v.get((r,c), 0.0)
        return arr

    def greedy_policy_from_v(self, v:Dict[Tuple[int,int],float]):
        pi = {}
        for s in self.nonterminal_states():
            best_a, best_q = None, -1e18
            for a in self.actions(s):
                q = 0.0
                for ns, p in self.transitions(s,a).items():
                    r = self.spec.reward_terminal if ns in self.terminal else self.spec.reward_step
                    q += p * (r + self.gamma * v.get(ns, 0.0))
                if q > best_q + 1e-12:
                    best_q, best_a = q, a
            pi[s] = best_a
        return pi

def example41_env(N:int=4) -> Gridworld:
    # Two terminal corners (both “gray”): (0,0) and (N-1,N-1)
    term = ((0,0),(N-1,N-1))
    spec = GridSpec(H=N, W=N, terminal=term, reward_step=-1.0, reward_terminal=0.0, gamma=1.0)
    return Gridworld(spec)
