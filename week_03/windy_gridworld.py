from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable, List
import numpy as np

from gridworld import ACTIONS, Action

@dataclass(frozen=True)
class WindSpec:
    H: int
    W: int
    start: Tuple[int,int]
    goal: Tuple[int,int]
    wind_by_col: Tuple[int, ...]      
    stochastic: bool = True
    p_up1: float = 0.1                
    p_base: float = 0.8               
    p_zero: float = 0.1               
    walls: Tuple[Tuple[int,int], ...] = tuple()
    reward_step: float = -1.0
    reward_goal: float = 0.0
    gamma: float = 1.0

class WindyGridworld:
    """Stochastic Windy Gridworld with 4 actions."""
    def __init__(self, spec: WindSpec):
        self.spec = spec
        self.H, self.W = spec.H, spec.W
        self.start = spec.start
        self.goal = spec.goal
        self.walls = set(spec.walls)

    def in_bounds(self, r:int, c:int) -> bool:
        return 0 <= r < self.H and 0 <= c < self.W and (r,c) not in self.walls

    def reset(self): return self.start

    def _apply(self, r:int, c:int, dr:int, dc:int):
        rr, cc = r + dr, c + dc
        base = self.spec.wind_by_col[c]
        if self.spec.stochastic:
            rnd = np.random.rand()
            if rnd < self.spec.p_up1:      w = base + 1
            elif rnd < self.spec.p_up1 + self.spec.p_base: w = base
            else:                           w = 0
        else:
            w = base
        rr -= w
        rr = max(0, min(self.H-1, rr))
        cc = max(0, min(self.W-1, cc))
        if (rr,cc) in self.walls:
            rr,cc = r,c
        return rr,cc

    def step(self, s, a:Action):
        if s == self.goal: return s, 0.0, True
        dr, dc = ((-1,0),(1,0),(0,-1),(0,1))[a]
        r,c = s
        nr,nc = self._apply(r,c,dr,dc)
        ns = (nr,nc)
        if ns == self.goal: return ns, self.spec.reward_goal, True
        return ns, self.spec.reward_step, False

    def actions(self, s) -> Iterable[Action]:
        if s == self.goal: return []
        return range(4)

    def all_states(self) -> List[Tuple[int,int]]:
        return [(r,c) for r in range(self.H) for c in range(self.W) if (r,c) not in self.walls]
