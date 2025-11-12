from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict, Any
import numpy as np

Array = np.ndarray


@dataclass
class ABCResult:
    """Result container for the ABC run."""
    x_best: Array
    f_best: float
    history: Dict[str, Any] 


class ArtificialBeeColony:
    def __init__(
        self,
        obj_func: Callable[[Array], float],
        bounds: Tuple[Array, Array],
        sn: int,
        limit: int,
        max_cycles: int,
        seed: Optional[int] = None,
    ):
        self.f = obj_func
        self.lower = np.array(bounds[0], dtype=float)
        self.upper = np.array(bounds[1], dtype=float)
        assert self.lower.shape == self.upper.shape, "lower/upper must have same shape"
        self.d = int(self.lower.size)
        self.sn = int(sn)
        self.limit = int(limit)
        self.max_cycles = int(max_cycles)
        self.rng = np.random.default_rng(seed)

        self.X = np.empty((self.sn, self.d), dtype=float)
        self.fvals = np.empty(self.sn, dtype=float)
        self.fits = np.empty(self.sn, dtype=float)
        self.trials = np.zeros(self.sn, dtype=int)
        self.best_x: Optional[Array] = None
        self.best_f: float = float("inf")

    @staticmethod
    def _fitness_from_objective(fx: float) -> float:
        return (1.0 / (1.0 + fx)) if fx >= 0.0 else (1.0 + abs(fx))

    def _boundary_clip(self, v: Array) -> Array:
        return np.minimum(np.maximum(v, self.lower), self.upper)

    def _update_best_from_pop(self):
        i = int(np.argmin(self.fvals))
        if self.fvals[i] < self.best_f:
            self.best_f = float(self.fvals[i])
            self.best_x = self.X[i].copy()

    def _evaluate_all(self):
        for i in range(self.sn):
            fx = float(self.f(self.X[i]))
            self.fvals[i] = fx
            self.fits[i] = self._fitness_from_objective(fx)

    def _roulette_select(self) -> int:
        s = float(self.fits.sum())
        if not np.isfinite(s) or s <= 1e-15:
            return int(self.rng.integers(0, self.sn))
        probs = self.fits / s
        return int(self.rng.choice(self.sn, p=probs))

    def _neighbor_one_dim(self, i: int) -> Array:
        k = i
        while k == i:
            k = int(self.rng.integers(0, self.sn))
        # choose random dimension
        j = int(self.rng.integers(0, self.d))
        phi = self.rng.uniform(-1.0, 1.0)

        v = self.X[i].copy()
        v[j] = self.X[i, j] + phi * (self.X[i, j] - self.X[k, j])
        return self._boundary_clip(v)

    def _employed_phase(self):
        for i in range(self.sn):
            v = self._neighbor_one_dim(i)
            fv = float(self.f(v))
            if fv <= self.fvals[i]: 
                self.X[i] = v
                self.fvals[i] = fv
                self.fits[i] = self._fitness_from_objective(fv)
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def _onlooker_phase(self):
        self.fits = np.array([self._fitness_from_objective(fx) for fx in self.fvals], dtype=float)
        count = 0
        while count < self.sn:
            i = self._roulette_select()
            v = self._neighbor_one_dim(i)
            fv = float(self.f(v))
            if fv <= self.fvals[i]:
                self.X[i] = v
                self.fvals[i] = fv
                self.fits[i] = self._fitness_from_objective(fv)
                self.trials[i] = 0
            else:
                self.trials[i] += 1
            count += 1

    def _scout_phase(self):
        for i in range(self.sn):
            if self.trials[i] >= self.limit:
                self.X[i] = self.rng.uniform(self.lower, self.upper, size=self.d)
                fx = float(self.f(self.X[i]))
                self.fvals[i] = fx
                self.fits[i] = self._fitness_from_objective(fx)
                self.trials[i] = 0

    def run(self) -> ABCResult:
        self.X = self.rng.uniform(self.lower, self.upper, size=(self.sn, self.d))
        self._evaluate_all()
        self._update_best_from_pop()

        history_best = np.empty(self.max_cycles, dtype=float)
        for t in range(self.max_cycles):
            self._employed_phase()
            self._onlooker_phase()
            self._scout_phase()
            self._update_best_from_pop()
            history_best[t] = self.best_f

        return ABCResult(
            x_best=self.best_x.copy(),
            f_best=float(self.best_f),
            history={"f_best_per_cycle": history_best},
        )
