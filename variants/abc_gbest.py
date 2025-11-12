# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Callable, Any, Dict
from dataclasses import dataclass
from abc_base import ArtificialBeeColony, ABCResult  

class GbestABC(ArtificialBeeColony):
    def __init__(self, *args, beta: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = float(beta)

    def _neighbor_one_dim(self, i: int) -> np.ndarray:
        # chọn k != i
        k = i
        while k == i:
            k = int(self.rng.integers(0, self.sn))
        j = int(self.rng.integers(0, self.d))
        phi = self.rng.uniform(-1.0, 1.0)
        v = self.X[i].copy()

        # nếu best_x chưa sẵn (đầu run đã có, nhưng vẫn phòng hờ)
        best = self.best_x if self.best_x is not None else self.X[i]

        pull = self.beta * self.rng.random() * (best[j] - self.X[i, j])
        v[j] = self.X[i, j] + phi * (self.X[i, j] - self.X[k, j]) + pull

        return self._boundary_clip(v)
