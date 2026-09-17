from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class FuzzyARTConfig:
    vigilance: float = 0.85
    beta: float = 0.5
    alpha: float = 1e-3
    max_categories: int | None = None


class FuzzyART:
    """Fuzzy ART implemented from the equations reported in the manuscript.

    Input must be scaled to [0,1]. Complement coding is intentionally NOT used
    because it is not part of the manuscript specification.
    """
    def __init__(self, config: FuzzyARTConfig | None = None):
        self.cfg = config or FuzzyARTConfig()
        self.weights: list[np.ndarray] = []
        self.category_labels: dict[int, int] = {}

    @staticmethod
    def _fuzzy_and(x, w):
        return np.minimum(x, w)

    def _choice(self, x, w):
        return self._fuzzy_and(x, w).sum() / (self.cfg.alpha + w.sum())

    def _match(self, x, w):
        return self._fuzzy_and(x, w).sum() / (x.sum() + 1e-8)

    def _update(self, x, w):
        return self.cfg.beta * self._fuzzy_and(x, w) + (1.0 - self.cfg.beta) * w

    def _select_category(self, x: np.ndarray):
        if not self.weights:
            return None
        choices = np.asarray([self._choice(x, w) for w in self.weights])
        order = np.argsort(-choices)
        for j in order:
            if self._match(x, self.weights[j]) >= self.cfg.vigilance:
                return int(j)
        return None

    def fit_categories(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if np.any(X < -1e-8) or np.any(X > 1 + 1e-8):
            raise ValueError("Fuzzy ART inputs must be scaled to [0,1]")
        assignments = []
        for x in X:
            j = self._select_category(x)
            if j is None:
                if self.cfg.max_categories is None or len(self.weights) < self.cfg.max_categories:
                    self.weights.append(x.copy())
                    j = len(self.weights) - 1
                else:
                    # manuscript's capped strategy: force assignment by best choice
                    j = int(np.argmax([self._choice(x, w) for w in self.weights]))
            self.weights[j] = self._update(x, self.weights[j])
            assignments.append(j)
        return np.asarray(assignments, dtype=int)

    def fit(self, X: np.ndarray, y: np.ndarray):
        assignments = self.fit_categories(X)
        votes: dict[int, list[int]] = defaultdict(list)
        for j, label in zip(assignments, np.asarray(y, dtype=int)):
            votes[int(j)].append(int(label))
        self.category_labels = {
            j: int(np.mean(labels) >= 0.5) for j, labels in votes.items()
        }
        return self

    def predict_category(self, X: np.ndarray) -> np.ndarray:
        out = []
        for x in np.asarray(X, dtype=np.float64):
            j = self._select_category(x)
            if j is None:
                j = int(np.argmax([self._choice(x, w) for w in self.weights])) if self.weights else -1
            out.append(j)
        return np.asarray(out, dtype=int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        cats = self.predict_category(X)
        return np.asarray([self.category_labels.get(int(j), 0) for j in cats], dtype=int)

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        """Continuous ART score for AUC: choice-weighted malignant-category evidence."""
        scores = []
        for x in np.asarray(X, dtype=np.float64):
            if not self.weights:
                scores.append(0.5); continue
            choices = np.asarray([max(self._choice(x, w), 0.0) for w in self.weights])
            if choices.sum() <= 0:
                scores.append(0.5); continue
            labels = np.asarray([self.category_labels.get(j, 0) for j in range(len(self.weights))], dtype=float)
            scores.append(float(np.dot(choices, labels) / choices.sum()))
        return np.asarray(scores, dtype=float)
