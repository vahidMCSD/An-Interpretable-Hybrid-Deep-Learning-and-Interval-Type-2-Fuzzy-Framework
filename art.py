import numpy as np

class FuzzyART:
    """
    Lightweight fuzzy ART implementation for complement-coded inputs.
    rho=0.85 and beta=0.5 follow the manuscript.

    The manuscript reports two mapping strategies; this class supports
    unrestricted categories followed by supervised majority-vote mapping.
    """
    def __init__(self, rho=0.85, beta=0.5, alpha=1e-3):
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.W = []
        self.cat_label = {}

    @staticmethod
    def _complement_code(x):
        x = np.clip(x, 0, 1)
        return np.concatenate([x, 1-x], axis=-1)

    @staticmethod
    def _fuzzy_and(a,b):
        return np.minimum(a,b)

    def fit(self, X, y):
        Xc = self._complement_code(X)
        cat_samples = {}

        for i, x in enumerate(Xc):
            if not self.W:
                self.W.append(x.copy())
                cat_samples[0] = [int(y[i])]
                continue

            choices = []
            for j,w in enumerate(self.W):
                num = np.sum(self._fuzzy_and(x,w))
                den = self.alpha + np.sum(w)
                choices.append((num/den, j))
            choices.sort(reverse=True)

            assigned = None
            for _, j in choices:
                w = self.W[j]
                match = np.sum(self._fuzzy_and(x,w)) / (np.sum(x) + 1e-12)
                if match >= self.rho:
                    self.W[j] = self.beta*self._fuzzy_and(x,w) + (1-self.beta)*w
                    assigned = j
                    break

            if assigned is None:
                assigned = len(self.W)
                self.W.append(x.copy())

            cat_samples.setdefault(assigned, []).append(int(y[i]))

        self.cat_label = {
            j: int(np.mean(labels) >= 0.5)
            for j, labels in cat_samples.items()
        }
        return self

    def predict(self, X):
        Xc = self._complement_code(X)
        preds = []
        for x in Xc:
            best = None
            for j,w in enumerate(self.W):
                choice = np.sum(self._fuzzy_and(x,w)) / (self.alpha + np.sum(w))
                if best is None or choice > best[0]:
                    best = (choice,j)
            preds.append(self.cat_label.get(best[1], 0))
        return np.asarray(preds, dtype=np.int64)
