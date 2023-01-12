from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from .utils import np1f, np2f, shrink, svd_threshold

@dataclass
class RPCA:
    """Robust PCA analysis"""
    L: np2f
    S: np2f

    @classmethod
    def fit(cls, m: np2f, mu:float|None=None, tol:float|None=None, max_iter=50, iter_print=1) -> "RPCA":  #type:ignore
        """Fits L and S from X = L + S given X. Algorithm from https://arxiv.org/pdf/0912.3599.pdf, Algorithm 1, page 29"""
        if mu is None:
            mu: float = np.prod(m.shape) / (4*np.linalg.norm(m, ord=1))  # type:ignore
        mu_recip = 1/mu

        lamda: float = 1/np.sqrt(np.max(m.shape))

        err = np.inf
        s = np.zeros_like(m)
        y = np.zeros_like(m)
        l = np.zeros_like(m)

        if not tol:
            tol:float = 1e-7 * np.linalg.norm(m, ord='fro')  # type:ignore

        print(f'{mu=}, {lamda=}, {tol=}')

        try:
            for i in tqdm(range(max_iter)):
                l = svd_threshold(m - s+mu_recip * y, mu_recip)
                s = shrink(m - l + (mu_recip * y), mu_recip * lamda)
                y += mu * (m - l - s)
                err = np.linalg.norm(m - l - s, ord='fro')

                if (i % iter_print) == 0 or i == max_iter:
                    tqdm.write(f'{err=}')
                
                if err <= tol:
                    tqdm.write(f'{err=} (under tolerance, stopping)')
                    break
        except KeyboardInterrupt:
            pass

        return cls(l, s)

    def __call__(self, x: np1f|np2f) -> np1f|np2f:
        return self.L @ x
