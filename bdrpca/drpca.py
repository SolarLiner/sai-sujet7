from dataclasses import dataclass
from math import sqrt
from typing import Optional

import numpy as np
from numpy.fft import fft2, ifft2
from numpy.linalg import norm, matrix_rank
from tqdm import tqdm

from .utils import np2f, np2c, shrink, svd_threshold

@dataclass
class DRPCA:
    lam: Optional[float] = None
    rho: float = 1
    mu: float = 1e-6
    eps: float = 2
    mu_max: float = 1e6
    tol: float = 1e-6
    max_iter: int = 50

    def __call__(self, S: np2f, H: Optional[np2c]=None, iter_print=1) -> tuple[np2f, np2f]:
        """Performs deconvolution. Returns (T, x) where T is a tissue mask, and x is the recovered high-resolution blood"""
        m, n = S.shape
        unobserved = np.isnan(S)
        S = np.where(unobserved, 0, S)
        norm_x = np.linalg.norm(S, 'fro')

        if (lam := self.lam) is None:
            lam = 1/sqrt(max(m,n))
        mu = self.mu

        if H is None:
            H = np.ones_like(S, dtype=np.complex64)

        T = np.zeros_like(S)
        x = np.zeros_like(S)
        z = np.zeros_like(S)
        N = np.zeros_like(S)
        W = np.zeros_like(S)
        Hx = np.real(ifft2(H * fft2(x)))

        try:
            for i in tqdm(range(self.max_iter)):
                # ADMM step: update T, x, z
                T = svd_threshold(S - Hx + (1/mu) * N, self.rho/mu)
                z = shrink(x + (1/mu) * W, lam/mu)
                x1 = np.real(ifft2(np.conj(H) * fft2(S-T))) + z + (1/mu) * (np.real(ifft2(np.conj(H) * fft2(N))) - W)
                h1 = 1/(np.abs(H) ** 2 + 1)
                x = np.real(ifft2(h1 * fft2(x1)))
                Hx = np.real(ifft2(H * fft2(x)))

                # Augmented lagradian multiplier
                Z1 = S - T - Hx
                Z1[unobserved] = 0
                N += mu * Z1
                Z2 = x - z
                Z2[unobserved] = 0
                W += mu * Z2
                mu = min(mu * self.eps, self.mu_max)

                err1 = norm(Z1, 'fro') / norm_x
                err2 = norm(Z2, 'fro') / norm_x

                if (i % iter_print) == 0 or i == self.max_iter:
                    rankT = matrix_rank(T)
                    cardS = len(np.nonzero(x[~unobserved]))
                    tqdm.write(f"{i=}\t{err1=}\t{err2=}\t{rankT=}\t{cardS=}")
                if err1 < self.tol and err2 < self.tol:
                    break
        except KeyboardInterrupt:
            pass
        return T, x
