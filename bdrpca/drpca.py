from dataclasses import dataclass
from math import sqrt
from typing import Optional, TypeAlias


import numpy as np
from numpy import ndarray

_M: TypeAlias = ndarray[tuple[int, int], np.dtype[np.float_]]
_V: TypeAlias = ndarray[int, np.dtype[np.float_]]

@dataclass
class DRPCA:
    lam: Optional[float] = None
    rho: float = 1
    mu: float = 1e-6
    eps: float = 2
    mu_max: float = 1e6
    tol: float = 1e-6
    max_iter: int = 50

    def __call__(self, S: _M, H: Optional[_M]=None) -> tuple[_M, _M]:
        """Performs deconvolution. Returns (T, x) where T is a tissue mask, and x is the recovered high-resolution blood"""
        m, n = S.shape
        unobserved = np.isnan(S)
        S = np.where(unobserved, 0, S)
        norm_x = np.linalg.norm(S, 'fro')

        lam = self.lam or (1/sqrt(max(m,n)))
        mu = self.mu

        if H is None:
            H = np.ones_like(S)

        T = np.zeros_like(S)
        x = np.zeros_like(S)
        z = np.zeros_like(S)
        N = np.zeros_like(S)
        W = np.zeros_like(S)
        Hx = np.real(np.fft.ifft2(H * np.fft.fft2(x)))

        for i in range(self.max_iter):
            # ADMM step: update T, x, z
            T = so(S - Hx + (1/mu) * N, self.rho/mu)
            z = so(S-Hx + (1/mu) * N, lam/mu)
            x1 = np.real(np.fft.ifft2(np.conj(H) * np.fft.fft2(S-T))) + z + (1/mu) * (np.real(np.fft.ifft2(np.conj(H) * np.fft.fft2(N))) - W)
            h1 = 1/(np.abs(H) ** 2 + np.ones_like(S))
            x = np.real(np.fft.ifft2(h1 * np.fft.fft2(x1)))
            Hx = np.real(np.fft.ifft2(H * np.fft.fft2(x)))

            # Augmented lagradian multiplier
            Z1 = S - T - Hx
            Z1[unobserved] = 0
            N += mu * Z1
            Z2 = x - z
            Z2[unobserved] = 0
            W += mu * Z2
            mu = min(mu * self.eps, self.mu_max)

            err1 = np.linalg.norm(Z1, 'fro') / norm_x
            err2 = np.linalg.norm(Z2, 'fro') / norm_x

            if i == 0 or (err1 > self.tol) or err2 > self.tol:
                rankT = np.linalg.matrix_rank(T)
                cardS = len(np.nonzero(x[~unobserved]))
                print(f"{i=}\t{err1=}\t{err2=}\t{rankT=}\t{cardS=}")
            if err1 < self.tol and err2 < self.tol:
                break
        return T, x
            



def so(S: _M, tau: float) -> _M:
    return np.sign(S) * np.maximum(np.abs(S) - tau, 0)

def do(S: _M, tau: float) -> _M:
    U, D, V = np.linalg.svd(S)
    return U @ so(D, tau) @ V