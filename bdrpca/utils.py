import numpy as np

np1f = np.ndarray[tuple[int], np.dtype[np.float_]]
np2f = np.ndarray[tuple[int,int], np.dtype[np.float_]]
np2c = np.ndarray[tuple[int,int], np.dtype[np.complex_]]

def shrink(arr: np2f, tau: float) -> np2f:
    return np.sign(arr) * np.maximum((np.abs(arr) - tau), 0)  #type:ignore

def svd_threshold(arr: np2f, tau: float) -> np2f:
    u, s, v = np.linalg.svd(arr, full_matrices=False)
    return u @ np.diag(shrink(s, tau)) @ v
