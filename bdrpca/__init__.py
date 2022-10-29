from argparse import ArgumentParser
from typing import TypeAlias
from PIL import Image
import numpy as np
from scipy import io
from pathlib import Path

from bdrpca.drpca import DRPCA


_M: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float_]]

IMG_WIDTH = 100


def args() -> tuple[Path]:
    p = ArgumentParser()
    p.add_argument('input', type=Path)

    args = p.parse_args()
    return args.input,


def run():
    path, = args()

    pca = DRPCA()
    mat: _M = io.matlab.loadmat(path)["phantom_amplitudes"].reshape((IMG_WIDTH, -1))
    Image.fromarray(mat * 255).convert('L').save('input.png')
    T, x = pca(mat)
    Image.fromarray(T * 255).convert('L').save('output_T.png')
    Image.fromarray(x * 255).convert('L').save('output_x.png')