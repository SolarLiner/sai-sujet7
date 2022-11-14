from argparse import ArgumentParser
from typing import TypeAlias
from PIL import Image
import numpy as np
from scipy import io
from pathlib import Path

from bdrpca.drpca import DRPCA


_M: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float_]]

IMG_WIDTH = 161


def args() -> tuple[Path, str]:
    p = ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('--name', type=str, help='Name within .mat file', default='M1')

    args = p.parse_args()
    return args.input, args.name


def run():
    path, name = args()

    pca = DRPCA()
    mat: _M = io.matlab.loadmat(path)[name]
    Image.fromarray(mat * 255).convert('L').save('input.png')
    T, x = pca(mat)
    Image.fromarray(T * 255).convert('L').save('output_T.png')
    Image.fromarray(x * 255).convert('L').save('output_x.png')
