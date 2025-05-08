import numpy as np
from scipy.fftpack import dct, idct

def dct2(matrix):
    """2D Discrete Cosine Transform"""
    return dct(dct(matrix.T, norm='ortho').T, norm='ortho')

def idct2(matrix):
    """2D Inverse Discrete Cosine Transform"""
    return idct(idct(matrix.T, norm='ortho').T, norm='ortho')