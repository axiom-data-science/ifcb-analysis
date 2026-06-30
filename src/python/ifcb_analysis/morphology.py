import numpy as np
from scipy.ndimage import correlate, measurements
from skimage.morphology import (binary_dilation, binary_erosion, disk,
                                reconstruction)

EIGHT = np.ones((3,3)).astype(bool)
FOUR = disk(1).astype(bool)

SE2 = disk(2).astype(bool)
SE3 = np.ones((5,5)).astype(bool)

def find_perimeter(B):
    B = np.array(B).astype(bool) * 1
    """find boundaries via erosion and logical and,
    using four-connectivity"""
    #return B & np.invert(binary_erosion(B,FOUR))
    S = np.array([[ 0,-1, 0],
                  [-1, 4,-1],
                  [ 0,-1, 0]])
    return correlate(B,S,mode='constant') > 0

def benkrid_perimeter(border_image):
    """Perimeter estimation based on K. Benkrid, D. Crookes.
    Design and FPGA Implementation of a Perimeter Estimator.
    The Queen's University of Belfast."""
    weights = np.zeros(50)
    weights[[5, 7, 15, 17, 25, 27]] = 1
    weights[[21, 33]] = np.sqrt(2)
    weights[[13, 23]] = (1 + np.sqrt(2)) / 2

    kernel = np.array([[10, 2, 10],
                       [ 2, 1,  2],
                       [10, 2, 10]])
    coded_image = correlate(border_image.astype(np.float32), kernel, mode='constant')
    weighted_coded_image = weights[coded_image.astype(int)]
    return np.sum(weighted_coded_image)

def hysthresh(img,T1,T2):
    """hysteresis thresholding"""
    """All pixels with values above T1 are marked as edges.
    All pixels adjacent to points that have been marked as edges
    and with values above T2 are also marked as edges. Eight-
    connectivity is used.
    Adapted from Peter Kovesi"""
    T2,T1 = sorted([T1,T2])
    edges = img > T1
    sum = 1
    while sum > 0:

        bd = (binary_dilation(edges,EIGHT) & (img > T2)) - edges
        edges = np.logical_or(bd, edges)
        sum = np.sum(bd)
    return edges

"""bwmorph_thin"""

"""
# here's how to make the LUTs

def nabe(n):
    return np.array([n>>i&1 for i in range(0,9)]).astype(bool)

def hood(n):
    return np.take(nabe(n), np.array([[3, 2, 1],
                                      [4, 8, 0],
                                      [5, 6, 7]]))
def G1(n):
    s = 0
    bits = nabe(n)
    for i in (0,2,4,6):
        if not(bits[i]) and (bits[i+1] or bits[(i+2) % 8]):
            s += 1
    return s==1

g1_lut = np.array([G1(n) for n in range(256)])

def G2(n):
    n1, n2 = 0, 0
    bits = nabe(n)
    for k in (1,3,5,7):
        if bits[k] or bits[k-1]:
            n1 += 1
        if bits[k] or bits[(k+1) % 8]:
            n2 += 1
    return min(n1,n2) in [2,3]

g2_lut = np.array([G2(n) for n in range(256)])

g12_lut = g1_lut & g2_lut

def G3(n):
    bits = nabe(n)
    return not((bits[1] or bits[2] or not(bits[7])) and bits[0])

def G3p(n):
    bits = nabe(n)
    return not((bits[5] or bits[6] or not(bits[3])) and bits[4])

g3_lut = np.array([G3(n) for n in range(256)])
g3p_lut = np.array([G3p(n) for n in range(256)])

g123_lut  = g12_lut & g3_lut
g123p_lut = g12_lut & g3p_lut
"""

G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 0, 0], dtype=bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], dtype=bool)

def bwmorph_thin(B, n_iter=1):
    mask = np.array([[ 8,  4,  2],
                     [16,  0,  1],
                     [32, 64,128]],dtype=np.uint8)
    skel = np.array(B).astype(bool).astype(np.uint8)
    for n in range(n_iter):
        for lut in [G123_LUT, G123P_LUT]:
            N = correlate(skel, mask, mode='constant')
            D = np.take(lut,N)
            skel[D]=0
    return skel.astype(bool)

def fill_holes(B,structure=FOUR):
    B = np.array(B).astype(bool)
    seed = np.copy(B)
    seed[1:-1,1:-1] = 1
    R = reconstruction(seed,B,method='erosion',selem=structure)
    return R > 0
