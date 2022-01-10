import shapely.geometry as sg
import shapely.ops as so
import numpy as np
from shapely.ops import transform
import math
import seaborn as sns

def uv2coords(uv):
    coordsX = uv[0] / (2 * math.pi) + 0.5
    coordsY = -uv[1] / math.pi + 0.5
    coords = (coordsX, coordsY)
    return coords


def xyz2uv(xyz):
    normXZ = math.sqrt(math.pow(xyz[0], 2) + math.pow(xyz[2], 2))
    if normXZ < 0.000001:
        normXZ = 0.000001
    normXYZ = math.sqrt(
        math.pow(xyz[0], 2) + math.pow(xyz[1], 2) + math.pow(xyz[2], 2))
    v = math.asin(xyz[1] / normXYZ)
    u = math.asin(xyz[0] / normXZ)
    if xyz[2] > 0 and u > 0:
        u = math.pi - u
    elif xyz[2] > 0 and u < 0:
        u = -math.pi - u
    uv = (u, v)
    return uv


def xyz2coords(xyz):
    uv = xyz2uv(xyz)
    coords = uv2coords(uv)
    return coords
