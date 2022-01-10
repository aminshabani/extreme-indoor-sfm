import shapely.geometry as sg
import shapely.ops as so
import numpy as np
from shapely.ops import transform
import math
import seaborn as sns


colors = sns.color_palette("bright", 8)
colors = [[x[0] * 255, x[1] * 255, x[2] * 255, 255] for x in colors]
colors = np.array(colors, dtype=int)


rcolors = sns.color_palette("dark", 10)
rcolors = [[x[0] * 255, x[1] * 255, x[2] * 255, 200] for x in rcolors]
rcolors = np.array(rcolors, dtype=int)

def flip(x, y):
    return x, -y


def non_max_suppression_fast(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")


def pano_to_fp(point, polygon, pano_size, rot90=0):
    x = point[0]
    degree = (pano_size[1] - x) / pano_size[1] * (2 * np.pi)
    degree = (degree + (np.pi / 2 * rot90))
    ray = [(0, 0), (512 * np.cos(degree), 512 * np.sin(degree))]
    ray = sg.LineString(ray)
    intersect = polygon.exterior.intersection(ray)
    if (intersect.type == "MultiPoint"):
        intersect = intersect[0]
    if (intersect.type == "LineString"):
        return intersect, 0
    x, y = polygon.exterior.coords.xy
    for i in range(1, len(x)):
        line = sg.LineString([(x[i - 1], y[i - 1]), (x[i], y[i])])
        check = line.intersects(ray)
        if (check):
            break
    x, y = line.xy
    if (abs(x[0] - x[1]) < abs(y[0] - y[1])):
        is_vertical = 1 if (x[0] < 0) else 3
    else:
        is_vertical = 0 if (y[0] < 0) else 2
    return intersect, is_vertical


def map_pano_to_tdv(pano):
    mapping = np.zeros([2, pano.size[0], pano.size[1]])
    for i in range(pano.size[1]):
        p, _ = pano_to_fp([i, 0], pano.poly, pano.size, rot90=1)
        mapping[0, 100:, i] = np.linspace(
            p.y, 0, num=pano.size[0] - 100, endpoint=False) + 512
        mapping[1, 100:, i] = np.linspace(
            p.x, 0, num=pano.size[0] - 100, endpoint=False) + 512
    mapping = mapping.astype(int)
    return mapping

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
