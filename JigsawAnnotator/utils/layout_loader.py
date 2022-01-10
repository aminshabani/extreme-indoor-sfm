import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
import math

colors = sns.color_palette("bright", 8)
colors = [[x[0] * 255, x[1] * 255, x[2] * 255, 255] for x in colors]
colors = np.array(colors, dtype=int)


rcolors = sns.color_palette("dark", 10)
rcolors = [[x[0] * 255, x[1] * 255, x[2] * 255, 200] for x in rcolors]
rcolors = np.array(rcolors, dtype=int)
#################################### Maapings
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


####################################


def get_pano_image(path):
    img = Image.open('{}.png'.format(path))
    img.show()


def get_tdv(path, room_type, room_color, door_color):
    res = Image.new('RGBA', (512, 512), 0)
    draw = ImageDraw.Draw(res)
    jsdata = json.load(open('{}.json'.format(path)))
    lp = jsdata['layoutPoints']
    lp = np.array([x['xyz'] for x in lp['points']])
    lo = jsdata['layoutObj2ds']['obj2ds']
    lt = np.array([x['obj_type'] for x in lo])
    lo = np.array([x['points'] for x in lo])

    lp = (lp * 25.6) + 256
    lp = [(x[0], x[2]) for x in lp]
    lo = (lo * 25.6) + 256
    if(room_color):
        draw.polygon(lp, outline=(0, 0, 0, 255), fill=tuple(rcolors[room_type]))
    else:
        draw.polygon(lp, outline=(0, 0, 0, 255), fill=(0,0,0,200))
    for i, x in enumerate(lo):
        if(door_color):
            draw.line([x[0, 0], x[0, 2], x[1, 0], x[1, 2]],
                      width=3,
                      fill=tuple(colors[lt[i]]))
        else:
            draw.line([x[0, 0], x[0, 2], x[1, 0], x[1, 2]],
                      width=3,
                      fill=(255,255,255,255))
    draw.ellipse([255, 255, 257, 257], fill=(255, 255, 255, 255))
    return res


def get_pano_mask(path, overimage=False):
    dpi = 100
    res = Image.new('RGBA', (1024, 512), 0)
    if (overimage):
        res = Image.open('{}.png'.format(path)).convert('RGBA')
    draw = ImageDraw.Draw(res)
    jsdata = json.load(open('{}.json'.format(path)))
    hf = jsdata['cameraHeight']
    hc = jsdata['cameraCeilingHeight']

    lp = jsdata['layoutPoints']
    lp = np.array([x['xyz'] for x in lp['points']])

    lo = jsdata['layoutObj2ds']['obj2ds']
    lt = np.array([x['obj_type'] for x in lo])
    lo = np.array([x['points'] for x in lo])

    ## DRAW Layout
    for x in lp:
        x[1] = hf
        p1 = list(xyz2coords(x))
        p1[0] = p1[0] * 1024
        p1[1] = 512 - p1[1] * 512
        draw.ellipse([p1[0] - 3, p1[1] - 3, p1[0] + 3, p1[1] + 3],
                     fill=(255, 0, 0, 255))
        x[1] = hc
        p2 = list(xyz2coords(x))
        p2[0] = p2[0] * 1024
        p2[1] = p2[1] * 512
        draw.ellipse([p2[0] - 3, p2[1] - 3, p2[0] + 3, p2[1] + 3],
                     fill=(255, 0, 0, 255))
        draw.line([p1[0], p1[1], p2[0], p2[1]], width=3, fill=(0, 0, 255, 255))

    # Draw Floor
    for i in range(len(lp)):
        p1 = lp[i]
        p2 = lp[(i + 1) % len(lp)]
        # pc = (p1+p2)/2
        x = np.linspace(p1[0], p2[0], num=dpi, endpoint=True)
        y = np.linspace(p1[2], p2[2], num=dpi, endpoint=True)
        xy = []
        for j in range(len(x)):
            p = np.array(xyz2coords([x[j], hf, y[j]])) * [1024, 512]
            xy.extend([p[0], 512 - p[1]])

        for j in range(0, len(xy) - 2, 2):
            if (xy[j + 2] - xy[j] < 0):
                draw.line(xy[:j + 2], width=3, fill=(0, 0, 255, 255))
                draw.line(xy[j + 2:], width=3, fill=(0, 0, 255, 255))
                break
        else:
            draw.line(xy, width=3, fill=(0, 0, 255, 255))

    # Draw Ceil
    for i in range(len(lp)):
        p1 = lp[i]
        p2 = lp[(i + 1) % len(lp)]
        # pc = (p1+p2)/2
        x = np.linspace(p1[0], p2[0], num=dpi, endpoint=True)
        y = np.linspace(p1[2], p2[2], num=dpi, endpoint=True)
        xy = []
        for j in range(len(x)):
            p = np.array(xyz2coords([x[j], hc, y[j]])) * [1024, 512]
            xy.extend([p[0], p[1]])

        for j in range(0, len(xy) - 2, 2):
            if (xy[j + 2] - xy[j] < 0):
                draw.line(xy[:j + 2], width=3, fill=(0, 0, 255, 255))
                draw.line(xy[j + 2:], width=3, fill=(0, 0, 255, 255))
                break
        else:
            draw.line(xy, width=3, fill=(0, 0, 255, 255))

    ## DRAW OBJECTS
    lo = np.round(lo, 5)
    for i, x in enumerate(lo):
        xmin = min(x[0, 0], x[1, 0])
        ymin = min(x[0, 1], x[1, 1])
        zmin = min(x[0, 2], x[1, 2])
        xmax = max(x[0, 0], x[1, 0])
        ymax = max(x[0, 1], x[1, 1])
        zmax = max(x[0, 2], x[1, 2])
        xy = []
        if (zmax == zmin):
            ps = [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin],
                  [xmin, ymax, zmin]]
            x = np.append(np.linspace(xmin, xmax, num=dpi, endpoint=True),
                          np.linspace(xmax, xmin, num=dpi, endpoint=True),
                          axis=0)
            y = np.append(np.linspace(ymin, ymin, num=dpi, endpoint=True),
                          np.linspace(ymax, ymax, num=dpi, endpoint=True),
                          axis=0)
            for j in range(len(x)):
                p = np.array(xyz2coords([x[j], y[j], zmin])) * [1024, 512]
                xy.extend([p[0], p[1]])
        else:
            ps = [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmax],
                  [xmin, ymax, zmin]]
            z = np.append(np.linspace(zmin, zmax, num=dpi, endpoint=True),
                          np.linspace(zmax, zmin, num=dpi, endpoint=True),
                          axis=0)
            y = np.append(np.linspace(ymin, ymin, num=dpi, endpoint=True),
                          np.linspace(ymax, ymax, num=dpi, endpoint=True),
                          axis=0)
            for j in range(len(z)):
                p = np.array(xyz2coords([xmin, y[j], z[j]])) * [1024, 512]
                xy.extend([p[0], p[1]])

        test_oversized = []
        for j in range(2, len(xy) - 2, 2):
            if (xy[j] == xy[j + 2]):
                continue
            if (xy[j] < xy[j - 2] and xy[j] < xy[j + 2]):
                test_oversized.append(j)
        if (len(test_oversized) == 0):
            draw.polygon(xy, fill=tuple(colors[lt[i]]))
        else:
            draw.polygon(xy[test_oversized[0] + 2:test_oversized[1] - 2],
                         fill=tuple(colors[lt[i]]))
            del xy[test_oversized[0] + 2:test_oversized[1]]
            draw.polygon(xy, fill=tuple(colors[lt[i]]))

    # res.show()
    # plt.imshow(np.array(res))
    # plt.axis('off')
    # plt.show()
    return res


# get_pano_mask("clean_data/10A0iX/aligned_1",False)
