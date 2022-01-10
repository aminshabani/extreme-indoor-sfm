import numpy as np
import shapely.geometry as sg
import shapely.ops as so
import matplotlib.pyplot as plt
from shapely import affinity
from shapely.ops import transform
from PIL import Image, ImageDraw, ImageOps
import random
from . import tools
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

rgbcolors = [[155, 35, 53], [255, 111, 97], [107, 91, 149], [136, 176, 75],
             [247, 202, 201], [146, 168, 209], [149, 82, 81], [214, 80, 118]]


def flip(x, y):
    return x, -y


def fp(house):
    img = house.get_fp_img()
    return img


def fp_aligned_vectorized(house):
    rgbcolors = [[155, 35, 53], [255, 111, 97], [107, 91, 149], [136, 176, 75],
                 [247, 202, 201], [146, 168, 209], [149, 82, 81],
                 [214, 80, 118]]
    rgbs = dict()
    img = house.get_fp_img()
    draw = ImageDraw.Draw(img, "RGBA")
    for i, pano in enumerate(house.panos):
        rgb = random.choice(rgbcolors)
        rgbcolors.remove(rgb)
        if (len(rgbcolors) == 0):
            rgbcolors = [[155, 35, 53], [255, 111, 97], [107, 91, 149],
                         [136, 176, 75], [247, 202, 201], [146, 168, 209],
                         [149, 82, 81], [214, 80, 118]]

        rgbs[i] = rgb
        poly = transform(flip, pano.poly)
        poly = affinity.rotate(poly, pano.rot + 90, (0, 0))
        poly = affinity.translate(poly, pano.center[0], pano.center[1])
        poly = np.asarray(poly.exterior.xy)
        poly = [(poly[0][i], poly[1][i]) for i in range(poly.shape[1])]
        draw.polygon(poly, outline='black', fill=tuple(rgb + [200]))

    for i, pano in enumerate(house.panos):
        p = sg.Point(0, 0)
        p = affinity.rotate(p, pano.rot + 90, (0, 0))
        p = affinity.translate(p, pano.center[0], pano.center[1])
        draw.rectangle([p.x, p.y, p.x + 7, p.y + 7],
                       fill="white",
                       outline='black')

    for i, pano in enumerate(house.panos):
        rgb = rgbs[i]
        for door in pano.doors:
            p = door.crc[0]
            p = transform(flip, p)
            p = affinity.rotate(p, pano.rot + 90, (0, 0))
            p = affinity.translate(p, pano.center[0], pano.center[1])
            draw.ellipse([p.x, p.y, p.x + 7, p.y + 7],
                         fill=tuple(rgb),
                         outline='white')
    return img, rgbs


def fp_aligned_panos(house):
    img = house.get_fp_img()
    draw = ImageDraw.Draw(img, "RGBA")
    for i, pano in enumerate(house.panos):
        img2 = pano.get_top_down_view()
        img2 = img2.rotate(-pano.rot)
        _, _, _, alpha = img2.split()
        alpha = Image.fromarray((np.array(alpha) * 0.7).astype(np.uint8))
        img.paste(img2, [pano.center[0] - 384, pano.center[1] - 384], alpha)
    return img


def detection_results(pano, color=None, indx=None):
    if (color is None):
        color = [0, 255, 0]
    pano_img = pano.get_panorama_img()
    pano_detection = ImageDraw.Draw(pano_img)
    for i, door in enumerate(pano.doors):
        if (indx is not None and i == indx):
            bbox = door.bbox
            pano_detection.rectangle(bbox, outline='red', width=9)
        else:
            bbox = door.bbox
            pano_detection.rectangle(bbox, outline=tuple(color), width=6)
    return pano_img


def show_aligned_vectorized(house):
    rgbcolors = [[155, 35, 53], [255, 111, 97], [107, 91, 149], [136, 176, 75],
                 [247, 202, 201], [146, 168, 209], [149, 82, 81],
                 [214, 80, 118]]
    rgbs = dict()
    for i, pano in enumerate(house.panos):
        rgb = random.choice(rgbcolors)
        rgbcolors.remove(rgb)
        if (len(rgbcolors) == 0):
            rgbcolors = [[155, 35, 53], [255, 111, 97], [107, 91, 149],
                         [136, 176, 75], [247, 202, 201], [146, 168, 209],
                         [149, 82, 81], [214, 80, 118]]

        rgbs[i] = rgb
        poly = transform(flip, pano.poly)
        poly = affinity.rotate(poly, pano.rot + 90, (0, 0))
        poly = affinity.translate(poly, pano.center[0], pano.center[1])
        plt.plot(*poly.exterior.xy, color=np.array(rgb) / 256)

    for i, pano in enumerate(house.panos):
        rgb = rgbs[i]
        for door in pano.doors:
            p = door.crc[0]
            p = transform(flip, p)
            p = affinity.rotate(p, pano.rot + 90, (0, 0))
            p = affinity.translate(p, pano.center[0], pano.center[1])
            plt.scatter([p.x], [p.y], color=np.array(rgb) / 256)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()


def show_panorama_all(pano, save_to_file=False, name=None):
    plt.subplot(3, 1, 2)
    pano_img = pano.get_panorama_img()
    plt.imshow(pano_img)
    plt.axis('off')

    plt.subplot(3, 1, 1)
    tmp_poly = pano.poly
    plt.plot(*tmp_poly.exterior.xy, color="black", linewidth=1)
    plt.fill(*tmp_poly.exterior.xy, color="black", alpha=0.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    for door in pano.doors:
        p = door.crc[0]
        plt.scatter([p.x], [p.y], color="blue", marker='.')

    plt.subplot(3, 1, 3)
    plt.axis('off')

    pano_detection = ImageDraw.Draw(pano_img)
    for door in pano.doors:
        bbox = door.bbox
        pano_detection.rectangle(bbox, outline='green', width=6)

    plt.imshow(pano_img)

    plt.show()


def show_house_all(house):
    fig = plt.figure(constrained_layout=True)
    img, rgbs = fp_aligned_vectorized(house)
    if (img.size[0] > img.size[1]):
        plt.subplot(2, 1, 1)
    else:
        plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('equal')
    plt.axis('off')

    for i, pano in enumerate(house.panos):
        if (img.size[0] > img.size[1]):
            plt.subplot(2, len(house.panos), 1 * len(house.panos) + 1 + i)
        else:
            id = 2 * (i + 1)
            plt.subplot(len(house.panos), 2, id)
        plt.imshow(detection_results(pano, rgbs[i]))
        plt.axis('off')

    plt.gcf().suptitle(house.name)
    plt.tight_layout(pad=0.05)
    plt.show()
    #plt.savefig("tmp/{}.jpg".format(house.name), dpi=400)
    plt.close()


def show_tree(house, pairs, num_pos, num_neg):

    rgbcolors = [[155, 35, 53], [255, 111, 97], [107, 91, 149], [136, 176, 75],
                 [247, 202, 201], [146, 168, 209], [149, 82, 81],
                 [214, 80, 118], [0, 255, 0], [0, 0, 255], [255, 0, 0]]

    memory = dict()
    polys = []
    for i, pair in enumerate(pairs):
        is_same = pair[1]
        pair = pair[0]
        p1 = house.panos[pair[0][0]]
        d1 = p1.doors[pair[0][1]]
        p2 = house.panos[pair[1][0]]
        d2 = p2.doors[pair[1][1]]
        if (i == 0):
            tmpimg = p1.get_top_down_view_raymapping()
            plt.imshow(tmpimg, extent=[-256, 256, -256, 256])
            rgb = rgbcolors[0]#p1.room_type]
            poly = transform(flip, p1.poly)
            plt.plot(*poly.exterior.xy, color=np.array(rgb) / 255)
            polys.append(poly)
            p = d1.crc[0]
            p = transform(flip, p)
            plt.scatter([p.x], [p.y], color=np.array(rgb) / 255)
            memory[pair[0][0]] = [0, 0, 0]

        memo = memory[pair[0][0]]
        if (abs(d2.crc[1] - d1.crc[1]) == 0):
            rot = memo[2] + 180
        elif (abs(d2.crc[1] - d1.crc[1]) == 2):
            rot = memo[2]
        else:
            rot = ((d2.crc[1] * 90) - (d1.crc[1] * 90) + memo[2]) % 360

        if (is_same):
            rot = (rot + 180) % 360

        p = d1.crc[0]
        p = transform(flip, p)
        p = affinity.rotate(p, memo[2], (0, 0))
        pp = d2.crc[0]
        pp = transform(flip, pp)
        pp = affinity.rotate(pp, rot, (0, 0))
        ddist = [p.x - pp.x, p.y - pp.y]

        tmpimg = np.rot90(p2.get_top_down_view_raymapping(), rot / 90)
        plt.imshow(tmpimg,
                   extent=[
                       -256 + ddist[0] + memo[0], 256 + ddist[0] + memo[0],
                       -256 + ddist[1] + memo[1], 256 + ddist[1] + memo[1]
                   ])
        rgb = rgbcolors[0]#p2.room_type]
        poly = transform(flip, p2.poly)
        poly = affinity.rotate(poly, rot, (0, 0))
        poly = affinity.translate(poly, ddist[0] + memo[0], ddist[1] + memo[1])
        plt.plot(*poly.exterior.xy, color=np.array(rgb) / 255)
        polys.append(poly)
        p = d2.crc[0]
        p = transform(flip, p)
        p = affinity.rotate(p, rot, (0, 0))
        p = affinity.translate(p, ddist[0] + memo[0], ddist[1] + memo[1])
        plt.scatter([p.x], [p.y], color=np.array(rgb) / 255)
        memory[pair[1][0]] = [ddist[0] + memo[0], ddist[1] + memo[1], rot]

        for door in p2.doors:
            p = door.crc[0]
            p = transform(flip, p)
            p = affinity.rotate(p, rot, (0, 0))
            p = affinity.translate(p, ddist[0] + memo[0], ddist[1] + memo[1])
            plt.scatter([p.x], [p.y], color=np.array(rgb) / 255, marker='+')

    iou = 0
    for i, p1 in enumerate(polys):
        tmppolys = polys.copy()
        tmppolys.pop(i)
        union = so.unary_union(tmppolys)
        iou += p1.intersection(union).area / p1.area

    iou = round(iou / (len(pairs) + 1), 3)
    randint = np.random.randint(10000)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.savefig("tmp/{}_{}_n{}_p{}_iou{}.png".format(randint, house.name, num_neg, num_pos, iou), dpi=300)
    # plt.savefig("tmp/{}_{}_z.jpg".format(randint, house.name), dpi=400, bbox_inches='tight')
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(10,10)
    canvas = FigureCanvas(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    img = np.fromstring(s, np.uint8).reshape((height, width, 4))
    img = img[:,:,:-1]

    plt.close()

    # img = Image.open("tmp/{}_{}_n{}_p{}_iou{}.png".format(randint, house.name, num_neg, num_pos, iou))
    # img = None
    # img = Image.open("tmp/{}_{}_z.jpg".format(randint, house.name))

    image_data = np.asarray(img)
    image_data_bw = image_data.max(axis=2)
    non_empty_columns = np.where(image_data_bw.min(axis=0)<255)[0]
    non_empty_rows = np.where(image_data_bw.min(axis=1)<255)[0]
    x = np.minimum(min(non_empty_rows), min(non_empty_columns))
    y = np.maximum(max(non_empty_rows), max(non_empty_columns))
    # cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    cropBox = (x,y,x,y)
    image_data_new = image_data[cropBox[0]-5:cropBox[1]+5, cropBox[2]-5:cropBox[3]+5 , :]
    # plt.imshow(image_data_new)
    # plt.show()
    img = Image.fromarray(image_data_new.astype(np.uint8))
    '''
    # img.save("tmp/{}_{}.jpg".format(randint, house.name))

    # plt.subplot(1,2,1)
    # plt.imshow(img)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # cols = np.ceil(np.sqrt(len(pairs)+1))
    cols = len(pairs)+1
    for i, pair in enumerate(pairs):
        is_same = pair[1]
        pair = pair[0]
        p1 = house.panos[pair[0][0]]
        d1 = p1.doors[pair[0][1]]
        p2 = house.panos[pair[1][0]]
        d2 = p2.doors[pair[1][1]]
        if(i==0):
            plt.subplot(2,cols,1)
            tmpimg = p1.get_top_down_view_raymapping(apply_lv=False)
            plt.imshow(tmpimg)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            plt.subplot(2,cols,cols+1)
            # tmpimg = p1.get_top_down_view_raymapping(apply_lv=True)
            tmpimg = p1.get_panorama_img()
            plt.imshow(tmpimg)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplot(2,cols,(i+2))
        tmpimg = np.rot90(p2.get_top_down_view_raymapping(apply_lv=False), memory[pair[1][0]][2]/90)

        # image_data = np.asarray(tmpimg)
        # image_data_bw = image_data.max(axis=2)
        # non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
        # non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
        # cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
        # image_data_new = image_data[cropBox[0]-5:cropBox[1]+5, cropBox[2]-5:cropBox[3]+5 , :]
        # tmpimg = Image.fromarray(image_data_new.astype(np.uint8))

        plt.imshow(tmpimg)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.subplot(2,cols,cols+(i+2))
        # tmpimg = np.rot90(p2.get_top_down_view_raymapping(apply_lv=True), memory[pair[1][0]][2]/90)
        tmpimg = p2.get_panorama_img()
        plt.imshow(tmpimg)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.gcf().tight_layout()
    plt.savefig("tmp/{}_{}_panos.jpg".format(randint, house.name), dpi=400, bbox_inches='tight')
    # plt.show()
    plt.close()
    '''
    return img, iou


def show_pair_panos(house, indx, is_negative=False):
    if (is_negative):
        pair = house.negatives[indx]
    else:
        pair = house.positives[indx]
    p1, d1 = house.dindex_to_panoindex(pair[1])
    p2, d2 = house.dindex_to_panoindex(pair[2])

    rgbcolors = [[155, 35, 53], [255, 111, 97], [107, 91, 149], [136, 176, 75],
                 [247, 202, 201], [146, 168, 209], [149, 82, 81],
                 [214, 80, 118]]
    rgbs = dict()
    for i, pano in enumerate(house.panos):
        rgb = random.choice(rgbcolors)
        rgbcolors.remove(rgb)
        if (len(rgbcolors) == 0):
            rgbcolors = [[155, 35, 53], [255, 111, 97], [107, 91, 149],
                         [136, 176, 75], [247, 202, 201], [146, 168, 209],
                         [149, 82, 81], [214, 80, 118]]

        rgbs[i] = rgb
    plt.subplot(2, 1, 1)
    plt.imshow(detection_results(house.panos[p1], rgbs[i], d1))
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(detection_results(house.panos[p2], rgbs[i], d2))
    plt.axis('off')

    plt.gcf().suptitle("{}_{}".format(house.name, not is_negative), y=0.05)
    #plt.tight_layout(pad=0.05)
    #plt.show()
    plt.savefig("tmp/{}_{}_{}_{}.jpg".format(np.random.randint(1000),
                                             house.name, indx,
                                             not is_negative),
                dpi=400)
    plt.close()


def show_pair(house, indx, is_negative=False):
    if (is_negative):
        pair = house.negatives[indx]
    else:
        pair = house.positives[indx]
    p1, d1 = house.dindex_to_panoindex(pair[1])
    p2, d2 = house.dindex_to_panoindex(pair[2])
    dist = pair[0]

    rgbcolors = [[155, 35, 53], [255, 111, 97], [107, 91, 149], [136, 176, 75],
                 [247, 202, 201], [146, 168, 209], [149, 82, 81],
                 [214, 80, 118]]
    rgbs = dict()
    img = house.get_fp_img()
    draw = ImageDraw.Draw(img, "RGBA")
    for i, pano in enumerate(house.panos):
        rgb = random.choice(rgbcolors)
        rgbcolors.remove(rgb)
        if (len(rgbcolors) == 0):
            rgbcolors = [[155, 35, 53], [255, 111, 97], [107, 91, 149],
                         [136, 176, 75], [247, 202, 201], [146, 168, 209],
                         [149, 82, 81], [214, 80, 118]]

        rgbs[i] = rgb
        poly = transform(flip, pano.poly)
        poly = affinity.rotate(poly, pano.rot + 90, (0, 0))
        poly = affinity.translate(poly, pano.center[0], pano.center[1])
        poly = np.asarray(poly.exterior.xy)
        poly = [(poly[0][i], poly[1][i]) for i in range(poly.shape[1])]
        if (i == p1):
            draw.polygon(poly, outline='black', fill=tuple(rgb + [200]))
        elif (i == p2):
            draw.polygon(poly, outline='black', fill=tuple(rgb + [200]))
        else:
            draw.polygon(poly, fill=tuple(rgb + [200]))

    for i, pano in enumerate(house.panos):
        p = sg.Point(0, 0)
        p = affinity.rotate(p, pano.rot + 90, (0, 0))
        p = affinity.translate(p, pano.center[0], pano.center[1])
        if (i == p1 or i == p2):
            draw.rectangle([p.x - 3, p.y - 3, p.x + 3, p.y + 3],
                           fill="black",
                           outline='black')
        else:
            draw.rectangle([p.x - 3, p.y - 3, p.x + 3, p.y + 3],
                           fill="white",
                           outline='black')

    for i, pano in enumerate(house.panos):
        rgb = rgbs[i]
        for j, door in enumerate(pano.doors):
            p = door.crc[0]
            p = transform(flip, p)
            p = affinity.rotate(p, pano.rot + 90, (0, 0))
            p = affinity.translate(p, pano.center[0], pano.center[1])
            if ((i == p1 and j == d1) or (i == p2 and j == d2)):
                continue
            else:
                draw.ellipse([p.x - 3, p.y - 3, p.x + 3, p.y + 3],
                             fill=tuple(rgb),
                             outline='white')

    for i, pano in enumerate(house.panos):
        rgb = rgbs[i]
        for j, door in enumerate(pano.doors):
            p = door.crc[0]
            p = transform(flip, p)
            p = affinity.rotate(p, pano.rot + 90, (0, 0))
            p = affinity.translate(p, pano.center[0], pano.center[1])
            if ((i == p1 and j == d1) or (i == p2 and j == d2)):
                draw.ellipse([p.x - 5, p.y - 5, p.x + 5, p.y + 5],
                             fill='black',
                             outline='red')
            else:
                continue

    if (img.size[0] > img.size[1]):
        plt.subplot(2, 1, 1)
    else:
        plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('equal')
    plt.axis('off')

    for i, pano in enumerate(house.panos):
        if (img.size[0] > img.size[1]):
            plt.subplot(2, len(house.panos), 1 * len(house.panos) + 1 + i)
        else:
            id = 2 * (i + 1)
            plt.subplot(len(house.panos), 2, id)
        if (i == p1):
            plt.imshow(detection_results(pano, rgbs[i], d1))
        elif (i == p2):
            plt.imshow(detection_results(pano, rgbs[i], d2))
        else:
            plt.imshow(detection_results(pano, rgbs[i]))
        plt.axis('off')

    plt.gcf().suptitle("{}_{}_{}".format(house.name, not is_negative,
                                         round(dist, 2)),
                       y=0.05)
    plt.tight_layout(pad=0.05)
    #plt.show()
    plt.savefig("tmp/{}_{}_{}_{}.jpg".format(np.random.randint(1000),
                                             house.name, indx,
                                             not is_negative),
                dpi=400)
    plt.close()
