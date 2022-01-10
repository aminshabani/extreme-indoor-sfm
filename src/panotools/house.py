import copy
import numpy as np
import shapely.geometry as sg
from shapely.ops import nearest_points
from shapely import affinity
import os
from . import visualize as vis
from .panorama import Panorama
from .tree import Tree
import json
from tqdm import tqdm
from PIL import Image

class House:
    def __init__(self, name: str, args):
        print(f"Loading house #{name}")
        self.path = args.data_dir
        self.name = name
        self.labeled = False
        self.fp = f"{self.path}/{self.name}/floorplan.jpg"
        if not os.path.isfile(self.fp):
            self.fp = None
        if os.path.exists(f"{self.path}/{self.name}/gt_labels/house_label.json"):
            self.labeled = True
        else:
            if args.use_gt:
                raise Exception("ERROR: no label is available but use_gt is enabled...")
        self.panos = []
        if self.labeled:
            data = json.load(open(f"{self.path}/{self.name}/gt_labels/house_label.json"))
            self.pano_names = data['pano_names']
            self.rotations = data['rotations']
            self.room_types = data['room_types']
            self.positions = data['positions']
            self.center_of_mass = np.mean(
                np.array(list(self.positions.values())), 0)
            self.scale = data['scales'][0]
            self.pano_scale = data['scales'][1]
            for name in self.pano_names:
                self.panos.append(Panorama(self.name, name, args, self.room_types[int(name)]))
        else:
            self.scale = 1
            self.room_types = np.load(f"{self.path}/{self.name}/room_type_preds.npy", allow_pickle=True).item()
            self.pano_names = [x[8:-4] for x in os.listdir(f"{self.path}/{self.name}/images")]
            for name in self.pano_names:
                self.panos.append(Panorama(self.name, name, args, self.room_types[name]))

        print(f"Number of panoramas: {len(self.panos)}")

        self.positive_pairs = []
        self.negative_pairs = []
        self.pairs = []

        self.create_pairs(args.use_rotations_input)
        if args.use_gt:
            self.create_gt_pairs()
        self.pair_mark = np.zeros(len(self.pairs))
        print(f"Total door pairs: {len(self.pairs)}, GT_pairs: {len(self.positive_pairs)}")

        self.gt_trees = []
        self.strong_positive_trees = []
        self.weak_positive_trees = []
        self.negative_trees = []


        # create sets
        self.gt_trees = self.create_trees(self.positive_pairs)
        print(f'number of gt alignments: {len(self.gt_trees)} ...')
        for tree in tqdm(self.gt_trees, desc="generating GT alignments"):
            tree.fix_positions(self)

        self.negative_trees = self.create_trees(self.pairs)


        # def run_func(tree):
            # tree.fix_positions(self)
        # from concurrent.futures import ThreadPoolExecutor
        # with ThreadPoolExecutor() as executor:
            # tqdm(executor.map(run_func, self.negative_trees[:20000]))


        print(f'number of alignments: {len(self.negative_trees)} ...')
        for tree in tqdm(self.negative_trees, desc="generating alignments"):
            tree.fix_positions(self)
        print(f'{len(self.negative_trees)} alignments is created...')

        # remove gt from negative set
        print('removing GT alignments from all...')
        remove_indices = []
        for i, tree in enumerate(self.negative_trees):
            for tree2 in self.gt_trees:
                if tree.is_equal(tree2):
                    remove_indices.append(i)
                    break
        self.negative_trees = [self.negative_trees[i] for i in range(len(self.negative_trees)) if i not in remove_indices]
        print(f'final number of alignments: {len(self.negative_trees)}')

        # create the rest
        self.weak_positive_trees = [tree for tree in self.negative_trees if tree.iou < 0.1]
        self.strong_positive_trees = [tree for tree in self.weak_positive_trees if tree.check_type_conditions(self)]

        # exclude sets
        self.negative_trees = [tree for tree in self.negative_trees if tree not in self.weak_positive_trees]
        self.weak_positive_trees = [tree for tree in self.weak_positive_trees if tree not in self.strong_positive_trees]


        if args.keep_sets_overlapped:
            self.strong_positive_trees.extend(self.gt_trees)
            self.weak_positive_trees.extend(self.strong_positive_trees)
            self.negative_trees.extend(self.weak_positive_trees)
        print(f'final number of alignments in each set: {len(self.negative_trees)}, {len(self.weak_positive_trees)}, {len(self.strong_positive_trees)}')

    def get_fp_img(self, type="RGB"):
        img = Image.open(self.fp).convert(type)
        img = img.resize(
            (int(img.size[0] * self.scale), int(img.size[1] * self.scale)))
        return img

    def dindex_to_panoindex(self, index):
        for i, pano in enumerate(self.panos):
            if (index < len(pano.doors)):
                return i, int(index)
            index -= len(pano.doors)

    def get_alignment(self, exp=None):
        fp = self.get_fp_img()
        for i in self.positions:
            pos = self.positions[i]
            i = int(i)
            pano = self.panos[i].get_top_down_view()
            pano = pano.rotate(-90 * self.rotations[i])
            pano = pano.resize(
                (int(pano.size[0] * self.pano_scale), int(pano.size[1] * self.pano_scale)))
            pano = pano.crop(
                (-pos[0], -pos[1], fp.size[0] - pos[0], fp.size[1] - pos[1]))
            alpha = pano.split()[-1]
            fp = Image.composite(pano, fp, alpha)
        return fp

    def save_alignment(self):
        img = self.get_alignment()
        # os.makedirs('tmp/alignments/{}'.format(exp, self.name), exist_ok=True)
        img.save("tmp/alignments/{}.png".format(self.name))

    def create_gt_pairs(self):
        objs = []
        for name in self.positions:
            pano = self.panos[int(name)]
            for j, obj in enumerate(pano.obj_list):
                bbox = obj.bbox * 25.6 + 256
                obj = sg.LineString(
                    [(bbox[0][0], bbox[0][2]), (bbox[1][0], bbox[1][2])])
                obj = affinity.rotate(
                    obj, 90 * self.rotations[int(name)], (256, 256))
                obj = affinity.translate(
                    obj, self.positions[name][0], self.positions[name][1])
                objs.append(
                    [obj, int(name), j, self.rotations[int(name)]+pano.obj_list[j].direction])
        dists = np.zeros([len(objs), len(objs)]) + 1e10
        for i in range(len(objs)):
            for j in range(len(objs)):
                if i == j:
                    continue
                tmp = nearest_points(objs[i][0].centroid, objs[j][0])
                d = tmp[1].distance(objs[i][0].centroid)
                dists[i, j] = d
        dists = np.round(dists, 3)
        args = np.argmin(dists, 1)
        dists = np.min(dists, 1)
        for i in range(len(objs)):
            for j in range(i+1, len(objs)):
                if args[i] == j and args[j] == i and dists[i] < 10:
                    if abs(objs[i][3]-objs[j][3]) % 4 == 2:
                        self.positive_pairs.append(
                            [objs[i][1:3], objs[j][1:3]])
                else:
                    # type1 = self.panos[objs[i][1]
                    #                    ].obj_list[objs[i][2]].get_type()
                    # type2 = self.panos[objs[j][1]
                    #                    ].obj_list[objs[j][2]].get_type()
                    # if type1 != type2:
                    self.negative_pairs.append([objs[i][1:], objs[j][1:]])

    def create_pairs(self, use_rotations=False):
        for i in range(len(self.panos)):
            for j in range(i+1, len(self.panos)):
                for d1 in range(len(self.panos[i].obj_list)):
                    if self.panos[i].obj_list[d1].get_type() == 3:
                        continue  # ignore windows
                    for d2 in range(len(self.panos[j].obj_list)):
                        if self.panos[j].obj_list[d2].get_type() == 3:
                            continue  # ignore windows
                        if use_rotations:
                            rotdiff = (((self.rotations[i] - self.rotations[j])) %
                                       4) + (self.panos[i].obj_list[d1].direction - self.panos[j].obj_list[d2].direction)
                            rotdiff = rotdiff % 4
                            if rotdiff == 2: # in case of having rotations as inputs
                                self.pairs.append([[i, d1], [j, d2]])
                        else:
                            self.pairs.append([[i, d1], [j, d2]])

    def create_trees(self, pairs):
        tree_array = []
        tree = Tree()
        self.DFS(None, tree, tree_array, pairs)
        # if len(tree_array) > 0: # no need, just keep the len(self.positions)
        #     max_size_tree = max(tree_array, key=len)  # just keep maximum size trees
        #     tree_array = [x for x in tree_array if len(x) == max_size_tree]
        return tree_array

    def DFS(self, pindx, tree, tree_array, pairs):
        if pindx is not None:
            pair = pairs[pindx]
            tree.add_pair(pair, self)

        if len(tree) == len(self.pano_names):
            tree_array.append(copy.copy(tree))

        def check_pair_once(y):
            for x in tree.pairs_list:
                if (x[0] == y).all() or (x[1] == y).all():
                    return False
            return True

        tmp_mark = []
        for i, pair in enumerate(pairs):
            if self.pair_mark[i] > 0:
                continue
            if pair[0][0] in tree.rooms:
                if pair[1][0] in tree.rooms:
                    continue
                if not check_pair_once(np.array(pair[0])):
                    continue
            else:
                if pair[1][0] not in tree.rooms and pindx is not None:
                    continue
                elif not check_pair_once(np.array(pair[1])):
                    continue

            self.pair_mark[i] = 1
            tmp_mark.append(i)
            self.DFS(i, tree, tree_array, pairs)

        for x in tmp_mark:
            self.pair_mark[x] = 0
        if pindx is not None:
            tree.drop_last()

    def visualize_alignment(self, args):
        fp = self.get_fp_img()
        for i in self.positions:
            pos = self.positions[i]
            i = int(i)
            pano = self.panos[i].get_top_down_view(args)
            pano = pano.rotate(-90 * self.rotations[i])
            pano = pano.resize(
                (int(pano.size[0] * self.pano_scale), int(pano.size[1] * self.pano_scale)))
            pano = pano.crop(
                (-pos[0], -pos[1], fp.size[0] - pos[0], fp.size[1] - pos[1]))
            alpha = pano.split()[-1]
            fp = Image.composite(pano, fp, alpha)
        # fp.show()
        # import matplotlib.pyplot as plt
        # plt.imshow(fp)
        # plt.show()
        os.makedirs(f'outputs/{args.ae_exp}/{self.name}', exist_ok=True)
        fp.save(f'outputs/{args.ae_exp}/{self.name}/floorplan.png')


#####################################################################################


if __name__ == "__main__":
    import time
