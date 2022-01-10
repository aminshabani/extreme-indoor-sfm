import os
import sys
from glob import glob
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import HorizonNet
from .dataset import visualize_a_data
from .misc import post_proc, panostretch, utils

from parser import config_parser


def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]


def augment(x_img, flip, rotate):
    x_img = x_img.numpy()
    aug_type = ['']
    x_imgs_augmented = [x_img]
    if flip:
        aug_type.append('flip')
        x_imgs_augmented.append(np.flip(x_img, axis=-1))
    for shift_p in rotate:
        shift = int(round(shift_p * x_img.shape[-1]))
        aug_type.append('rotate %d' % shift)
        x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
    return torch.FloatTensor(np.concatenate(x_imgs_augmented, 0)), aug_type


def augment_undo(x_imgs_augmented, aug_type):
    x_imgs_augmented = x_imgs_augmented.cpu().numpy()
    sz = x_imgs_augmented.shape[0] // len(aug_type)
    x_imgs = []
    for i, aug in enumerate(aug_type):
        x_img = x_imgs_augmented[i*sz : (i+1)*sz]
        if aug == 'flip':
            x_imgs.append(np.flip(x_img, axis=-1))
        elif aug.startswith('rotate'):
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug == '':
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()

    return np.array(x_imgs)


def inference(net, x, device, flip=False, rotate=[], visualize=False,
              force_cuboid=True, min_v=None, r=0.05):
    '''
    net   : the trained HorizonNet
    x     : tensor in shape [1, 3, 512, 1024]
    flip  : fliping testing augmentation
    rotate: horizontal rotation testing augmentation
    '''

    H, W = tuple(x.shape[2:])

    # Network feedforward (with testing augmentation)
    x, aug_type = augment(x, flip, rotate)
    y_bon_, y_cor_ = net(x.to(device))
    y_bon_ = augment_undo(y_bon_.cpu(), aug_type).mean(0)
    y_cor_ = augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)

    # Visualize raw model output
    if visualize:
        vis_out = visualize_a_data(x[0],
                                   torch.FloatTensor(y_bon_[0]),
                                   torch.FloatTensor(y_cor_[0]))
    else:
        vis_out = None

    y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
    y_cor_ = y_cor_[0, 0]

    # Init floor/ceil plane
    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

    # Detech wall-wall peaks
    if min_v is None:
        min_v = 0 if force_cuboid else 0.05
    r = int(round(W * r / 2))
    N = 4 if force_cuboid else None
    xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

    # Generate wall-walls
    cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
    if not force_cuboid:
        # Check valid (for fear self-intersection)
        xy2d = np.zeros((len(xy_cor), 2), np.float32)
        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
            xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
        if not Polygon(xy2d).is_valid:
            print(
                'Fail to generate valid general layout!! '
                'Generate cuboid as fallback.',
                file=sys.stderr)
            xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
            cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor)*2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j*2] = cor[j, 0], cor[j, 1]
        cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]

    # Normalized to [0, 1]
    cor_id[:, 0] /= W
    cor_id[:, 1] /= H

    return cor_id, z0, z1, vis_out


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    device = args.device
    # Loaded trained model
    net = utils.load_trained_model(HorizonNet, f'HorizonNet/{args.lt_model_weight}').to(device)
    net.eval()

    # test_names = [line.rstrip() for line in open(args.test_set)]
    # folder_list = [f"{args.data_dir}/{x}" for x in test_names]
    folder_list = glob(f"{args.data_dir}/*")
    print("_______________")
    print("selected houses:")
    for name in folder_list:
        print(name)
    print("_______________")
    folder_list = sorted(folder_list)
    paths = []
    for f in tqdm(folder_list):
        path = f"{f}/layout_preds"
        os.makedirs(path, exist_ok=True)
        files = glob(f'{f}/images/aligned_*.png')

        # Inferencing
        with torch.no_grad():
            for img in files:
                name = img.split('/')[-1]
                print(name)
                # Load image
                img_pil = Image.open(img)
                if img_pil.size != (1024, 512):
                    img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
                img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
                x = torch.FloatTensor([img_ori / 255])

                # Inferenceing corners
                cor_id, z0, z1, vis_out = inference(net, x, device,
                                                    args.lt_flip, args.lt_rotate,
                                                    args.lt_visualize,
                                                    args.lt_force_cuboid,
                                                    args.lt_min_v, args.lt_r)

                # Output result
                with open(os.path.join(path, name.replace('.png', '.json')), 'w') as f:
                    json.dump({
                        'z0': float(z0),
                        'z1': float(z1),
                        'uv': [[float(u), float(v)] for u, v in cor_id],
                    }, f)

                if vis_out is not None:
                    vis_path = os.path.join(path, name)
                    vh, vw = vis_out.shape[:2]
                    Image.fromarray(vis_out)\
                         .resize((vw//2, vh//2), Image.LANCZOS)\
                         .save(vis_path)
