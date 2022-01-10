import numpy as np
import coloredlogs
import logging
import os
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import sys
import functools
from PIL import Image

from parser import config_parser
from src.panotools.house import House
from src.models.model import get_model
from src.panotools import visualize

logger = logging.getLogger('log')
coloredlogs.install(level="DEBUG",
                    logger=logger,
                    fmt='%(asctime)s, %(name)s, %(levelname)s %(message)s')
logging.root.setLevel(logging.INFO)

def main(args):
    print("____________args____________")
    for key in args.__dict__:
        print("{}: {}".format(key, args.__dict__[key]))
    print("____________________________")

    logger.setLevel(args.log)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    model = get_model(args.ar_model, inp_dim=3)
    pretrained_dict = torch.load(args.ar_model_weight)
    model.load_state_dict(pretrained_dict, strict=True)
    iter = 0
    model.eval()
    os.makedirs(f'outputs/{args.ar_exp}', exist_ok=True)

    train_names = [line.rstrip() for line in open(args.train_set)]
    test_names = [line.rstrip() for line in open(args.test_set)]

    train_houses = []
    test_houses = []

    for name in train_names:
        train_houses.append(House(name, args))
    for name in test_names:
        test_houses.append(House(name, args))

    for house in tqdm(test_houses, position=0, desc='Houses'):
        with torch.no_grad():
            if(args.prediction_level=='lv1'):
                samples = house.strong_positive_trees
            elif(args.prediction_level=='lv2'):
                samples = house.weak_positive_trees
            else:
                samples = house.negative_trees
            for tree in tqdm(samples, desc='processing alignments'):
                masks = tree.get_masks(house, False)
                output_masks = [transform(np.ones((256, 256, 16), dtype=float)*0.5)]
                for i, mask in enumerate(masks):
                    mask = mask.astype(float)/255
                    output_masks.append(transform(mask))
                imgs = torch.stack(output_masks).unsqueeze(0)

                iter += 1
                imgs = torch.as_tensor(imgs,
                                    dtype=torch.float,
                                    device=torch.device('cuda'))
                imgs = imgs.transpose(1, 0)

                pred = model(imgs)
                pred = pred.squeeze(1)
                nppred = torch.tanh(pred).data.cpu().numpy()

                os.makedirs(f'outputs/{args.ar_exp}/{house.name}', exist_ok=True)
                align_img = [np.array(visualize.show_tree(house, tree))]
                align_img = Image.fromarray(align_img[0])
                align_img.convert('RGB').save(f'outputs/{args.ar_exp}/{house.name}/{round(max(nppred)*100000)}_{iter}.png')
                # if house.labeled:
                    # house.visualize_alignment(args)

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)
