import argparse
import numpy as np
import coloredlogs
import logging
import sys
from src.loaders.room_type_classification import dataset
from src.models.model import get_model
from tqdm import tqdm
import torch
import torch.nn as nn
import cv2
from glob import glob
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import torchvision.transforms as transforms

from parser import config_parser


logger = logging.getLogger('log')
coloredlogs.install(level="DEBUG",
                    logger=logger,
                    fmt='%(asctime)s, %(name)s, %(levelname)s %(message)s')
logging.root.setLevel(logging.INFO)


def main(args):
    parser = config_parser()
    args = parser.parse_args()
    logger.setLevel(args.log)
    model = get_model(args.rc_model, 3, 10)

    pretrained_dict = torch.load(args.rc_model_weight)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in model.state_dict()
    }
    model.load_state_dict(pretrained_dict, strict=True)
    model.eval()

    if(args.rc_is_eval):
        test_loader = dataset('test', args)
        labels = []
        preds = []
        for (img, label) in tqdm(test_loader, position=0):
            with torch.no_grad():
                img = torch.as_tensor(img,
                                        dtype=torch.float,
                                        device=torch.device('cuda'))
                img = nn.functional.interpolate(img, size=(256, 256))
                label = torch.as_tensor(label,
                                        dtype=torch.long,
                                        device=torch.device('cuda'))
                pred = model(img)
                labels.extend(label.cpu().data.numpy().tolist())
                pred = torch.argmax(pred ,1)
                preds.extend(pred.cpu().data.numpy().tolist())
            conf_mat = confusion_matrix(labels, preds)
            print(conf_mat)
            print(metrics.accuracy_score(labels, preds))
            print(metrics.average_precision_score(labels, preds))
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        folder = glob(f"{args.data_dir}/*")
        for house in tqdm(folder):
            files = glob(f'{house}/images/aligned_*.png')
            preds = dict()
            for path in files:
                name = path.split('/')[-1][8:-4]
                img = cv2.imread(path)
                img = img.astype(float)/255
                img = transform(img)
                with torch.no_grad():
                    img = torch.as_tensor(img,
                                            dtype=torch.float,
                                            device=torch.device('cuda'))
                    img = img.unsqueeze(0)
                    img = nn.functional.interpolate(img, size=(256, 256))

                    pred = model(img)
                    pred = torch.softmax(pred, -1)
                    preds[name] = pred[0].cpu().data.numpy()
            np.save(f'{house}/room_type_preds.npy', preds)


if __name__ == '__main__':
    main(sys.argv[1:])
