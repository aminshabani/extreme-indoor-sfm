import os
from glob import glob

from . import panotools

def load_data(set_name, args):
    if set_name=='all':
        names = [line.rstrip() for line in open(args.train_set)] + [line.rstrip() for line in open(args.test_set)]
    elif set_name=='train':
        names = [line.rstrip() for line in open(args.train_set)]
    elif set_name=='test':
        names = [line.rstrip() for line in open(args.test_set)]
    else:
        print("set name is not defined properly...")
        return
    folder_list = [os.path.join(args.data_dir, x) for x in names]
    folder_list = sorted(folder_list)
    panos = []
    for folder in folder_list:
        img_list = glob(f"{folder}/images/aligned_*.png")
        for f in img_list:
            name = f.split("/")[-1][:-4]
            path = os.path.join(folder, 'images')
            pano = panotools.panorama.Panorama(path, name)
            annotation = pano.get_detectron_annotation(len(panos))
            panos.append(annotation)
    if args.det_is_eval:
        panos = [x for x in panos if len(x['annotations'])>0]
    return panos
