import os
import sys
from shutil import copy2
from glob import glob
from panotools.panorama import Panorama

if __name__ == '__main__':
    folder_list = glob('clean_data/*')
    folder_list.sort()
    imgpath = []
    for folder in folder_list:
        file_list = glob('{}/aligned_*.json'.format(folder))
        for f in file_list:
            imgpath.append(f)

    for f in imgpath[:-20]:
        house_name = f.split('/')[1]
        pano_name = f.split('/')[2][8:-5]
        copy2(f.replace('json','png'),'img/{}_{}.png'.format(house_name,pano_name))
        pano = Panorama('{}/{}'.format('clean_data',house_name), 'aligned_{}'.format(pano_name))
        with open('label_cor/{}_{}.txt'.format(house_name,pano_name), 'w') as fc:
            points = pano.get_layout_points()
            for p in points:
                p = [int(x) for x in p]
                fc.write('{} {}\n'.format(p[0], p[1]))
    for f in imgpath[-20:]:
        house_name = f.split('/')[1]
        pano_name = f.split('/')[2][8:-5]
        copy2(f.replace('json','png'),'val/img/{}_{}.png'.format(house_name,pano_name))
        pano = Panorama('{}/{}'.format('clean_data',house_name), 'aligned_{}'.format(pano_name))
        with open('val/label_cor/{}_{}.txt'.format(house_name,pano_name), 'w') as fc:
            points = pano.get_layout_points()
            for p in points:
                p = [int(x) for x in p]
                fc.write('{} {}\n'.format(p[0], p[1]))



