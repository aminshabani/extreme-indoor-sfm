import os
import glob
import sys
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



houses = glob.glob('clean_data/*')
houses.sort()
for house in houses:
    if(os.path.isfile('{}/room_types.txt'.format(house))):
        continue
    print(house)
    house_name = house.split('/')[1]
    ricoh_data = json.load(open('annotations/{}.json'.format(house_name)))
    ricoh_data = ricoh_data['images']
    ricoh_data = [[x['file_name'][:-4], x['room_type']] for x in ricoh_data]
    mapping = {'Washing_room': 7, 'Bathroom': 8, 'Kitchen': 5, 'Balcony': 0, 'Toilet': 9,
               'Japanese-style_room': 3, 'Verandah': 0, 'Western-style_room': 2, 'Entrance': 6}
    ricoh_data = [[x[0], mapping[x[1]]] for x in ricoh_data]
    for x in ricoh_data:
        def press(event):
            print('press', event.key)
            if event.key=='1':
                print('got LDK...')
                x[1] = 4
                plt.close()
            elif event.key=='2':
                print('got western...')
                x[1] = 2
                plt.close()
        if x[1] == 2:
            img = Image.open('{}/aligned_{}.png'.format(house, x[0]))
            fig, ax = plt.subplots()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.imshow(img)
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            plt.show()
    with open('{}/room_types.txt'.format(house),'w') as f:
        for x in ricoh_data:
            f.write("pano: {} \t type: {} \n".format(x[0], x[1]))
            