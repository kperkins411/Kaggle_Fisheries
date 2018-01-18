'''
zips out and retreived json annotation files created by kaggle user
'''

import ujson as json
import os
from keras.utils.data_utils import get_file
from PIL import Image
import utils_data as ud
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

import numpy as np

def download_annotations(path,force_reload=False):
    '''
    download annot_urls from url_prefix if the files dont exits locally or force_reload==True
    :param force_reload:
    :return:
    '''
    cache_subdir = os.path.abspath(os.path.join(path, 'annos'))

    # if already there and no force_reload then do not reload!
    if os.path.exists(os.path.join(cache_subdir,'alb_labels.json')) and force_reload is False:
        return

    annot_urls = {
        '5458/bet_labels.json': 'bd20591439b650f44b36b72a98d3ce27',
        '5459/shark_labels.json': '94b1b3110ca58ff4788fb659eda7da90',
        '5460/dol_labels.json': '91a25d29a29b7e8b8d7a8770355993de',
        '5461/yft_labels.json': '9ef63caad8f076457d48a21986d81ddc',
        '5462/alb_labels.json': '731c74d347748b5272042f0661dad37c',
        '5463/lag_labels.json': '92d75d9218c3333ac31d74125f2b380a'
    }
    url_prefix = 'https://kaggle2.blob.core.windows.net/forum-message-attachments/147157/'

    if not os.path.exists(cache_subdir):
        os.makedirs(cache_subdir)

    for url_suffix, md5_hash in annot_urls.items():
        fname = url_suffix.rsplit('/', 1)[-1]
        get_file(fname, url_prefix + url_suffix, cache_subdir=cache_subdir, md5_hash=md5_hash)

def get_files_and_sizes(path,dir):
    '''
    recursively iterate over all dirs and files in path/dir
    and generate a list of files
    :param path:
    :param dir:
    :return:
    '''
    dirs_and_paths=[]
    data_dir = os.path.join(path,dir)
    dirs = os.listdir(data_dir)
    for dir in dirs:
        if dir[0] is '.':
            continue
        d = os.path.join(data_dir, dir)
        files = os.listdir(d)
        for f in files:
            fqn = os.path.join(d, f)
            # get the size of the image file
            size = Image.open(os.path.join(d, f)).size

            dirs_and_paths.append((f,size, fqn))
    return dirs_and_paths

def scale_boundingbox(bb_json, file,filesize):
    '''
    going to scale the bounding boxes in bb_json to 224x224

    :param file: name of an image file
    :param filesize: its original size
    :return:
    '''
    scale_x = 224./filesize[0]
    scale_y = 224./filesize[1]

    bb=bb_json[file]

    bb['height']    = bb['height']* scale_y
    bb['width' ]    = bb['width'] * scale_x
    bb['y']         = max(bb['y'] * scale_y,0)
    bb['x']         = max(bb['x'] * scale_x,0)

    bb_json[file] = bb
    pass

def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def main():
    anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']

    path = os.getcwd()
    download_annotations(path)

    # //get a list of all images and coordinates from
    bb_json = {}
    for c in anno_classes:
        if c == 'other': continue  # no annotation file for "other" class
        f = '{}/annos/{}_labels.json'.format(path, c)
        j = json.load(open(f, 'r'))
        for l in j:
            if 'annotations' in l.keys() and len(l['annotations']) > 0:
                bb_json[l['filename'].split('/')[-1]] = sorted(
                    l['annotations'], key=lambda x: x['height'] * x['width'])[-1]

    print(bb_json['img_04908.jpg'])

    files = []
    files_and_sizes = []
    import settings

    data_dir = os.path.join(path, settings.DATA)

    files_and_sizes.extend(get_files_and_sizes(data_dir, settings.TRAIN_FOLDER_NAME))
    files_and_sizes.extend(get_files_and_sizes(data_dir, settings.VALIDATE_FOLDER_NAME))
    files_and_sizes.extend(get_files_and_sizes(data_dir, settings.TEST_FOLDER_NAME))

    # raw_filenames = [f[0].split('/')[-1] for f in dirs_and_files]

    empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}

    # if we have a file that is not annotated then annotate it with an empty box
    for entry in files_and_sizes:
        if not entry[0] in bb_json.keys():
            bb_json[entry[0]] = empty_bbox

    # now lets scale all images in data folders to 224x224
    for entry in files_and_sizes:
        scale_boundingbox(bb_json,entry[0], entry[1])
        bb = bb_json[entry[0]]

    #display a few images and bounding boxes
    for i in range(10):
        fn = files_and_sizes[i][0]
        fqn = files_and_sizes[i][2]
        bb = bb_json[fn]

        img = mpimg.imread(fqn)
        plt.imshow( img)

        # Create a Rectangle patch
        rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')

        plt.add_patch(rect)




if __name__ == '__main__':
    main()