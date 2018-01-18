import bcolz
import numpy as np
from keras.preprocessing import image, sequence
import keras.utils as ku
import os
import settings
from matplotlib import pyplot as plt

def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]

def onehot(x):
    return ku.to_categorical(x)

# for save/load processed images (speedup)
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()

def load_array(path, fname):
    try:
        # load if it exists
        processed_file = os.path.join(os.path.join(path,settings.PRE_PROCESSED_IMAGES),fname)
        return bcolz.open(processed_file)[:]
    except FileNotFoundError:
        # uh oh, does not exists, so generate it (time consumming)
        pth = os.path.join(path,fname)
        data = get_data(pth)

        # and save it for future load (big speedup)
        save_array(processed_file,data)
        return data

# get one batch at a time (lower memory cost)
def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

# get the entire dataset (higher memory cost)
def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.samples)])

def get_classes(path):
    batches = get_batches(path+settings.TRAIN_FOLDER_NAME, shuffle=False, batch_size=1)
    val_batches = get_batches(path+settings.VALIDATE_FOLDER_NAME, shuffle=False, batch_size=1)
    test_batches = get_batches(path+settings.TEST_FOLDER_NAME, shuffle=False, batch_size=1)
    return (val_batches.classes, batches.classes, onehot(val_batches.classes), onehot(batches.classes),
        val_batches.filenames, batches.filenames, test_batches.filenames)

from keras import backend as K
def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)

def plot(img):
    plt.imshow(to_plot(img))
