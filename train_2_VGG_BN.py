import os
import settings
import utils_data as ut

# taken care of in ~/.keras/keras.json
# from keras import backend
# backend.set_image_dim_ordering('tf')

def path_setup():
    # which dataset to work on?large or the small
    path = settings.DATA
    # where are we?
    datadir = os.path.join(os.getcwd(), settings.DATA_FOLDER_NAME)
    return path,datadir


def main():

    path,datadir = path_setup()

    # create model with correct number of outputs
    # this pops off the existing last layer, sets remaining layers to not trainable
    # and puts on a layer with the correct number of outputs
    from vgg16bn import Vgg16BN
    vgg = Vgg16BN()
    vgg.ft(settings.NUM_CLASSES)
    model = vgg.model

    #load training data
    trn = ut.load_array(path,settings.TRAIN_FOLDER_NAME)
    val = ut.load_array(path,settings.VALIDATE_FOLDER_NAME)
    tst = ut.load_array(path,settings.TEST_FOLDER_NAME)

    # gen = image.ImageDataGenerator()

    from keras import optimizers
    model.compile(optimizer=optimizers.Adam(1e-3),loss = 'categorical_crossentropy',metrics = ['accuracy'])

    (val_classes, trn_classes, val_labels, trn_labels, val_filenames, filenames, test_filenames) = ut.get_classes(path)

    # see https://stackoverflow.com/questions/41771965/error-when-checking-model-input-expected-convolution2d-input-1-to-have-shape-n
    trn = trn.transpose(0, 3, 1, 2)
    val = val.transpose(0, 3, 1, 2)
    # tst = tst.transpose(0, 3, 1, 2)

    model.fit(trn,trn_labels, batch_size=settings.BATCH_SIZE, nb_epoch=settings.NUMBER_EPOCHS, validation_data=(val,val_labels))

    file_best_weights = os.path.join(path,settings.CHECKPOINTFILE_2)
    print("saving bestweights to "+  file_best_weights)
    model.save_weights(file_best_weights)


if __name__ == '__main__':
    main()

