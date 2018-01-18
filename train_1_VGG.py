
'''
Once done the weights of the finetuned model (with new FC layer)
are in settings.CHECKPOINTFILE.
'''
import os
import settings
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


def main():
    #taken care of in ~/.keras/keras.json
    # from keras import backend
    # backend.set_image_dim_ordering('tf')

    #which dataset to work on?large or the small
    # path = settings.DATA_ALL
    path = settings.DATA_SAMPLE

    # where are we?
    datadir = os.path.join(os.getcwd(), settings.DATA_FOLDER_NAME)

    # create model with correct number of outputs
    # this pops off the existing last layer, sets remaining layers to not trainable
    # and puts on a layer with the correct number of outputs
    # Import our class, and instantiate
    from vgg16 import Vgg16
    vgg = Vgg16()

    # Grab a few images at a time for training and validation.
    # NB: They must be in subdirectories named based on their category
    batches = vgg.get_batches(os.path.join(path,settings.TRAIN_FOLDER_NAME), batch_size=settings.BATCH_SIZE)
    val_batches = vgg.get_batches(os.path.join(path, settings.VALIDATE_FOLDER_NAME), batch_size=settings.BATCH_SIZE * 2)

    # add a different fully connected layer at the end
    # load the best weights from CHECKPOINTFILE_1 (if it exists)
    vgg.finetune(batches, checkpointfile=settings.CHECKPOINTFILE_1)

    # create checkpoint to be used by model.fit to save the best model via callback
    filepath = settings.CHECKPOINTFILE_1
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # fit the model
    vgg.fit(batches, val_batches, nb_epoch=settings.NUMBER_EPOCHS, callbacks=callbacks_list)

if __name__ == '__main__':
    main()