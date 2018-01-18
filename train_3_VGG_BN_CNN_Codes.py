import os
import settings
import utils_data as ut
from keras.layers.convolutional import Convolution2D,Conv2D
from keras.models import Sequential
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
#from keras.regularizers import l2,  l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from keras import optimizers


# taken care of in ~/.keras/keras.json
# from keras import backend
# backend.set_image_dim_ordering('tf')

def path_setup():
    # which dataset to work on?large or the small
    path = settings.DATA

    # where are we?
    datadir = os.path.join(os.getcwd(), settings.DATA_FOLDER_NAME)
    return path,datadir

# 

def load_CNN_Codes(model,data, path, fname):
    '''
    call demo load_CNN_Codes(model_no_head,settings.TRAIN_FOLDER_NAME,path, settings.CONV_TRN_FEAT)
    call this function 
    
    :param model: 
    :param data: 
    :param path: 
    :param fname: 
    :return: 
    '''
    try:
        # load if it exists
        results_file = os.path.join(os.path.join(path,settings.RESULTS),fname)
        return ut.bcolz.open(results_file)[:]
    except FileNotFoundError:
        # uh oh, does not exists, so make predictions based on that data (time consuming)
        features = model.predict(data)

        # and save it for future load (big speedup)
        ut.save_array(results_file,features)
        return features

def get_fc_layers(p,conv_layers):
    '''
    :param p: dropout
    :param conv_layers: used here strictly to find the input size for the first fully connected layer
    :return: a fully connected topper for the conv net defined in conv_layers (but NOT appended to it)
    '''
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        BatchNormalization(axis=1),
        Dropout(p/4),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        Dense(8, activation='softmax')
    ]
def display_summary(info,model):
    print("***************")
    print(info)
    model.summary()
    print("***************")
    pass


def main():

    path,datadir = path_setup()

    # create model with correct number of outputs
    # this pops off the existing last layer, sets remaining layers to not trainable
    # and puts on a layer with the correct number of outputs
    from vgg16bn import Vgg16BN
    vgg = Vgg16BN()
    vgg.ft(settings.NUM_CLASSES)
    model = vgg.model

    # now load wth the best weights (created by train_2_VGG_BN.py
    file_best_weights = os.path.join(path,settings.CHECKPOINTFILE_2)
    print("loading bestweights from "+  file_best_weights)
    try:
        model.load_weights(file_best_weights)
    except OSError:
        print('WHOAH!!!! Run train_2_VGG_BN.py in order to create '+ settings.CHECKPOINTFILE_2)
        return

    #load training data
    trn = ut.load_array(path,settings.TRAIN_FOLDER_NAME)
    val = ut.load_array(path,settings.VALIDATE_FOLDER_NAME)
    tst = ut.load_array(path,settings.TEST_FOLDER_NAME)

    # see https://stackoverflow.com/questions/41771965/error-when-checking-model-input-expected-convolution2d-input-1-to-have-shape-n
    trn = trn.transpose(0, 3, 1, 2)
    val = val.transpose(0, 3, 1, 2)
    tst = tst.transpose(0, 3, 1, 2)

    (val_classes, trn_classes, val_labels, trn_labels, val_filenames, trn_filenames, tst_filenames) = ut.get_classes(path)

    # lets strip off the FC layers and just calculate the stacked conv layer outputs for the network
    # then create a fc network of appropriate input and output size) and use above conv outputs
    # to train it, much faster since most of the back prop compute is in the conv layers.
    # this means the only training is happenning in the FC layers
    conv_layers, fc_layers = ut.split_at(model, Conv2D)
    conv_model = Sequential(conv_layers)

    #whats it look like
    plot_model(conv_model, to_file=settings.CONV_LAYERS_MODEL_PNG, show_shapes=True)
    # display_summary("CONV model size is", conv_model)

    # get the predictions
    conv_trn_feat = load_CNN_Codes(conv_model,trn,path, settings.CONV_TRN_FEAT)
    conv_val_feat = load_CNN_Codes(conv_model, val, path, settings.CONV_VAL_FEAT)
    conv_tst_feat = load_CNN_Codes(conv_model, tst, path, settings.CONV_TST_FEAT)

    print ("Feature shape:" + str(conv_val_feat.shape))

    #********************************************************************************
    # the follwoing bits create and train a fully connected top layer using CNN_Codes
    fc_top_layer_model = create_and_train_FC_top_layer_using_CNN_Codes(conv_layers, conv_trn_feat, conv_val_feat, path,
                                                                       trn_labels, val_labels)

    #********************************************************************************
    #tape the fully connected top layer on to the conv base layer and train the whole thingt once more (conv layers are not trainable)
    tape_FC_Top_layer_onto_conv_base_and_train(conv_model, fc_top_layer_model, trn, trn_labels, val, val_labels)

    ###########################################
    #there are thousands of images but only a few image sizes(<10), assumme each image size corresponds to a particular
    # boat and that each boat is going for a particular type of fish, then can we assumme a higher percentage of pix for
    # that kind of fish on that particular boat? (Probably, but the neural net has already taken this into account, so
    # no accuracy boost)
    create_new_model_that_includes_size_of_image_as_input(conv_trn_feat, conv_val_feat, model, path, trn_filenames,
                                                          trn_labels, val_filenames, val_labels)


def create_new_model_that_includes_size_of_image_as_input(conv_trn_feat, conv_val_feat, model, path, trn_filenames,
                                                          trn_labels, val_filenames, val_labels):
    from get_normalized_one_hot_file_sizes import get_one_hot_sizes_for_files
    trn_normalized_one_hot_file_sizes = get_one_hot_sizes_for_files(os.path.join(path, settings.TRAIN_FOLDER_NAME),
                                                                    trn_filenames)
    # number of 1 hot encoded values, used to ensure one hot encoding consisytant accross trn, val, tst
    numb_classes = len(trn_normalized_one_hot_file_sizes[0])
    sz_one_hot = Input((numb_classes,))
    print("numb_classes=", numb_classes)
    val_normalized_one_hot_file_sizes = get_one_hot_sizes_for_files(os.path.join(path, settings.VALIDATE_FOLDER_NAME),
                                                                    val_filenames, numb_classes)
    # tst_normalized_one_hot_file_sizes = get_one_hot_sizes_for_files(os.path.join(path,settings.TEST_FOLDER_NAME), tst_filenames,numb_classes)
    # dropout
    p = 0.6
    # build a top with extra inputs for the one hot encodings
    conv_layers, _ = ut.split_at(model, Conv2D)
    # conv_model = Sequential(conv_layers)
    # need an input consisting of both an image and a set of one hot encoded values
    # size of each image
    inp = Input(conv_layers[-1].output_shape[1:])
    # batch normalize these one hots
    bn_one_hots = BatchNormalization()(sz_one_hot)
    # create the FC net
    x = MaxPooling2D()(inp)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(p / 4)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(p)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(p / 2)(x)
    x = merge([x, bn_one_hots], 'concat')
    x = Dense(8, activation='softmax')(x)
    # When we compile the model, we have to specify all the input layers in an array.
    model = Model([inp, sz_one_hot], x)
    # load the best weights
    load_FC_best_weights(model,
                         path, settings.FC_TOP_LAYER_BEST_WEIGHTS_WITH_IMAGE_SIZE_INPUTS)
    model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    display_summary("model with extra inputs", model)
    print("size of trn_normalized_one_hot_file_sizes=" + str(trn_normalized_one_hot_file_sizes.size))
    model.fit([conv_trn_feat, trn_normalized_one_hot_file_sizes], trn_labels, batch_size=settings.BATCH_SIZE,
              nb_epoch=settings.NUMBER_EPOCHS,
              validation_data=([conv_val_feat, val_normalized_one_hot_file_sizes], val_labels))
    # save the best weights
    file_best_weights = os.path.join(path, settings.FC_TOP_LAYER_BEST_WEIGHTS_WITH_IMAGE_SIZE_INPUTS)
    print("saving bestweights to " + file_best_weights)
    model.save_weights(file_best_weights)


def tape_FC_Top_layer_onto_conv_base_and_train(conv_model, fc_top_layer_model, trn, trn_labels, val, val_labels):
    conv_model.add(fc_top_layer_model)
    # whats it look like
    plot_model(conv_model, to_file=settings.CONV_LAYERS_AND_FC_TOP_LAYER_MODEL_PNG, show_shapes=True)
    # display_summary("Conv+ FC, after adding 2 layers the size of the model is", conv_model)
    # try compiling so changes take effect
    conv_model.compile(loss='binary_crossentropy',
                       optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                       metrics=['accuracy'])
    # lets see how well the whole thing performs
    conv_model.fit(trn, trn_labels, batch_size=settings.BATCH_SIZE, nb_epoch=settings.NUMBER_EPOCHS,
                   validation_data=(val, val_labels))


def create_and_train_FC_top_layer_using_CNN_Codes(conv_layers, conv_trn_feat, conv_val_feat, path, trn_labels,
                                                  val_labels):
    # how much dropout
    p = 0.6
    # create fully connected topper
    fc_top_layer_model = Sequential(get_fc_layers(p, conv_layers))
    load_FC_best_weights(fc_top_layer_model,
                         path,settings.FC_TOP_LAYER_BEST_WEIGHTS )  # lets save what this model looks like (image saved in following file)
    plot_model(fc_top_layer_model, to_file=settings.FC_TOP_LAYER_MODEL_PNG, show_shapes=True)
    # display_summary("FC model size is", fc_top_layer_model)
    fc_top_layer_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # now train it
    fc_top_layer_model.fit(conv_trn_feat, trn_labels, batch_size=settings.BATCH_SIZE, nb_epoch=settings.NUMBER_EPOCHS,
                           validation_data=(conv_val_feat, val_labels))
    # reduce the learning rate a bit and train again
    fc_top_layer_model.optimizer.lr = 1e-4
    fc_top_layer_model.fit(conv_trn_feat, trn_labels, batch_size=settings.BATCH_SIZE, nb_epoch=settings.NUMBER_EPOCHS,
                           validation_data=(conv_val_feat, val_labels))
    file_best_weights = os.path.join(path, settings.FC_TOP_LAYER_BEST_WEIGHTS)
    print("saving bestweights to " + file_best_weights)
    fc_top_layer_model.save_weights(file_best_weights)
    stl = fc_top_layer_model.evaluate(conv_val_feat, val_labels)
    print("For Val , Loss:" + str(stl[0]) + " Accuracy:" + str(stl[1]))
    return fc_top_layer_model


def load_FC_best_weights(fc_top_layer_model, path, fn):
    # lets load the saved best weights (if any) to the model
    file_best_weights = os.path.join(path,fn )
    print("loading bestweights from " + file_best_weights)
    try:
        fc_top_layer_model.load_weights(file_best_weights)
    except OSError:
        print('WHOAH!!!! First time to run, no best weights yet in file  ' + fn)


if __name__ == '__main__':
    main()