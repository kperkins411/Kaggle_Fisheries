NUMBER_OF_SAMPLES = 10

TRAIN_PERCENT = 0.8
VALIDATE_PERCENT = 0.2

# As large as you can, but no larger than 64 is recommended.
# If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
BATCH_SIZE = 4
TEST_BATCH_SIZE = 128
NUMBER_EPOCHS = 10

# where weights are saved after training
CHECKPOINTFILE_1 = "vgg.hdf5"
CHECKPOINTFILE_2 = "vggBN.h5"
FC_TOP_LAYER_BEST_WEIGHTS = "FC_TOP_LAYER_BEST_WEIGHTS.h5"
FC_TOP_LAYER_BEST_WEIGHTS_WITH_IMAGE_SIZE_INPUTS = "FC_TOP_LAYER_BEST_WEIGHTS_WITH_IMAGE_SIZE_INPUTS.h5"

#where visualizations of models go
CONV_LAYERS_MODEL_PNG = "CONV_LAYERS_MODEL_PNG.png"
FC_TOP_LAYER_MODEL_PNG = "FC_TOP_LAYER_MODEL_visual.png"
CONV_LAYERS_AND_FC_TOP_LAYER_MODEL_PNG = "CONV_LAYERS_AND_FC_TOP_LAYER_MODEL_PNG.png"
CONV_LAYERS_AND_FC_TOP_LAYER_MODEL_PNG_1 = "CONV_LAYERS_AND_FC_TOP_LAYER_MODEL_PNG_1.png"

#classes (as annotated by subdirs in train)
NUM_CLASSES = 8
classes=["ALB", "BET","DOL","LAG","NoF", "OTHER","SHARK","YFT"]

# all data here
DATA_FOLDER_NAME = "data"

TRAIN_ALL_FOLDER_NAME = "train_all"
TRAIN_SAMPLES_FOLDER_NAME="train_sample"

TRAIN_FOLDER_NAME = "train"
VALIDATE_FOLDER_NAME = "valid"
TEST_FOLDER_NAME = "test"
PRE_PROCESSED_IMAGES = "PRE_PROCESSED_IMAGES"
RESULTS  = "RESULTS"

CONV_TRN_FEAT = "conv_trn_feat.dat"
CONV_VAL_FEAT = "conv_val_feat.dat"
CONV_TST_FEAT = "conv_tst_feat.dat"

#the datasets
DATA_ALL = "data/train_all/"
DATA_SAMPLE = "data/train_sample/"

# *********** which dataset to work on *************
DATA = DATA_ALL










