#!/usr/bin/env python

# USAGE
# python parse_train_data.py --train train.zip --test test.zip

#splits the train data into 90% train and 10% validation
#the test set is for testing model, results submitted to kaggle
'''
#dir structure
data

        train_all
            test
            PRE_PROCESSED
            kaggle
               #uncategorized kaggle test data
            train
                ALB
                BET
                DOL
                LAG
                NoF
                OTHER
                SHARK
                YFT
            validate
                ALB
                BET
                DOL
                LAG
                NoF
                OTHER
                SHARK
                YFT
        train_sample
            PRE_PROCESSED
            test
                kaggle
                   #uncategorized kaggle test data
            train
                ALB
                BET
                DOL
                LAG
                NoF
                OTHER
                SHARK
                YFT
            validate
                ALB
                BET
                DOL
                LAG
                NoF
                OTHER
                SHARK
                YFT
     train.zip # zip of all data
     test.zip  #zip of kaggle data
'''
import argparse
import os
import utilsKP
import settings as set
import logging

logging.basicConfig(
    #filename = 'parse_data.log', #comment this line out if you want data in the console
    format = "%(levelname) -10s %(asctime)s %(module)s:%(lineno)s %(funcName)s %(message)s",
    level = logging.DEBUG
)
utilities = utilsKP.utils()

#used to create proper directory structure
def makeDirs(*dirs):
    if not dirs:
        logging.debug("called makedirs with no dirs to make!")
    for dir in dirs:
        utilities.makeDir(dir)

def makesubdirs(cats, folder):
    #make subdirs
    utilities.makeDir(folder)
    # now lets make each of the categorical subfolders
    for cat in cats:
        tmp = os.path.join(folder, cat)
        utilities.makeDir(tmp)

import shutil


def generate_splits(srcdir, destdir, cats, *, percent=None, absolute_number=None, movefile=True):
    '''
     for each cat (a directory in srcdir), copies or moves files from srcdir to destdir according to keyword only args

     :param srcdir: where files pulled from
     :param datadir: where files go
     :param cats: list of categories (presumably the dir structure in srcdir)
     :param percent: copy percentage of srcdir files?
     :param absolute_number: copy absolute_number of srcdir files?
     :param movefile: move or copy?
     :return:
     '''

    for cat in cats:
        src_catdir = os.path.join(srcdir,cat)
        dst_catdir = os.path.join(destdir,cat)
        copy_or_move(src_catdir, dst_catdir,percent=percent, absolute_number=absolute_number, movefile=movefile)


def copy_or_move(srcdir,destdir,*,percent=None,absolute_number=None, movefile = True, appendThisCategoryNameToCopyOrMovedFile = None):
    '''
     either moves or copies files from srcdir, to destdir
     creates a list of dirs in dest dir from cats
     and copies or moves files from srcdir to destdir according to keyword only args

     :param srcdir: where files pulled from
     :param datadir: where files go
     :param cats: list of categories (presumably the dir structure in srcdir)
     :param percent: copy percentage of srcdir files?
     :param absolute_number: copy absolute_number of srcdir files?
     :param movefile: move or copy?
     :param appendThisCategoryNameToCopyOrMovedFile: the test dir is a single dir full of data, prepend category
                                    to name to annotate what correct label should be
     :return:
     '''

    #how many files do we have?
    files = os.listdir(srcdir)
    numbfiles = len(files)

    #how many go to destdir/cat?
    numb_to_move = absolute_number if absolute_number is not None else int(numbfiles*percent)

    #do we copy or move files?
    fun = os.rename if movefile == True else shutil.copy2

    for file in files:
        numb_to_move -= 1
        if (numb_to_move) < 0:
            break
        dstfile = (appendThisCategoryNameToCopyOrMovedFile + file) if appendThisCategoryNameToCopyOrMovedFile is not None else file
        fun(os.path.join(srcdir, file), os.path.join(destdir, dstfile))

def main():
    # construct the argument parser
    parser = argparse.ArgumentParser(description='prepare training data for fisheries competition')
    parser.add_argument("-train", "--train", default="train.zip", help="train zip file, default = train.zip")
    parser.add_argument("-test", "--test", default="test_stg1.zip", help="test zip file, default = test_stg1.zip")
    args = vars(parser.parse_args())

    # where is data going
    datadir = os.path.join(os.getcwd(),set.DATA_FOLDER_NAME)

    #FQ zip files for all train data and all test data
    zipfile_all_train = os.path.join(datadir, args["train"])
    zipfile_all_test = os.path.join(datadir, args["test"])

    #main folders for all train/val data and sample train/val data
    train_all_folder = os.path.join(datadir, set.TRAIN_ALL_FOLDER_NAME)
    sample_all_folder = os.path.join(datadir, set.TRAIN_SAMPLES_FOLDER_NAME)

    #if not there then create it
    utilities.makeDir(train_all_folder)
    utilities.makeDir(sample_all_folder)

    # unzip
    utilities.unzip_to_dir(zipfile_all_train, train_all_folder)

    #go to train data
    traindir = os.path.join(train_all_folder,set.TRAIN_FOLDER_NAME)

    #will traverse path and get a list of categories (directories)
    cats = [cat for cat in os.listdir(traindir) if os.path.isdir(os.path.join(traindir,cat))]

    #-------------
    # make the main validation folder
    folder = os.path.join(train_all_folder, set.VALIDATE_FOLDER_NAME)
    makesubdirs(cats, folder)

    #make the test folder
    testfolder = os.path.join(train_all_folder, set.TEST_FOLDER_NAME)
    utilities.makeDir(folder)
    print("unzip the test data to " + folder)
    utilities.unzip_to_dir(zipfile_all_test, testfolder)

    # make the preprocessd image folder (used in later programs for bcolz
    # to dump preprocessed image files to)
    preprocessedfolder = os.path.join(train_all_folder, set.PRE_PROCESSED_IMAGES)
    utilities.makeDir(preprocessedfolder)

    # and the results folder
    resultsfolder = os.path.join(train_all_folder, set.RESULTS)
    utilities.makeDir(resultsfolder)

    # -------------
    # make the sample train folders
    folder = os.path.join(sample_all_folder, set.TRAIN_FOLDER_NAME)
    makesubdirs(cats, folder)

    # make the sample val folders
    folder = os.path.join(sample_all_folder, set.VALIDATE_FOLDER_NAME)
    makesubdirs(cats, folder)

    # make the test folder
    testfolder = os.path.join(sample_all_folder, set.TEST_FOLDER_NAME)
    utilities.makeDir(folder)
    print("unzip the test data to " + folder)
    utilities.unzip_to_dir(zipfile_all_test, testfolder)

    # make the preprocessd image folder (used in later programs for bcolz
    # to dump preprocessed image files to)
    preprocessedfolder = os.path.join(sample_all_folder, set.PRE_PROCESSED_IMAGES)
    utilities.makeDir(preprocessedfolder)

    # and the results folder
    resultsfolder = os.path.join(sample_all_folder, set.RESULTS)
    utilities.makeDir(resultsfolder)

    # -------------
    #now lets move a subset from training to validate and test
    srcdir = os.path.join(train_all_folder,set.TRAIN_FOLDER_NAME)
    valdir = os.path.join(train_all_folder,set.VALIDATE_FOLDER_NAME)
    generate_splits(srcdir, valdir, cats,percent = set.VALIDATE_PERCENT)

    #copy a subset from training to sample (for fast verification)
    srcdir = os.path.join(train_all_folder,set.TRAIN_FOLDER_NAME)
    trndir = os.path.join(sample_all_folder,set.TRAIN_FOLDER_NAME)
    generate_splits(srcdir, trndir, cats,absolute_number = set.NUMBER_OF_SAMPLES, movefile = False)

    #move a subset from sample training to validate and test (for fast verification)
    valdir = os.path.join(sample_all_folder,set.VALIDATE_FOLDER_NAME)
    tstdir = os.path.join(sample_all_folder,set.TEST_FOLDER_NAME)
    generate_splits(trndir, valdir, cats,percent = set.VALIDATE_PERCENT)

if __name__ == '__main__':
    main()

