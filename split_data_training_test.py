#!usr/bin/python
import os
import sys
import random
from shutil import copyfile


"""This script allow to create training, testing and validation directory
    and then fill each one with every class according to split_size parameters """

def split_data(SOURCE, TRAINING, TESTING, VALIDATION, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        if not filename.startswith('.'):
            file = SOURCE + "\\" + filename
            if os.path.getsize(file) > 0:
                files.append(filename)
            else:
                print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int((len(files) - training_length)/2)
    validation_length = testing_length
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[training_length:training_length+testing_length]
    validation_set = shuffled_set[training_length+testing_length:]

    for filename in training_set:
        this_file = SOURCE + "\\" + filename
        destination = TRAINING + "\\" + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + "\\" + filename
        destination = TESTING + "\\" + filename
        copyfile(this_file, destination)

    for filename in validation_set:
        this_file = SOURCE + "\\" + filename
        destination = VALIDATION + "\\" + filename
        copyfile(this_file, destination)


def run_split_train_test(MAIN_DIRECTORY):

    ### CREATE ALL FOLDERS ###
    parent_dir = MAIN_DIRECTORY
    os.chdir(parent_dir)
    folders = os.listdir()

    if 'training' not in str(parent_dir):
        os.mkdir('training')
        print("Create Training folder")
        os.chdir('training')
        for folder in folders:
            os.mkdir(folder)
            
    os.chdir(parent_dir)
        
    if 'testing' not in str(parent_dir):
        os.mkdir('testing')
        print("Create Testing folder")
        os.chdir('testing')
        for folder in folders:
            os.mkdir(folder)

    os.chdir(parent_dir)
        
    if 'validation' not in str(parent_dir):
        os.mkdir('validation')
        print("Create validation folder")
        os.chdir('validation')
        for folder in folders:
            os.mkdir(folder)

    os.chdir(parent_dir)

    SOURCE = os.getcwd()
    TRAINING_DIR = os.getcwd()+'\\training'
    TESTING_DIR = os.getcwd()+'\\testing'
    VALIDATION_DIR = os.getcwd()+'\\validation'

    ### FILLING TRAINING, TESTING AND VALIDATION FOLDER FOR EACH CLASS ###

    for folder in folders:
        if folder == 'training' or folder == 'testing' or folder == 'validation':
            continue
        else:
            split_data(SOURCE+'\\'+folder,TRAINING_DIR+'\\'+folder,TESTING_DIR+'\\'+folder,VALIDATION_DIR+'\\'+folder,0.7)
            print("Done {}".format(folder))
    
    print("Split Done")

if __name__ == '__main__':
    if len(sys.argv)<=1:
        print("ERROR: Need directory argument")
        sys.exit()

    print(len(sys.argv))
    directory_folder =  sys.argv[1]
    run_split_train_test(directory_folder)