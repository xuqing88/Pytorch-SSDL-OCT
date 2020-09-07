import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


import os
import shutil
import random
random.seed(0)
from shutil import copyfile


def img_train_test_split(img_source_dir, train_size):
    """
    Randomly splits images over a train and validation folder, while preserving the folder structure

    Parameters
    ----------
    img_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path

    train_size : float
        Proportion of the original images that need to be copied in the subdirectory in the train folder
    """
    if not (isinstance(img_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(img_source_dir):
        raise OSError('img_source_dir does not exist')

    if not os.path.exists('./dataset/OCT2017/train_semi'):
        os.makedirs('./dataset/OCT2017/train_semi')
    if not os.path.exists('./dataset/OCT2017/unlabel'):
        os.makedirs('./dataset/OCT2017/unlabel')


    # Get the subdirectories in the main image folder
    subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]

    for subdir in subdirs:
        subdir_fullpath = os.path.join(img_source_dir, subdir)
        if len(os.listdir(subdir_fullpath)) == 0:
            print(subdir_fullpath + ' is empty')
            break

        train_subdir = os.path.join('./dataset/OCT2017/train_semi', subdir)
        validation_subdir = os.path.join('./dataset/OCT2017/unlabel', subdir)

        # Create subdirectories in train and validation folders
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)

        if not os.path.exists(validation_subdir):
            os.makedirs(validation_subdir)

        train_counter = 0
        validation_counter = 0

        # Calculate how many files for train_org, val and test
        file_counts = len(os.listdir(subdir_fullpath))
        # train_count = int(train_size * file_counts)
        train_count = train_size
        unlabel_count = int(file_counts - train_count)
        temp_list = list(os.listdir(subdir_fullpath))

        # File Partition
        random.shuffle(temp_list)
        file_train_list = temp_list[0:train_count]

        # Randomly assign an image to train or validation folder
        for filename in os.listdir(subdir_fullpath):
            if filename in file_train_list:
                copyfile(os.path.join(subdir_fullpath, filename),
                         os.path.join(train_subdir, filename))
                train_counter += 1
            else:
                copyfile(os.path.join(subdir_fullpath, filename),
                         os.path.join(validation_subdir, filename))
                validation_counter += 1

        print('Copied ' + str(train_counter) + ' images to ./dataset/OCT2017/train_semi/' + subdir)
        print('Copied ' + str(validation_counter) + ' images to ./dataset/OCT2017/unlabel/' + subdir)

    print("Split End")

# Check the dataset
if not os.path.exists('./dataset/OCT2017'):
    print("CELL dataset not exists")
    exit()

# remove the folder before regenerating the result
if os.path.exists('./dataset/OCT2017/train_semi/'):
    shutil.rmtree('./dataset/OCT2017/train_semi/')

if os.path.exists('./dataset/OCT2017/unlabel/'):
    shutil.rmtree('./dataset/OCT2017/unlabel/')

data_dir = './dataset/OCT2017/train'
train_size = 20  #number of training samples for each class
img_train_test_split(data_dir, train_size)
