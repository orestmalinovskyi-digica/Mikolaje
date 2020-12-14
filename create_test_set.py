from os import walk
import os
from sklearn.model_selection import train_test_split
from shutil import copy2


def ret_structure(mypath):
    # SOURCE: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    return f


def check_create(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def make_sets(class_name, class_dir, test_split=.1):
    # Check if all needed directories exist and create them if they don't
    test_dir = "images/test"
    train_dir = "images/train"
    check_create(test_dir)
    check_create(train_dir)
    test_dir = os.path.join(test_dir, class_name)
    train_dir = os.path.join(train_dir, class_name)
    check_create(test_dir)
    check_create(train_dir)

    # make a list of files for the class and split it
    f_set = ret_structure(class_dir)
    train, test = train_test_split(f_set, test_size=test_split)

    # Print to Console what and how much was splitted
    print(f'{class_name}: Created {len(train)} instances for training set '
          f'and {len(test)} instances for test set')

    # copy the files to corresponding directories
    for piece in test:
        copy2(src=os.path.join(class_dir, piece), dst=test_dir)
    for piece in train:
        copy2(src=os.path.join(class_dir, piece), dst=train_dir)


make_sets("santa", "images/raw/santas")
make_sets("person", "images/raw/peops")
