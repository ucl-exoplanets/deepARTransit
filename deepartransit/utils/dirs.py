"""Module for directory handling.

create_dirs function adapted from https://github.com/MrGemy95/Tensorflow-Project-Template/blob/master/utils/dirs.py (Apache 2.0 license)
"""
import os
import shutil


def create_dirs(dirs):
    """Create dirs listed in input.

    :param dirs: a list of directories to create if these directories are not found
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.isdir(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def delete_dirs(dirs):
    """ Delete dirs contained input list.

    :param dirs: list of directories to delete if these directories are found:
    :return exit_code: 0:success  / -1:failed
    """
    try:
        for dir_ in dirs:
            if os.path.isdir(dir_):
                shutil.rmtree(dir_, ignore_errors=True)
        return 0
    except Exception as err:
        print("Deleting directories error: {0}".format(err))
        exit(-1)
