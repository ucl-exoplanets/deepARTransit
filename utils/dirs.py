import os
import shutil

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
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
    """
    dirs - a list of directories to delete if these directories are found
    :param dirs:
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