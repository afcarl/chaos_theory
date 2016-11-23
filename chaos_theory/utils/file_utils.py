import os
import errno

UTILS_DIR = os.path.dirname(os.path.realpath(__file__))
CODE_DIR = os.path.dirname(UTILS_DIR)
ROOT_DIR = os.path.dirname(CODE_DIR)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise