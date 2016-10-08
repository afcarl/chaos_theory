import tempfile
import os
import pickle

def get_wt_string(saver, sess):
    with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
        saver.save(sess, f.name)
        f.seek(0)
        with open(f.name, 'r') as f2:
            wts = f2.read()
        os.remove(f.name + '.meta')
    return wts

def restore_wt_string(saver, sess, wts):
    with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
        f.write(wts)
        f.seek(0)
        saver.restore(sess, f.name)


def dump_pickle_str(obj):
    with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
        pickle.dump(obj, f)
        f.seek(0)
        s = f.read()
    return s


def load_pickle_str(string):
    with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
        f.write(string)
        f.seek(0)
        obj = pickle.load(f)
    return obj
