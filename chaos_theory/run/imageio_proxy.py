import numpy as np

def mimwrite(fname, image_list, **kwargs):
    if fname.endswith('.gif'):
        fname = fname[:-4] + '.npz'
    with open(fname, 'wb') as f:
        np.savez(f, image_list)