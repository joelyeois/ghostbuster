# from ctypes import *
# ctools = CDLL('./ctools.so')

from .ctools import interp3d, slice3d

def images_into_volume(images, coords, return_weights=False):
    size = max(images.shape[1:])
    v,w = interp3d(coords[:,:,0].flatten(), coords[:,:,1].flatten(), coords[:,:,2].flatten(), images.flatten(), size)
    v[w>0] /= w[w>0]
    if return_weights:
        return v,w
    else:
        return v

def volume_into_images(volume, coords, return_weights=False):
    size = max(volume.shape)
    s,w = slice3d(coords[:,:,0].flatten(), coords[:,:,1].flatten(), coords[:,:,2].flatten(), volume, size)
    s[w>0] /= w[w>0]
    if return_weights:
        return s,w
    else:
        return s
