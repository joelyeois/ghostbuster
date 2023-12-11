"""
Legacy C code for tomographic reconstruction.
@author: Joel Yeo, joelyeo@u.nus.edu
"""

from .ctools import interp3d


def images_into_volume(images, coords, return_weights=False):
    """
    Interpolates a collection of differently rotated 2D slices into a 3D volume. For
    tomographic reconstruction.

    Parameters
    ----------
    images : 3D ndarray
        The 2D slices, with shape (no. slices, len_y, len_x).
    coords : 3D ndarray
        The x,y,z coordinates for each pixel in the slices. Has shape
        (no. slices, no. pixels per slice, 3).
    return_weights : bool, optional
        If True, returns both interpolation weights and volume. Else, only returns
        volume. Useful for performing batchwise Fourier reconstruction. Default: False.

    Returns
    -------
    v : 3D ndarray
        The interpolated volume.
    w : 3D ndarray, optional
        The weights associated with the volume.

    """
    size = max(images.shape[1:])
    v, w = interp3d(
        coords[:, :, 0].flatten(),
        coords[:, :, 1].flatten(),
        coords[:, :, 2].flatten(),
        images.flatten(),
        size,
    )
    v[w > 0] /= w[w > 0]
    if return_weights:
        return v, w
    else:
        return v