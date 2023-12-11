"""
Tomographic reconstruction and projection functions.
@author: Joel Yeo, joelyeo@u.nus.edu
"""

import numpy as np
import quaternion
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from . import interpolate

fft2 = lambda array: torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(array)))
ifft2 = lambda array: torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(array)))
fftn = lambda array: torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(array)))
ifftn = lambda array: torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(array)))


def fourier_reconstruction(
    projections,
    quaternions,
    mask=False,
    is_fourier=False,
    return_fourier=False,
    return_real=True,
    return_weights=False,
):
    """
    Uses legacy C code to perform 3D reconstruction from 2D projections via the
    Fourier slice theorem.

    Parameters
    ----------
    projections : 3D ndarray
        The 2D projections, with shape (no. projections, len_y, len_x).
    quaternions : 2D tensor
        Array of quaternions associated with the intensities list.
        Structured as (no. measurements, 4), where the columns are the
        quaternion components.
    mask : bool, optional
        Applies a mask in the Fourier domain. Default: False.
    is_fourier : bool, optional
        Set to true if input projections are already Fourier transformed. Default: False.
    return_fourier : bool, optional
        Set to true to return 3D volume in Fourier domain. Default: False.
    return_real : bool, optional
        Set to true to return the real part of 3D volume in real space. Default: False.
    return_weights : bool, optional
        Set to true if user wants to implement batchwise Fourier reconstruction methods.
        Default: False.

    Returns
    -------
    volume : 3D ndarray
        Note that the volume is transposed to match RELION's axes convention of
        (z,y,x). The c code returns a volume in (x,y,z) instead.

    """
    quat = quaternion.as_quat_array(quaternions)
    nxy = max(projections.shape[1:])
    x = np.arange(nxy) - (nxy) // 2
    xx, yy, zz = np.meshgrid(x, x, x, indexing="xy")
    if mask:
        Fmask = (xx**2 + yy**2 + zz**2) < (((nxy) // 2) ** 2)
    else:
        Fmask = np.ones((nxy, nxy, nxy))
    coords = quaternion.rotate_vectors(
        quat,
        np.vstack(
            [
                xx[:, :, nxy // 2].flatten(),
                yy[:, :, nxy // 2].flatten(),
                zz[:, :, nxy // 2].flatten(),
            ]
        ).T,
    )
    if is_fourier:
        Fimages = projections
    else:
        Fimages = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(projections, axes=(1, 2))), axes=(1, 2)
        )
    Fvol_real, weights_real = interpolate.images_into_volume(
        Fimages.real, coords, return_weights=True
    )
    Fvol_imag, weights_imag = interpolate.images_into_volume(
        Fimages.imag, coords, return_weights=True
    )
    if return_weights:
        Fvol_real = Fvol_real * Fmask
        Fvol_imag = Fvol_imag * Fmask
        return [Fvol_real.T, weights_real.T], [Fvol_imag.T, weights_imag.T]
    else:
        Fvolume = (Fvol_real * Fmask) + 1j * (Fvol_imag * Fmask)
        if return_fourier:
            volume = Fvolume
        elif return_real:
            volume = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(Fvolume))).real
        else:
            volume = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(Fvolume)))
        return volume.T


def quaternion_to_rotation_matrices(quaternions):
    """
    Converts quaternions to rotation matrices.

    Parameters
    ----------
    quaternions : 2D tensor
        Array of quaternions associated with the intensities list.
        Structured as (no. measurements, 4), where the columns are the
        quaternion components.

    Returns
    -------
    rm : 3D ndarray
        Rotation matrices of shape (no. measurements, 3, 3).

    """

    # Roll to match scipy's convention of (x,y,z,w)
    if len(quaternions.shape) == 1:
        R = Rotation.from_quat(np.roll(quaternions, -1))
    elif len(quaternions.shape) == 2:
        R = Rotation.from_quat(np.roll(quaternions, -1, axis=1))
    return R.as_matrix()


def quaternion_to_affine_matrices(quats):
    n = len(quats)
    rm = torch.from_numpy(quaternion_to_rotation_matrices(quats))
    rm = torch.cat([rm, torch.zeros([n, 3, 1])], dim=2)
    return rm


def slices_from_volume(vol, quats, batchsize=10):
    """
    Extracts 2D slices rotated by the specified quaternion from a 3D volume.

    Parameters
    ----------
    vol : 3D tensor
        Volume to extract 2D slices from.
    quats : 2D tensor
        Array of quaternions associated with the intensities list.
        Structured as (no. measurements, 4), where the columns are the
        quaternion components.
    batchsize : int, optional
        Interpolation of 2D slices is done in iterations. Batchsize sets the number
        of slices to be interpolated per iteration. Set to a lower number to reduce
        memory load on your device. Default: 10.

    Returns
    -------
    projs : 3D tensor
        The 2D slices with shape (no. measurements, len_y, len_x).

    """
    nz = 1
    ny = vol.shape[1]
    nx = vol.shape[2]
    cz = int(nz / 2)
    cy = int(ny / 2)
    cx = int(nx / 2)
    n = len(quats)
    rms = quaternion_to_affine_matrices(quats)
    device = vol.device
    rms = rms.to(device)
    tiled_vol = torch.tile(vol[None, None, ...], (batchsize, 1, 1, 1, 1))

    projs = []
    for i in range(0, n, batchsize):
        rm = rms[i : i + batchsize]
        # create null grid
        null_rot = torch.zeros_like(rm)
        null_rot[:, 0, 0] = 1.0
        null_rot[:, 1, 1] = 1.0
        null_rot[:, 2, 2] = 1.0
        grid = F.affine_grid(null_rot, (batchsize, 1, nz, ny, nx), align_corners=False)
        # ensures z-coordinates we want to sample are at nx(or ny)//2
        grid[..., 2] = grid[0, 0, cx, cy, 1]

        # rotate the grid around the center defined by RELION
        center = grid[:, cz, cy, cx]  # (B, 3)
        center = center.unsqueeze(1)
        grid = grid.view(batchsize, nz * ny * nx, 3)
        grid = (grid - center).bmm(rm[..., :-1].transpose(1, 2)) + center
        grid = grid.view(batchsize, nz, ny, nx, 3)

        #
        # grid[...,2] = torch.squeeze(center)[0]

        # sample
        sampled = F.grid_sample(
            tiled_vol, grid, padding_mode="border", align_corners=False, mode="bilinear"
        )
        projs.append(torch.squeeze(sampled))
    projs = torch.concat(projs, dim=0)
    return projs


def projection(vol, quats, batchsize=10, mask=False):
    """
    Computes the 2D projection from a 3D volume for the quaternions specified using
    the Fourier slice theorem.

    Parameters
    ----------
    vol : 3D tensor
        Volume to extract 2D slices from.
    quats : 2D tensor
        Array of quaternions associated with the intensities list.
        Structured as (no. measurements, 4), where the columns are the
        quaternion components.
    batchsize : int, optional
        Interpolation of 2D slices is done in iterations. Batchsize sets the number
        of slices to be interpolated per iteration. Set to a lower number to reduce
        memory load on your device. Default: 10.
    mask : bool, optional
        Applies a mask in the Fourier domain. Default: False.

    Returns
    -------
    slices : 3D tensor
        The 2D projections with shape (no. measurements, len_y, len_x).

    """
    if mask:
        nxy = max(vol.shape)
        x = torch.arange(nxy) - (nxy) // 2
        xx, yy, zz = torch.meshgrid(x, x, x, indexing="xy")
        Fmask = (xx**2 + yy**2 + zz**2) < ((nxy) // 2) ** 2
    else:
        Fmask = 1
    # origin
    fvol = fftn(vol) * Fmask

    fslices_r = slices_from_volume(fvol.real, quats, batchsize=batchsize)
    fslices_i = slices_from_volume(fvol.imag, quats, batchsize=batchsize)
    fslices = fslices_r + 1j * fslices_i
    slices = ifft2(fslices)
    return slices