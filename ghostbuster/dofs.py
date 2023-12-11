"""
Depth of field stretching (DOFS) algorithm.
@author: Joel Yeo, joelyeo@u.nus.edu
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from . import utils

fft2 = lambda array: torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(array)))
ifft2 = lambda array: torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(array)))


class DOFS:
    def __init__(
        self,
        exitwaves,
        quats,
        wavelength,
        voxelsize=1,
        batchsize=10,
        device="cpu",
        alpha=0.1,
    ):
        """
        Parameters
        ----------
        exitwaves : 3D tensor
            Structure should be (no. measurements, len_y, len_x)
        quats : 2D tensor
            Array of quaternions associated with the exitwaves list.
            Structured as (no. measurements, 4), where the columns are the
            quaternion components in (w,x,y,z).
        wavelength : float
            Wavelength of the electron beam in [Å]
        voxelsize : float, optional
            Voxel size in [Å]. Default: 1.
        batchsize : int, optional
            Number of exitwaves/intensities per iteration. Default: 10.
        device : str, optional
            'cpu' or 'cuda:i', where i is the gpu number. Default: 'cpu'.
        alpha : float, optional
            Amplitude contrast ratio. Default: 0.1.
        """

        self.device = device
        self.exitwaves = exitwaves.to(self.device)
        self.sh = exitwaves.shape[-1]
        self.num_data = len(exitwaves)

        if self.num_data < batchsize:
            self.batchsize = self.num_data
        else:
            self.batchsize = batchsize

        # setup orientations
        self.quats = quats.to(self.device)
        self.matrices = torch.from_numpy(
            self.quaternion_to_rotation_matrices(quats)
        ).to(self.device)

        # setup kspace
        self.voxelsize = voxelsize
        self.k = (utils.create_kspace(self.sh, self.voxelsize)).to(self.device)

        self.wavelength = wavelength
        self.alpha = torch.tensor(alpha)

        # create propagators and transfer functions
        self.create_linear_propagators()

    def quaternion_to_rotation_matrices(self, quaternions):
        """
        Generates rotation matrices from quaternions.

        Parameters
        ----------
        quaternions : 1D or 2D tensor
            Quaternion array is (w,x,y,z) in accordance to quaternion.py convention.
            If 2D, dimensions are (num_angles, 4)

        Returns
        -------
        rm : 3D ndarray
            (num_angles, 3, 3) rotation matrices.

        """
        # Roll to match scipy's convention of (x,y,z,w)
        if len(quaternions.shape) == 1:
            R = Rotation.from_quat(np.roll(quaternions, -1))
        elif len(quaternions.shape) == 2:
            R = Rotation.from_quat(np.roll(quaternions, -1, axis=1))
        return R.as_matrix()

    def create_linear_propagators(self):
        """
        Generates propagators p_m for each mth slice within the volume.
        """
        # assume equally spaced slices. First element corresponds to
        # the first slice located furthest from the exitwave plane
        self.distances = torch.from_numpy(
            np.flip(np.arange(self.sh) + 1) * self.voxelsize
        ).to(self.device)
        self.linear_forward_kernels = torch.zeros(
            (self.sh, self.sh, self.sh), dtype=torch.complex128
        ).to(self.device)
        self.linear_backward_kernels = torch.zeros_like(self.linear_forward_kernels).to(
            self.device
        )
        for i, d in enumerate(self.distances):
            self.linear_forward_kernels[i] = torch.exp(
                1j * torch.pi * self.wavelength * d * (self.k**2)
            )
            self.linear_backward_kernels[i] = torch.exp(
                -1j * torch.pi * self.wavelength * d * (self.k**2)
            )

    def initialize(self, vol=None, groundtruth=None, store_mse=True):
        """
        Initializes DOFS algorithm

        Parameters
        ----------
        vol : 3D tensor, optional
            The initial guess for the 3D particle
        groundtruth : 3D tensor, optional
            The ground truth 3D particle for error metric calculation
        store_mse : bool, optional
            Flag to store mean squared error, calculated between current estimate
            and ground truth.

        """
        self.store_mse = store_mse
        self.iteration_number = 0
        self.currentbatchorder = torch.arange(self.num_data)

        if vol is None:
            self.current = torch.zeros(
                (self.sh, self.sh, self.sh), dtype=torch.complex128, device=self.device
            )
        else:
            self.current = vol.to(self.device)

        if groundtruth is not None:
            self.groundtruth = groundtruth.to(self.device)
            if store_mse:
                self.mses = [self.compute_mse(self.current, self.groundtruth)]

    def iterate(self, niter=1000, stepsize=1, positivity=True, amplitude_contrast=True):
        """
        Performs DOFS iterations.

        Parameters
        ----------
        niter : int, optional
            Number of iterations.
        stepsize : float, optional
            Gradient descent stepsize. Default: 1.
        positivity : bool, optional
            Enforces positivity constraint. Default: True.
        amplitude_contrast : bool, optional
            Enforces amplitude contrast ratio between imaginary and real part of
            slices. Default: True.

        """
        self.stepsize = stepsize
        self.niter = niter
        self.currentbatch = self.currentbatchorder[: self.batchsize]
        self.positivity = positivity
        self.amplitude_contrast = amplitude_contrast

        for i in tqdm(range(niter)):
            self.iteration_number += 1
            self.update(stepsize=stepsize)

            # change batch order
            self.currentbatchorder = torch.roll(self.currentbatchorder, -self.batchsize)
            self.currentbatch = self.currentbatchorder[: self.batchsize]

    def update(self, stepsize=1):
        """
        Performs DOFS update for the current iteration.

        Parameters
        ----------
        stepsize : float, optional
            Gradient descent stepsize. Default: 1.

        """
        # rotation matrices for current batch
        matrices = self.matrices[self.currentbatch]

        # rotate volumes
        rotated_currents = utils.rotate_3d(
            torch.tile(self.current, (self.batchsize, 1, 1, 1)), matrices
        )
        # compute gradient
        current_exitwaves = self.linear_multislice(rotated_currents)

        # calculate residues
        residues = self.exitwaves[self.currentbatch] - current_exitwaves

        # compute gradient
        grads = self.linear_backpropagate_residues(residues)

        # backrotate gradients
        back_rotate_grads = utils.rotate_3d(grads, torch.swapaxes(matrices, 1, 2))

        # average gradients
        averaged_grad = torch.mean(back_rotate_grads, axis=0) / self.sh

        # update current volume
        self.current += stepsize * averaged_grad

        # positivity constraint
        if self.positivity:
            self.current[self.current.real < 0] = 0
            self.current[self.current.imag < 0] = 0

        # amplitude contrast constraint
        if self.amplitude_contrast:
            self.current = self.current.real + 1j * self.current.real / (
                torch.sqrt(1 - self.alpha**2) / self.alpha
            )

        # calculate metrics
        if self.store_mse:
            self.mses.append(self.compute_mse(self.current, self.groundtruth))

    def linear_multislice(self, slices):
        """
        Calculates a batch of 2D exitwaves from 3D volumes using first Born multislice.

        Parameters
        ----------
        slices : 4D tensor
            Has shape of (batchsize, len_z, len_y, len_x).

        Returns
        -------
        exitwave : 3D tensor
            The resultant 2D complex-valued exitwaves with shape (batchsize, len_y, len_x).

        """
        slices_f = fft2(slices)
        p_slices = ifft2(slices_f * self.linear_forward_kernels)
        exitwave = 1 + 1j * p_slices.sum(axis=1)
        return exitwave

    def linear_backpropagate_residues(self, residues):
        """
        Computes a batch of 3D gradients from the error in the 2D exitwaves between
        first Born multislice and 'known' exitwaves.

        Parameters
        ----------
        residues : 3D tensor
            The error with shape of (batchsize, len_y, len_x).

        Returns
        -------
        p_residues : 4D tensor
            The batch of 3D gradients after backpropagation. Has shape of
            (batchsize, len_z, len_y, len_x).

        """
        residues_f = fft2(residues)
        tiled_residues_f = torch.tile(residues_f[:, None, :, :], (1, self.sh, 1, 1))
        p_residues = ifft2(tiled_residues_f * self.linear_backward_kernels)
        return -2j * p_residues

    def compute_mse(self, vol1, vol2):
        """
        Computes the mean squared error between two volumes

        Parameters
        ----------
        vol1 : 3D tensor
            First volume.
        vol2 : 3D tensor
            Second volume.

        Returns
        -------
        mse : float
            The MSE.

        """
        return torch.linalg.norm(vol1 - vol2)