"""
Depth of field stretching (DOFS) algorithm.
@author: Joel Yeo, joelyeo@u.nus.edu
"""
import torch
from scipy.spatial.transform import Rotation
import numpy as np
from tqdm import tqdm
from . import utils

fft2 = lambda array: torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(array)))
ifft2 = lambda array: torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(array)))

class DOFS:
    def __init__(self, exitwaves, quats, wavelength, voxelsize=1, batchsize=10, device='cpu',
                alpha=0.1):
        """
        Parameters
        ----------
        exitwaves : 3D ndarray, optional
            Structure should be (no. measurements, ROI_x, ROI_y)
        quats : 2D ndarray
            Array of quaternions associated with the exitwaves list. 
            Structured as (no. measurements, 4), where the columns are the
            quaternion components in (w,x,y,z).
        wavelength : float
            Wavelength of the electron beam in [Å]. Not required if energy provided.
        voxelsize : float, optional
            Voxel size in [Å]
        batchsize : int
            Number of exitwaves/intensities per epoch. Default: 10.
        device : str
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
        
        #setup orientations
        self.quats = quats.to(self.device)
        self.matrices = torch.from_numpy(self.quaternion_to_rotation_matrices(quats)).to(self.device)

        #setup kspace
        self.voxelsize = voxelsize
        self.k = (utils.create_kspace(self.sh, self.voxelsize)).to(self.device)
                
        self.wavelength = wavelength
        self.alpha = torch.tensor(alpha)
                        
        # create propagators and transfer functions
        self.create_linear_propagators()

    def quaternion_to_rotation_matrices(self, quaternions):
        '''
        Generates rotation matrices from quaternions.
    
        Parameters
        ----------
        quaternions : 1D or 2D array
            Quaternion array is (w,x,y,z) in accordance to quaternion.py convention.
            If 2D, dimensions are (num_angles, 4)
    
        Returns
        -------
        rm : 3D array
            (num_angles, 3, 3) rotation matrices.
    
        '''
        # Roll to match scipy's convention of (x,y,z,w)
        if len(quaternions.shape) == 1:
            R = Rotation.from_quat(np.roll(quaternions, -1))
        elif len(quaternions.shape) == 2:
            R = Rotation.from_quat(np.roll(quaternions, -1, axis=1))
        return R.as_matrix()

    def create_linear_propagators(self):
        # assume equally spaced slices. First element corresponds to
        # the first slice located furthest from the exitwave plane
        self.distances = torch.from_numpy(np.flip(np.arange(self.sh) + 1) * self.voxelsize).to(self.device)
        self.linear_forward_kernels = torch.zeros((self.sh, self.sh, self.sh), dtype=torch.complex128).to(self.device)
        self.linear_backward_kernels = torch.zeros_like(self.linear_forward_kernels).to(self.device)
        for i, d in enumerate(self.distances):
            self.linear_forward_kernels[i] = torch.exp(1j * torch.pi * self.wavelength * d * (self.k**2))
            self.linear_backward_kernels[i] = torch.exp(-1j * torch.pi * self.wavelength * d * (self.k**2))

    def initialize(self, vol=None, groundtruth=None, store_mse=True):
        self.store_mse = store_mse
        self.iteration_number = 0
        self.currentbatchorder = torch.arange(self.num_data)
        
        if vol is None:
            self.current = torch.zeros((self.sh, self.sh, self.sh), dtype=torch.complex128, device=self.device)
        else:
            self.current = vol.to(self.device)
                 
        if groundtruth is not None:
            self.groundtruth = groundtruth.to(self.device)
            if store_mse:
                self.mses = [self.compute_mse(self.current, self.groundtruth)]
    
    def iterate(self, niter=1000, stepsize=1, positivity=True, amplitude_contrast=True):
        self.stepsize = stepsize
        self.niter = niter
        self.currentbatch = self.currentbatchorder[:self.batchsize]
        self.positivity = positivity
        self.amplitude_contrast = amplitude_contrast
        
        for i in tqdm(range(niter)):
            self.iteration_number += 1
            self.update(stepsize=stepsize)
            
            #change batch order
            self.currentbatchorder = torch.roll(self.currentbatchorder, -self.batchsize)
            self.currentbatch = self.currentbatchorder[:self.batchsize]
    
    def update(self, stepsize=1):
        # rotation matrices for current batch
        matrices = self.matrices[self.currentbatch]
        
        # rotate volumes
        rotated_currents = utils.rotate_3d(torch.tile(self.current, (self.batchsize, 1, 1, 1)), matrices)
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
            self.current = self.current.real + 1j * self.current.real / (torch.sqrt(1 - self.alpha**2) / self.alpha)
        
        # calculate metrics
        if self.store_mse:
            self.mses.append(self.compute_mse(self.current, self.groundtruth))
    
    def linear_multislice(self, slices, forward_kernels=None):
        if forward_kernels is None:
            forward_kernels = self.linear_forward_kernels
        slices_f = fft2(slices)
        p_slices = ifft2(slices_f * self.linear_forward_kernels)
        exitwaves = 1 + 1j * p_slices.sum(axis=1)
        return exitwaves
    
    def linear_backpropagate_residues(self, residues, backward_kernels=None):
        if backward_kernels is None:
            backward_kernels = self.linear_backward_kernels
        
        residues_f = fft2(residues)
        tiled_residues_f = torch.tile(residues_f[:, None, :, :], (1, self.sh, 1, 1))
        p_residues = ifft2(tiled_residues_f * backward_kernels)
        return -2j * p_residues
 
    def compute_mse(self, vol1, vol2):
        return torch.linalg.norm(vol1 - vol2)