import numpy as np
import matplotlib.pyplot as plt
from .dofs import DOFS
from .raf import RAF
import torch
from . import utils
from tqdm import tqdm

fft2 = lambda array: torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(array)))
ifft2 = lambda array: torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(array)))
ifftn = lambda array: torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(array)))
fftn = lambda array: torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(array)))

class Ghostbuster:
    def __init__(self, intensities, quats, defocuses, wavelength,
                 pixelsize=1, raf_device='cpu', dofs_device = 'cuda', alpha=0.1, 
                 dose=1, cs=2, dofs_batchsize=10):
        """
        Parameters
        ----------
        intensities : 3D ndarray
            Structure should be (no. measurements, ROI_x, ROI_y)
        quats : 2D ndarray
            Array of quaternions associated with the intensities list.
            Structured as (no. measurements, 4), where the columns are the
            quaternion components.
        defocuses : 1D ndarray
            Array of defocus parameters associated with the intensities list.
            This defocus is defined w.r.t. exitplane, NOT particle center.
            Structured as (no. measurements).
        wavelength : float
            Wavelength of electron beam in [Å]
        pixelsize : float
            Pixel size of detector in [Å]
        dose : float
            Dose per pixel area.
        """

        self.intensities = intensities
        self.quats = quats
        self.defocuses = defocuses
        self.wavelength = wavelength
        self.dx = pixelsize
        self.raf_device = raf_device
        self.dofs_device = dofs_device
        self.alpha = alpha
        self.dose = dose
        self.cs = cs
        self.dofs_batchsize = 10
        self.sh = intensities.shape[-1]
        self.n_particles = len(intensities)

    def ghostbusting(self, raf_niter=50, dofs_niter=1000, groundtruth=None,
                    dofs_stepsize=1, raf_stepsize=1, dofs_amplitude_contrast=True,
                    dofs_positivity=True, mask_size=200):
        self.raf_niter = raf_niter
        self.dofs_niter = dofs_niter
        self.groundtruth = groundtruth
        self.dofs_stepsize = dofs_stepsize
        self.raf_stepsize = raf_stepsize
        self.dofs_amplitude_contrast = dofs_amplitude_contrast
        self.dofs_positivity = dofs_positivity
        self.mask_size = 200

        # Begin RAF
        print('Begin RAF.')
        self.raf = RAF(self.intensities, self.quats, self.defocuses, self.wavelength,
                       pixelsize=self.dx, device=self.raf_device, alpha=self.alpha, 
                       dose=self.dose, cs=self.cs)
        self.raf.initialize(groundtruth=self.groundtruth)
        self.raf.iterate(niter=raf_niter)
        self.surrogate_volume = self.raf.volume.cpu()
        self.raf_exitwaves = self.raf.project(self.surrogate_volume)

        # Normalize raf exitwaves
        print('Normalizing RAF exitwaves')
        self.exitwaves = self.normalize_exitwaves(self.raf_exitwaves, self.mask_size)

        # Begin DOFS
        print('Begin DOFS')
        self.dofs = DOFS(self.exitwaves, self.quats, self.wavelength, voxelsize=self.dx, device=self.dofs_device, batchsize=self.dofs_batchsize)
        self.dofs.initialize(groundtruth=self.groundtruth)
        self.dofs.iterate(
            niter=self.dofs_niter, stepsize=self.dofs_stepsize, 
            amplitude_contrast=self.dofs_amplitude_contrast, positivity=self.dofs_positivity
        )
        self.reconstructed_particle = self.dofs.current.detach().cpu()
        
        print('Done!')
    
    def normalize_exitwaves(self, exitwaves=None, mask_size=200):
        if exitwaves is not None:
            exitwaves = self.raf_exitwaves.cpu().numpy()
        
        circle_mask = 1 - utils.circle2d(self.sh, mask_size)
        circle_mask_tiled = np.tile(circle_mask.numpy()[None,:,:], (self.n_particles, 1, 1)).astype(bool)

        #using numpy here as torch.mean does not have kwarg: where.
        background_phase = np.mean(np.angle(exitwaves), where=circle_mask_tiled)
        background_amp = np.mean(np.abs(exitwaves), where=circle_mask_tiled)
        waves_subtracted = np.zeros_like(exitwaves)
        for i in tqdm(range(self.n_particles)):
            phase = np.angle(exitwaves[i]) - background_phase
            waves_subtracted[i] = np.abs(exitwaves[i]) * np.exp(1j*phase) / background_amp
        return torch.from_numpy(waves_subtracted)