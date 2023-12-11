import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from . import utils
from .dofs import DOFS
from .raf import RAF

fft2 = lambda array: torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(array)))
ifft2 = lambda array: torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(array)))
ifftn = lambda array: torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(array)))
fftn = lambda array: torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(array)))


class Ghostbuster:
    def __init__(
        self,
        intensities,
        quats,
        defocuses,
        wavelength,
        pixelsize=1,
        raf_device="cpu",
        dofs_device="cuda",
        alpha=0.1,
        dose=1,
        cs=2,
        dofs_batchsize=10,
    ):
        """
        Parameters
        ----------
        intensities : 3D tensor
            Structure should be (no. measurements, len_y, len_x)
        quats : 2D tensor
            Array of quaternions associated with the intensities list.
            Structured as (no. measurements, 4), where the columns are the
            quaternion components.
        defocuses : 1D tensor
            Array of defocus parameters associated with the intensities list.
            This defocus is defined w.r.t. exitplane, NOT particle center.
            Structured as (no. measurements).
        wavelength : float
            Wavelength of electron beam in [Å]
        pixelsize : float, optional
            Pixel size of detector in [Å]. Default: 1.
        raf_device : str, optional
            The device to perform RAF. Default: 'cpu'.
        dofs_device : str, optional
            The device to perform DOFS. Default: 'cuda'.
        alpha : float, optional
            Amplitude contrast ratio. Default: 0.1.
        dose : float, optional
            Dose per pixel area. Default: 1.
        cs : float, optional
            Spherical aberration parameter in [mm]. Default: 2.
        dofs_batchsize : optional
            Batchsize for DOFS algorithm. Default: 10.
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

    def ghostbusting(
        self,
        raf_niter=50,
        dofs_niter=5000,
        groundtruth=None,
        dofs_stepsize=1,
        raf_stepsize=1,
        dofs_amplitude_contrast=True,
        dofs_positivity=True,
        mask_size=200,
    ):
        """
        The Ghostbuster algorithm.

        Parameters
        ----------
        raf_niter : int, optional
            Number of RAF iterations. Default: 50.
        dofs_niter : int, optional
            Number of DOFS iterations. Default is 5000.
        groundtruth : 3D tensor, optional
            The ground truth 3D particle for error metric calculation.
        dofs_stepsize : float, optional
            The stepsize for DOFS. Default: 1.
        raf_stepsize : float, optional
            The stepsize for RAF. Default: 1.
        dofs_amplitude_contrast : bool, optional
            Enforces amplitude contrast ratio constraint between imaginary and real
            part of slices for DOFS. Default: True.
        dofs_positivity : bool, optional
            Enforces positivity constraint for DOFS. Default: True.
        mask_size : int, optional
            The mask size in [pixels] used to normalize the exitwaves to unit
            background with zero phase. Default: 200.
        """
        self.raf_niter = raf_niter
        self.dofs_niter = dofs_niter
        self.groundtruth = groundtruth
        self.dofs_stepsize = dofs_stepsize
        self.raf_stepsize = raf_stepsize
        self.dofs_amplitude_contrast = dofs_amplitude_contrast
        self.dofs_positivity = dofs_positivity
        self.mask_size = 200

        # Begin RAF
        print("Begin RAF.")
        self.raf = RAF(
            self.intensities,
            self.quats,
            self.defocuses,
            self.wavelength,
            pixelsize=self.dx,
            device=self.raf_device,
            alpha=self.alpha,
            dose=self.dose,
            cs=self.cs,
        )
        self.raf.initialize(groundtruth=self.groundtruth)
        self.raf.iterate(niter=raf_niter)
        self.surrogate_volume = self.raf.volume.cpu()
        self.raf_exitwaves = self.raf.project(self.surrogate_volume)

        # Normalize raf exitwaves
        print("Normalizing RAF exitwaves")
        self.exitwaves = self.normalize_exitwaves(self.raf_exitwaves, self.mask_size)

        # Begin DOFS
        print("Begin DOFS")
        self.dofs = DOFS(
            self.exitwaves,
            self.quats,
            self.wavelength,
            voxelsize=self.dx,
            device=self.dofs_device,
            batchsize=self.dofs_batchsize,
        )
        self.dofs.initialize(groundtruth=self.groundtruth)
        self.dofs.iterate(
            niter=self.dofs_niter,
            stepsize=self.dofs_stepsize,
            amplitude_contrast=self.dofs_amplitude_contrast,
            positivity=self.dofs_positivity,
        )
        self.reconstructed_particle = self.dofs.current.detach().cpu()

        print("Done!")

    def normalize_exitwaves(self, exitwaves=None, mask_size=200):
        """
        Normalizes the exitwaves by setting background to unit amplitude and zero
        phase.

        Parameters
        ----------
        exitwaves : 3D tensor, optional
            The exitwaves to be normalized, with shape of (batchsize, len_y, len_x).
        mask_size : int, optional
            To be set to the diameter of the particle in pixels. Every pixel
            outside the mask will be used to compute the mean background amplitude
            and phase. Default: 200.

        Returns
        -------
        normalized_exitwaves : 3D tensor
            The normalized exitwaves.

        """
        if exitwaves is not None:
            exitwaves = self.raf_exitwaves.cpu().numpy()

        circle_mask = 1 - utils.circle2d(self.sh, mask_size)
        circle_mask_tiled = np.tile(
            circle_mask.numpy()[None, :, :], (self.n_particles, 1, 1)
        ).astype(bool)

        # using numpy here as torch.mean does not have kwarg: where.
        background_phase = np.mean(np.angle(exitwaves), where=circle_mask_tiled)
        background_amp = np.mean(np.abs(exitwaves), where=circle_mask_tiled)
        waves_subtracted = np.zeros_like(exitwaves)
        for i in tqdm(range(self.n_particles)):
            phase = np.angle(exitwaves[i]) - background_phase
            waves_subtracted[i] = (
                np.abs(exitwaves[i]) * np.exp(1j * phase) / background_amp
            )
        return torch.from_numpy(waves_subtracted)