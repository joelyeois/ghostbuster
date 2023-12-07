"""
Reweighted amplitude flow (RAF) algorithm.
Modified from Wang et al. (2018): 10.1109/TSP.2018.2818077
@author: Joel Yeo, joelyeo@u.nus.edu
"""
import numpy as np
from . import tomo
import torch
from . import utils
from tqdm import tqdm

# helper functions
fft2 = lambda array: torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(array)))
ifft2 = lambda array: torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(array)))
ifftn = lambda array: torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(array)))
normalize = lambda arr: (arr - torch.median(arr))


class RAF:
    def __init__(
        self,
        intensities,
        quats,
        defocuses,
        wavelength,
        pixelsize=1,
        device="cpu",
        alpha=0.1,
        dose=1,
        cs=2,
    ):
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
        self.amplitudes = torch.sqrt(intensities)
        self.quats = quats
        self.defocuses = defocuses
        self.wavelength = wavelength
        self.dx = pixelsize
        self.dose = torch.tensor(dose)
        self.alpha = torch.tensor(alpha)
        self.cs = torch.tensor(cs)
        self.sh = intensities[0].shape[0]
        self.k = utils.create_kspace(self.sh, self.dx)
        self.device = device
        self.vignette_masks = None
        self.store_mse = False

        self.create_transfer_function()

    def create_transfer_function(self):
        """
        Creates transfer function for each particle.

        .. math::
        H = \exp{-i\chi}
        """
        # tile arrays for broadcasting multiplication later, becomes size
        # (no. measurements, self.sh, self.sh)
        defocus_tiled = torch.tile(self.defocuses[:, None, None], (1, self.sh, self.sh))

        # aberration function
        self.chi = torch.pi * (
            -defocus_tiled * self.wavelength * self.k**2
            + 0.5 * self.cs * self.wavelength**3 * self.k**4
        )

        # transfer function
        self.H = torch.exp(-1j * self.chi).to(self.device)

    def create_ctf(self):
        """
        Creates CTF for each particle.

        .. math::
        \chi = \sqrt{1-\alpha^2} \sin{\chi} - \alpha \cos{\chi}
        """
        self.ctf = torch.sqrt(1 - self.alpha**2) * torch.sin(
            self.chi
        ) - self.alpha * torch.cos(self.chi)

    def initialize(self, vol=None, groundtruth=None):
        """
        Creates initial 3D estimate of the exitwave volume

        Parameters
        ----------
        vol : 3D ndarray, optional
            User-specified initial 3D exitwave volume estimate. The default
            is None.
        """
        if vol is not None:
            # initialize with user provided volume
            self.volume = vol
        else:
            print("Initializing volume via Pseudoinverse.")
            # reconstruct complex volume
            self.volume = self.pseudoinverse()

        if groundtruth is not None:
            self.groundtruth = groundtruth
            self.store_mse = True
            if self.store_mse:
                self.mses = [self.compute_mse(self.volume, self.groundtruth)]
        self.init_vol = self.volume.clone()

        if self.vignette_masks is None:
            self.create_vignette_masks()

    def iterate(self, stepsize=1, niter=50):
        """
        Performs RAF iterations.

        Parameters
        ----------
        stepsize : float, optional
            Gradient stepsize. Default is 1.
        niter : int, optional
            Number of iterations. Default is 50.
        """
        self.stepsize = stepsize
        self.niter = niter
        for i in tqdm(range(niter)):
            self.volume = self.raf_update(self.volume)

            if self.store_mse:
                self.mses.append(self.compute_mse(self.volume, self.groundtruth))

    def propagate(self, waves, H, direction="backward"):
        """
        Propagates waves forward or backwards through microscope optics.

        Parameters
        ----------
        waves : 3D tensor
            User-specified initial 3D exitwave volume estimate. The default
            is None.
        H : 3D tensor
            Associated transfer function for each exitwave. Must be same size
            as waves.
        """
        if direction == "forward":
            wavesf = fft2(waves * torch.sqrt(self.dose))
            waves_out = ifft2(wavesf * H)
        elif direction == "backward":
            wavesf = fft2(waves / torch.sqrt(self.dose))
            waves_out = ifft2(wavesf / H)
        return waves_out

    def reconstruct(self, projs, quats=None):
        if quats is None:
            quats = self.quats

        # tomography functions only work on numpy arrays for now.
        projs_np = projs.cpu().numpy()
        quats_np = quats.cpu().numpy()

        vw_real, vw_imag = tomo.fourier_reconstruction(
            projs_np,
            quats_np,
            is_fourier=False,
            return_real=False,
            return_weights=True,
            mask=True,
        )
        vw = vw_real + vw_imag
        fsum = vw[0] + 1j * vw[2]

        # bring back to torch
        fvol = torch.from_numpy(fsum)
        return ifftn(fvol)

    def project(self, vol, quats=None):
        if quats is None:
            quats = self.quats

        # tomography functions only work on numpy arrays for now.
        # vol_np = vol.cpu().numpy()
        # quats_np = quats.cpu().numpy()

        # projs = tomo.fourier_slices(vol_np, quats_np,
        #                            is_fourier=False,
        #                            return_fourier=False,
        #                            return_real=False,
        #                            mask=False)

        # use new torch implementation
        projs = tomo.projection(vol, quats, batchsize=10, mask=False)

        if self.vignette_masks is None:
            return projs
        else:
            return projs / self.vignette_masks

    def create_vignette_masks(self, size=None, quats=None):
        print("Creating vignette masks.")
        if quats is None:
            quats = self.quats
        if size is None:
            size = self.sh
        n = len(quats)

        flat_projections = torch.ones((n, size, size))
        vignette_vol = self.reconstruct(flat_projections, quats)
        self.vignette_masks = self.project(vignette_vol, quats)

        print("Vignette masks creation completed.")

    def raf_update(self, vol, eta=0.8):
        # Reweighted amplitude flow algorithm, Wang (2018),
        # DOI: 10.1109/TSP.2018.2818077

        # potential projections
        projections = self.project(vol).to(self.device)

        # forward propagate to defocus planes
        waves = self.propagate(projections, self.H, direction="forward")
        waves = waves.cpu()

        # compute ratio and weights
        ratio = torch.abs(waves) / self.amplitudes
        ratio[torch.isnan(ratio)] = 0
        weights = ratio / (ratio + eta)
        weights[torch.isnan(weights)] = 0

        # compute residues
        C = weights * (waves - self.amplitudes * waves / torch.abs(waves))

        # backpropagate residues
        gradients = self.propagate(C.to(self.device), self.H, direction="backward")

        # reconstruct 3D gradient
        grad_vol = self.reconstruct(gradients)

        # update solution
        vol = vol - self.stepsize * grad_vol
        return vol

    def pseudoinverse(self, eps=0.1):
        # Eq.2.22
        # create CTF first
        self.create_ctf()

        # spectral intensities
        G = fft2(torch.abs(self.amplitudes) ** 2)

        # creates boolean array for CTF elements larger than eps
        nonzero = torch.abs(self.ctf) > eps

        # storage array for reconstructed spectrum
        F = torch.zeros_like(self.amplitudes, dtype=torch.complex128)

        # Pseudoinverse
        F[nonzero] = (G / (2 * self.dose * self.ctf))[nonzero]
        F[torch.isnan(F)] = 0

        # recover 2D atomic potential slices in real-space
        vz = ifft2(F).real

        # normalize projections w.r.t. background
        vz_norm = normalize(vz)

        # convert to complex potentials
        potentials = self.complex_potential(vz_norm)

        # convert to exitwaves
        exitwaves = torch.exp(1j * potentials)

        # reconstruct 3D exitwave volume
        vol = self.reconstruct(exitwaves)
        return vol

    def complex_potential(self, real_v):
        return torch.sqrt(1 - self.alpha**2) * real_v + 1j * self.alpha * real_v

    def compute_mse(self, vol1, vol2):
        return torch.linalg.norm(vol1 - vol2)