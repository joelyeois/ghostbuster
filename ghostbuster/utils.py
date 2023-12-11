"""
Helper functions.
@author: Joel Yeo, joelyeo@u.nus.edu
"""

import torch

def rotate_3d(V, rotation_matrix, padding_mode='border'):
    '''
    Takes in a 3D array and a 3x3 rotation matrix, and performs a rotation of
    the 3D array using pytorch interpolation.

    Parameters
    ----------
    V : 3D or 4D tensor
        The volume to be rotated. Can be either real or complex. If 4D, the shape is
        (no. vols, len_z, len_y, len_x).
    rotation_matrix : 2D or 3D tensor
        The 3x3 rotation matrix. If 3D, the shape is (no. vols, 3, 3).
        quaternions.
    padding_mode : str, optional
        The padding mode for F.grid_sample.

    Returns
    -------
    rotated_V : 3D or 4D tensor
        The rotated volume or volumes.
    '''
    #no_grad disables unnecessary gradient calculation since we just want
    #to use affine_grid for rotation
    with torch.no_grad():
        
        #new axes and batch sizes are to convert the tensor shapes into a form
        #which pytorch likes
        if len(rotation_matrix.shape) == 2:
            B = 1
            nz, ny, nx = V.shape
            V_gpu = V[None, None, ...]
            rm_gpu = rotation_matrix[None,...]
        elif len(rotation_matrix.shape) == 3:
            B, nz, ny, nx = V.shape
            V_gpu = V[:, None, :, :, :]
            rm_gpu = rotation_matrix
        
        
        device = V_gpu.device
        #appending the translation vector to fom an affine transform matrix
        rm_gpu = torch.cat([rm_gpu, torch.zeros([B, 3, 1]).to(device)], dim=2)
        
        # create the grid
        null_rot = torch.zeros_like(rm_gpu)
        null_rot[:,0,0] = 1.0
        null_rot[:,1,1] = 1.0
        null_rot[:,2,2] = 1.0
        grid = torch.nn.functional.affine_grid(null_rot, (B, 1, nz, ny, nx), align_corners=False)
        
        # rotate the grid around the center defined by RELION
        cz = int(nz/2)
        cy = int(ny/2)
        cx = int(nx/2)
        center = grid[:, cz, cy, cx] # (B, 3)
        center = center.unsqueeze(1)
        grid = grid.view(B, nz*ny*nx, 3)
        grid = (grid - center).bmm(rm_gpu[..., :-1].transpose(1,2)) + center

        # translate the grid
        dx = rm_gpu[..., -1].unsqueeze(1) # (B, 1, 3)
        grid = grid + dx
        grid = grid.view(B, nz, ny, nx, 3)
        
        #run one time for real-volume
        if not torch.is_complex(V_gpu):
            VV = torch.nn.functional.grid_sample(V_gpu, grid, align_corners=False, padding_mode=padding_mode)
            return torch.squeeze(VV)
        
        #run twice for real and imag parts and piece together
        else:
            real_VV = torch.nn.functional.grid_sample(V_gpu.real, grid, align_corners=False, padding_mode=padding_mode)
            imag_VV = torch.nn.functional.grid_sample(V_gpu.imag, grid, align_corners=False, padding_mode=padding_mode)
            return torch.squeeze(real_VV + 1j * imag_VV)

def create_kspace(n, pixelsize):
    '''
    Creates kspace meshgrid from real space (n,n) array.

    Parameters
    ----------
    n : int
        Number of pixels along x and y directions. Assumed same in both directions.
    pixelsize : float
        Pixel size in [A]. Assume same along both x and y directions.

    Returns
    -------
    k : 2D array
        The kspace meshgrid.
    '''
    kx = torch.fft.fftfreq(n, pixelsize)
    ky = torch.fft.fftfreq(n, pixelsize)
    kxx, kyy = torch.meshgrid(kx, ky, indexing='ij')
    k = torch.fft.fftshift(torch.sqrt(kxx**2 + kyy**2))
    return k

def circle2d(N, d):
    """
    Generates 2D tensor for a filled-in circle.

    Parameters
    ----------
    N : int
        Length of grid in pixels.
    d : int
        Diameter in pixels.

    Returns
    -------
    circle : 2D tensor
        2D tensor with centered circle
    ....
    """
    circle = torch.zeros((N, N))
    x = torch.linspace(-1, 1, N)
    y = x
    [X, Y] = torch.meshgrid(x, y, indexing='xy')
    circle[X**2 + Y**2 <= (d / N)**2] = 1
    return circle