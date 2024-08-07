# Ghostbuster: a phase retrieval diffraction tomography algorithm for cryo-EM
[Paper](https://doi.org/10.1016/j.ultramic.2024.113962) | [Data](https://doi.org/10.5281/zenodo.10297508)

[Joel Yeo](https://scholar.google.com/citations?user=2HW3Xs0AAAAJ&hl=en&inst=3212728378801010220&oi=sra), [Benedikt J. Daurer](https://scholar.google.com/citations?user=ukSgXPcAAAAJ&hl=en&inst=3212728378801010220&oi=ao), [Dari Kimanius](https://scholar.google.com/citations?user=noWvpR8AAAAJ&hl=en&inst=3212728378801010220&oi=ao), [Deepan Balakrishnan](https://scholar.google.com/citations?user=lRXoHK4AAAAJ&hl=en&inst=3212728378801010220&oi=ao), [Tristan Bepler](https://scholar.google.com/citations?user=Roxjki8AAAAJ&hl=en&inst=3212728378801010220&oi=ao), [Yong Zi Tan](https://scholar.google.com/citations?user=MO8j13QAAAAJ&hl=en&inst=3212728378801010220&oi=ao), [N. Duane Loh](https://scholar.google.com/citations?user=mLO7dRwAAAAJ&hl=en&inst=3212728378801010220&oi=ao)

This repository presents a minimal working example of the Ghostbuster algorithm for cryo-EM single particle reconstruction. Ghostbuster uses a two-step process to stretch the depth of field of a 3D particle reconstruction by accounting for the effects of wave propagation within a thick particle (similar to Ewald sphere curvature correction). First, phase retrieval is performed on the particle stack to recover the 2D complex-valued exitwaves from the real-valued intensity images -- here we use the [RAF](http://dx.doi.org/10.1109/TSP.2018.2818077) algorithm. Second, our developed DOFS algorithm performs a minibatch stocastic gradient descent to solve for the 3D particle which best reproduces the phase-retrieved exitwaves.

## Installation
To run the code, users may first install the required packages listed via ```pip install -r requirements.txt```.
After which, the `ghostbuster` package can be installed via ```python setup.py install```.
The dataset for the demo notebooks may be downloaded [here](https://doi.org/10.5281/zenodo.10297508).

## Demos
After installation, users may use the demo notebooks provided in the `notebooks` folder to run either RAF or DOFS individually, or the full Ghostbuster algorithm.

## Citation
If you find our work useful in your research, please cite:
```
@article{YEO2024113962,
title = {Ghostbuster: A phase retrieval diffraction tomography algorithm for cryo-EM},
journal = {Ultramicroscopy},
volume = {262},
pages = {113962},
year = {2024},
issn = {0304-3991},
doi = {https://doi.org/10.1016/j.ultramic.2024.113962},
url = {https://www.sciencedirect.com/science/article/pii/S030439912400041X},
author = {Joel Yeo and Benedikt J. Daurer and Dari Kimanius and Deepan Balakrishnan and Tristan Bepler and Yong Zi Tan and N. Duane Loh},
keywords = {Diffraction tomography, Phase retrieval, Cryogenic electron microscopy, Single particle reconstruction, Ewald sphere curvature correction},
}
```

## License
Our code is licensed under CC BY-NC-SA 4.0 DEED. By downloading the software, you agree to the terms of this License.
