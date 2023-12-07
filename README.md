# Ghostbuster: a diffraction tomography algorithm for cryo-EM single particle reconstruction via phase retrieval
[Paper](*insert url*) | [Data](*insert url*)

[Joel Yeo](https://github.com/joelyeois/), [Benedikt J. Daurer](https://scholar.google.com/citations?user=ukSgXPcAAAAJ&hl=en&inst=3212728378801010220&oi=ao), [Dari Kimanius](https://scholar.google.com/citations?user=noWvpR8AAAAJ&hl=en&inst=3212728378801010220&oi=ao), [Deepan Balakrishnan](https://scholar.google.com/citations?user=lRXoHK4AAAAJ&hl=en&inst=3212728378801010220&oi=ao), [Tristan Bepler](https://scholar.google.com/citations?user=Roxjki8AAAAJ&hl=en&inst=3212728378801010220&oi=ao), [Yong Zi Tan](https://scholar.google.com/citations?user=MO8j13QAAAAJ&hl=en&inst=3212728378801010220&oi=ao), [N. Duane Loh](https://scholar.google.com/citations?user=mLO7dRwAAAAJ&hl=en&inst=3212728378801010220&oi=ao)

This repository presents a minimal working example of the Ghostbuster algorithm for cryo-EM single particle reconstruction. Ghostbuster uses a two-step process to stretch the depth of field of a 3D particle reconstruction by accounting for the effects of wave propagation within a thick particle (similar to Ewald sphere curvature correction). First, phase retrieval is performed on the particle stack to recover the 2D complex-valued exitwaves from the real-valued intensity images -- here we use the [RAF](10.1109/TSP.2018.2818077) algorithm. Second, our developed DOFS algorithm performs a minibatch stocastic gradient descent to solve for the 3D particle which best reproduces the phase-retrieved exitwaves.

## Installation
To run the code, users may first install the required packages listed via ```pip -r requirements.txt```.
After which, the `ghostbuster` package can be installed via ```python setup.py install```.

## Demos
After installation, users may use the demo notebooks provided in the `notebooks` folder to run either RAF or DOFS individually, or the full Ghostbuster algorithm.

## Citation
If you find our work useful in your research, please cite:
```
@article{...,
  author={...},
  title={...},
  journal={...},
  year={...},
  month={...},
  day={...},
  volume={...},
  number={...},
  pages={...}
}
```

## License
Our code is licensed under GPL-3.0. By downloading the software, you agree to the terms of this License.
