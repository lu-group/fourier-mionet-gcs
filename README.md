# Fourier-MIONet for geological carbon sequestration (GCS)

The data and code for the paper [Z. Jiang, M. Zhu, & L. Lu. Fourier-MIONet: Fourier-enhanced multiple-input neural operators for multiphase modeling of geological carbon sequestration. *Reliability Engineering & System Safety*, 251, 110392, 2024.](https://doi.org/10.1016/j.ress.2024.110392)

## Data
Download data from [google drive](https://drive.google.com/drive/folders/1OJruFzi2dO8Xwo7XrS_zAhmsGFJQ-imL?usp=sharing)

## Code
Note: When you run the codes, please use the modified deepxde under the directory of corresponding python file. The codes cannot run under the public version of deepxde.

#### Fourier-MIONet
- Run [Fourier-MIONet_sg.py](Fourier-MIONet_sg.py) for gas saturation.
- Run [Fourier-MIONet_dP.py](Fourier-MIONet_dP.py) for pressure buildup.

#### vanilla MIONet
- Run [MIONet_vanilla_SG.py](baselines/MIONet_vanilla_SG.py) for gas saturation.
- Run [MIONet_vanilla_dP.py](baselines/MIONet_vanilla_dP.py) for pressure buildup.

#### MIONet-FNN
- Run [MIONet_FNN_SG.py](baselines/MIONet_FNN_SG.py) for gas saturation.
- Run [MIONet_FNN_dP.py](baselines/MIONet_FNN_dP.py) for pressure buildup.

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{jiang2024fourier,
  title   = {{Fourier-MIONet}: Fourier-enhanced multiple-input neural operators for multiphase modeling of geological carbon sequestration},
  author  = {Jiang, Zhongyi and Zhu, Min and Lu, Lu},
  journal = {Reliability Engineering \& System Safety},
  volume  = {251},
  pages   = {110392},
  year    = {2024},
  doi     = {https://doi.org/10.1016/j.ress.2024.110392}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
