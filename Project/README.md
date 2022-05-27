# Deep learning miniprojects

This repository contains our implementation of the two mini-projects of François Fleuret's _Deep Learning_ course at EPFL (EE-559).

Authors:
* *Guanqun Liu*
* [*Xianjie Dai*](https://github.com/xianjiedai)

## Miniproject 1: U-Net based Noise2Noise Auto-encoder Image Denoiser
The goal of this project is to implement a neural network to predict the laterality of an upcoming finger movement (left or right hand) from the EEG recording 130 ms before key-press. This is a standard two-class classification problem.


## Miniproject 2: Implementation of a basic deep learning framework from scratch 
The goal of this project is to design a mini auto-encoder using only pytorch’s tensor operations and the standard math library, hence in particular without using autograd or the neural-network modules.

The following modules were implemented:
- ReLU activation function,
- Conv2d/ConvTranspose2d,
- MSE loss function,
- SGD optimizer,
- A sequential module to combine several modules in a basic sequential structure.

Complete description and instructions of both mini-projects can be found in `EE559_2022ProjDescription.pdf`.

© 2022 GitHub, Inc.
