# Introduction 

This is the code repository for "Neural network-based audio reconstruction from
undersampled STFT magnitude data". Read our paper published in EUSIPCO 2024 (here)[https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000406.pdf]!

# Dataset

Our experiments were conducted on the AudioMNIST dataset. You can download it here: 
[https://github.com/soerenab/AudioMNIST]. Store all data in a directory titled `AudioMNIST/`. 

# Training NN for phase retrieval

We trained an NN to do STFT phase retrieval for 4L, 2L, L, L/2 measurements where 
L = 8000. To train an NN on the AudioMNIST set with the given number of measurements,
run the corresponding python file among `Train_16254.py`, `Train_32379.py`, `Train_8190.py`,
or `Train_4095.py`. These require standard Python packages including `torch`, `numpy`, `scipy`,
and `pandas`. For convenience, we have included a YAML file `pr.yml` so you may compile a python
environment with all the correct dependencies for running the above scripts out of the box. 

# Evaluating the NN 

After training the NN for the given sampling ratio, you can evaluate it in the `TestModels.py` 
file to generate the data for the respective row in Table 1 of the paper. 

# Benchmarking: FGL

Fast Griffin Lim (FGL): The FGL column of table 1 can be generated by running the `TestGL.py`
file. The tests for Table 2 may also be performed using this script. 

# Benchmarking: AF/WF 

We ran the amplitude and Wirtinger flow (AF/WF) algorithms in MATLAB via `phasepack`. To run these 
benchmarks you will need a MATLAB installation. After installing MATLAB, clone the GitHub repo 
`https://github.com/tomgoldstein/phasepack-matlab`. To make the reconstructions, run the `audio_phase_ret.m` 
file, setting the `opts.Algorithm` parameter to either `AmplitudeFlow` or `WirtFlow`. Depending on the measurement
setting, line 15 of `audio_phase_ret.m` may be changed to load any one of the measurement matrices `A_sparse_{32-4}K.mat`

To compute the SC metric, use the `computer_metrix.m` file. 

# Data and model availability 

Trained models can be found [here](https://umd.box.com/s/5gey8m0r98o9ycll8or23lvecl5i0wiu). 
