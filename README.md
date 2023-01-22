# Tutorials
This code uses a Metropolis Markov-Chain Monte-Carlo algorithm in order to sample parameters of a toy gravitational wave signal buried in coloured LISA like noise. The waveform model we will use is

$$ h(t;a,f,\dot{f}) = a \sin \left(2\pi t \left[f + \frac{1}{2}\dot{f}t\right]\right) $$

and we aim to estimate the parameter set $\boldsymbol{\theta} = \{a,f,\dot{f}\}$.

## Getting started
1. Install Anaconda if you do not have it
2. Create a virtual environment using:

    > `conda create -n mcmc_tutorial -c conda-forge numpy scipy matplotlib astropy corner 
    conda activate mcmc_tutorial`



## The code structure
1. The script `LISA_utils.py` is a utility script containing useful python functions. Such functions include an approximate parametric model for the LISA power spectral density, for example.
2. The script `mcmc_func.py` includes the waveform model and scripts used to build the Metropolis sampler.
3. The script `mcmc.py` executes the metropolis algorithm. Within this script, set `Generate_Plots = True` if you want to create movies as the sampler goes along. 

## How to use the code

To execute the code:
1. Source the environment above: `conda activate mcmc_tutorial` 
2. Run `python mcmc.py`

## How to generate movies
                                  
1. For MACOSX, If you want to create movies you will want to install: `brew install ImageMagick`.
2. Create directories for the movies by writing in the shell: `source setup/set_direc` . This will build the relevant directories to save the still images to.
3. Run the script: `python mcmc.py` with `Generate_Plots = True` in the function `MCMC_run` located in the `mcmc.py` script.
4. Create the movies by writing in the shell `source setup/make_movies` this may take some time, so be patient. It will also delete the directories that the still images were saved to. 
5. Enjoy your movies! Watch them with your friends and family! 
