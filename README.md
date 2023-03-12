# Tutorials
This code uses a Markov-Chain Monte-Carlo algorithms in order to sample parameters of a toy gravitational wave signal buried in coloured LISA like noise. The waveform model we will use is

$$ h(t;a,f,\dot{f}) = a \sin \left(2\pi t \left[f + \frac{1}{2}\dot{f}t\right]\right) $$

and we aim to estimate the parameter set $\boldsymbol{\theta} = \{a,f,\dot{f}\}$ using various samplers

## Code structure
In this repository there exists two directories, one called `metropolis` and the other called `eryn`. The first is a basic metropolise MCMC algorithm that can be used to teach the basics of simple samplers. The second is Eryn, developed by Michael Katz, Nikos Karnesis, Natalia Korsokova and Jonathan Gair. Built off `emcee` with extra fantastic features such as reversible jump MCMC [unknown number of signals with unknown number of parameters] and parallel tempering. The documentation can be found [here](https://mikekatz04.github.io/Eryn/html/user/ensemble.html).

## Getting started
1. Install Anaconda if you do not have it.
2. Create a virtual environment using:

    > `conda create -n mcmc_tutorial -c conda-forge numpy scipy matplotlib astropy corner tqdm jupyter 
    conda activate mcmc_tutorial`
3. If you intend to use `eryn` then after step 2 type into the shell 
   > `pip install git+https://github.com/mikekatz04/Eryn.git`


## The code structure -- metropolis
1. The script `LISA_utils.py` is a utility script containing useful python functions. Such functions include an approximate parametric model for the LISA power spectral density, for example.
2. The script `mcmc_func.py` includes the waveform model and scripts used to build the Metropolis sampler.
3. The script `mcmc.py` executes the metropolis algorithm. Within this script, set `Generate_Movies = True` if you want to create movies as the sampler goes along. 
4. The script `plotting_code.py` plots important parts of the sampler as it runs. It plots: 
   1. Evolution of the proposed waveform as the points get closer and closer to the true waveform.
   2. The trace plot before burnin.
   3. The matched filter statistic that should tend towards the optimal matched filtering signal-to-noise ratio.
   4. The trace plot after burnin.
   5. The evolution of the posterior shown by a corner plot.

### How to use the code 

To execute the code:
1. Locate the `metropolis` directory 
2. Source the environment above: `conda activate mcmc_tutorial` 
3. Run `python mcmc.py`
4. Once the code has finished executing, it will terminate on a `breakpoint()`. Here you can investigate the samples within your shell.   

### How to generate movies
                                  
1. For MACOSX, If you want to create movies you will want to install: `ImageMagick` using a package manager. I usually use brew: ``brew install ImageMagick``.
2. Create directories for the movies by writing in the shell: `source setup/set_direc` . This will build the relevant directories to save the still images to.
3. Run the script: `python mcmc.py` with `Generate_Movies = True` in the function `MCMC_run` located in the `mcmc.py` script.
4. Create the movies by writing in the shell `source setup/make_movies` this may take some time, so be patient. It will also delete the directories that the still images were saved to. 
5. Enjoy your movies! Watch them with your friends and family! 

## The code structure -- eryn

1. The script `LISA_utils.py` is a utility script containing useful python functions. Such functions include an approximate parametric model for the LISA power spectral density, for example.
2. The script `mcmc_run.py` executes eryn with the choice to use parallel tempering.

### How to use the code

To execute the code:
1. Locate the `eryn` directory
2. source the environment above `conda activate mcmc_tutorial`
3. Run python `mcmc_run.py`
4. A backend will be saved with some file name. Opening `jupyter notebook Analyse_samples.ipynb` will describe how to generate corner plots and trace plots.

