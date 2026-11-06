### Project Checkpoint Code

The repo contains two different folders, practice and rotated-MNIST. The practice folder contains the assignments that were completed using the neuron data from the Allen Institute, where neuron unit firings were recorded using an electrode insertion into mice across different types of visual stimuli. 

The rotated-MNIST contains the data, setup, and starter code for the rotated MNIST dataset that will be used for the quarter 1 project. 

## Data Access and running notebooks

1. In order to access the data, start by creating and activating the environment using the following lines. 

conda env create -f environment.yml -n allensdk-test
conda activate allensdk-test

2. Download the recordings

python ecephys_setup.py
If this doesn't work, try conda run --no-capture-output -n allensdk-test python ecephys_setup.py

Wait a couple minutes until the session data starts downloading and finishes for both sessions. The terminal should print: 
Session 715093703 ready (884 units)
Session 750749662 ready (761 units)

3. Now the notebooks in the practice folder are ready to run! (The data access ipynb can be skipped)