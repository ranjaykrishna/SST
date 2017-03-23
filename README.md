# SST
SST: Single-Stream Temporal Action Proposal

# Dependencies
1. pytorch
2. numpy
3. h5py

# Data
Currently, the code is setup to work with [ActivityNet](http://activity-net.org/). The raw [ActivityNet version 1.3](http://activity-net.org/download.html) must downloaded as a json in the data directory. Also, an hdf5 database with the PCA'ed 500 dimensional C3D features [availabel here](http://activity-net.org/challenges/2016/download.html#c3d). 

## Other datasets
If you want to use other datasets, write a new ```ProposalDataset``` class that is defined in ```data.py```. Follow the guidelines used in the ```ActivityNet``` class.

# Training

Run ```python train.py --help```. That should print out all the options along with explanations on how to train SST.
