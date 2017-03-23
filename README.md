# SST
SST: Single-Stream Temporal Action Proposal

# Dependencies
1. pytorch
2. numpy
3. h5py

# Data
Currently, the code is setup to work with [ActivityNet](http://activity-net.org/). The raw [ActivityNet version 1.3](http://activity-net.org/download.html) must be downloaded as a json in the ```data/ActivityNet``` directory. Also, an hdf5 database with the PCA'ed 500 dimensional C3D features [availabel here](http://activity-net.org/challenges/2016/download.html#c3d). 

## Other datasets
If you want to use other datasets, write a new ```ProposalDataset``` class that is defined in ```data.py```. Follow the guidelines used in the ```ActivityNet``` class.

# Training

Run ```train.py```. 

```
arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of the data class to use from data.py
  --data DATA           location of the dataset
  --features FEATURES   location of the video features
  --save SAVE           path to folder where to save the final model and log
                        files and corpus
  --save-every SAVE_EVERY
                        Save the model every x epochs
  --clean               Delete the models and the log files in the folder
  --W W                 The rnn kernel size to use to get the proposal
                        features
  --K K                 Number of proposals
  --max-W MAX_W         maximum number of windows to return per video
  --iou-threshold IOU_THRESHOLD
                        threshold above which we say something is positive
  --rnn-type RNN_TYPE   type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --rnn-num-layers RNN_NUM_LAYERS
                        Number of layers in rnn
  --rnn-dropout RNN_DROPOUT
                        dropout used in rnn
  --video-dim VIDEO_DIM
                        dimensions of video (C3D) features
  --hidden-dim HIDDEN_DIM
                        dimensions output layer of video network
  --lr LR               initial learning rate
  --dropout DROPOUT     dropout between RNN layers
  --momentum MOMENTUM   SGD momentum
  --weight-decay WEIGHT_DECAY
                        SGD weight decay
  --epochs EPOCHS       upper epoch limit
  --batch-size BATCH_SIZE
                        batch size
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --debug               Print out debug sentences
  --num-samples NUM_SAMPLES
                        Number of training samples to train with
  --shuffle SHUFFLE     whether to shuffle the data
  --nthreads NTHREADS   number of worker threas used to load data
  --resume              reload the model
```
