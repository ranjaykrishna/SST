import argparse
import h5py
import json
import os

parser = argparse.ArgumentParser(description='video features to LSTM Language Model')
parser.add_argument('--data', type=str, default='data/ActivityNet/activity_net.v1-3.min.json',
                    help='location of the dataset')
parser.add_argument('--features', type=str, default='data/ActivityNet/sub_activitynet_v1-3.c3d.hdf5',
                    help='location of the video features')
parser.add_argument('--labels', type=str, default='data/ActivityNet/labels.hdf5',
                    help='location of the proposal labels')
args = parser.parse_args()

assert os.path.exists(args.data)
assert os.path.exists(args.features)
data = json.load(open(args.data))
features = h5py.File(args.features)
labels = h5py.File(args.labels)

for video_id in data['database']:
    f = features['v_' + video_id]['c3d_features']
    l = labels[video_id]
    nf = f.shape[0]
    nl = l.shape[0]
    assert(nf > 0)
    assert(nf == nl)
