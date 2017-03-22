from torch.utils.data import Dataset

import h5py
import json
import numpy as np
import os
import torch

class ProposalDataset(object):
    """
    All dataset parsing classes will inherit from this class.
    """

    def __init(self, args):
        """
        args must contain the following:
            data - the file that contains the Activity Net json data.
            features - the location of where the PCA C3D 500D features are.
        """
        assert os.path.exists(args.data)
        assert os.path.exists(args.features)
        self.data = json.load(open(args.data))
        self.features = h5py.File(args.features)


class ActivityNet(ProposalDataset):
    """
    ActivityNet is responsible for parsing the raw activity net dataset and converting it into a
    format that DataSplit (defined below) can use. This level of abstraction is used so that
    DataSplit can be used with other dataset and we would only need to write a class similar
    to this one.
    """

    def __init__(self, args):
        super(self.__class__, self).__init__(args)
        for split in ['training', 'validation', 'testing']:
            setattr(self, split + '_ids', [])
            setattr(self, split + '_segments', {})
            setattr(self, split + '_durations', {})
        for video_id in self.data['database']:
            split = self.data['database'][video_id]['subset']
            annotations = self.data['database'][video_id]['annotations']
            segments = [ann['segment'] for ann in annotations]
            getattr(self, split + '_ids').append(video_id)
            getattr(self, split + '_segments')[video_id] = segments
            getattr(self, split + '_durations')[video_id] = self.data['database'][video_id]['duration']

class DataSplit(Dataset):

    def __init__(self, video_ids, segments, durations, features, args):
        """
        video_ids - list of video ids in the split
        segments - dictionary from video_id to list of start and end times of proposals
        durations - dictionary from video_id to the duration of video in seconds
        features - the h5py file that contain all the C3D features for
                   all the videos
        args.W - the size of the window (the number of RNN steps to use)
        args.K - The number of proposals per time step
        args.max_W - the maximum number of windows to pass to back
        args.num_samples (optional) - contains how many of the videos in the list to use
        """
        self.video_ids = video_ids
        self.segments = segments
        self.durations = durations
        self.features = features
        self.num_samples = args.num_samples
        self.W = args.W
        self.K = args.K
        self.strides = args.strides
        self.max_W = args.max_W

    def iou(self, interval, featstamps):
        """
        Measures temporal IoU
        """
        start_i, end_i = interval[0], interval[1]
        output = 0.0
        for start, end in featstamps:
            intersection = max(0, min(end, end_i) - max(start, start_i))
            union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
            overlap = float(intersection) / (union + 1e-8)
            if overlap >= output:
                output = overlap
        return output

    def timestamp_to_featstamp(self, timestamp, nfeats, duration):
        """
        Function to measure 1D overlap
        Convert the timestamps to feature indices
        """
        start, end = timestamp
        start = min(int(round(start / duration * nfeats)), nfeats-1)
        end = max(int(round(end /duration * nfeats)), start+1)
        return start, end

    def collate_fn(data):
        """
        This function will be used by the DataLoader to concatenate outputs from
        multiple called to __get__item(). It will concatanate the windows along
        the first dimension
        """
        return torch.stack(data, dim=0)

    def __getitem__(self, index):
        # Now let's get the video_id
        video_id = self.video_ids[index]
        features = self.corpus.features[video_id]['c3d_features']
        nfeats = features.shape[0]
        duration = self.durations[video_id]
        timestamps = self.segments[video_id]
        featstamps = [self.timestamp_to_featstamp(x, nfeats, duration) for x in timestamps]
        nWindows = max(1, nfeats - self.W + 1)
        unrolled = np.zeros((nWindows, self.W, features.shape[1]))
        masks = np.zeros((nWindows, self.W, self.K))
        labels = np.zeros((nWindows, self.W, self.K))
        for j, w_start in enumerate(nWindows):
            w_end = min(w_start + self.W, nfeats)

            for index in range(self.W): # This is the index of the window in window_space
                masks[w_start, index, :min(self.K, index)] = 1
                t = index + w_start # This is the actual time of the video index in feature_space
                if w_start == 0:
                    for k in xrange(self.K):
                        if self.iou([t-k, t+1], featstamps) >= self.iou_threshold:
                            labels[w_start, index, k] = 1
                else:
                    labels[w_start, index, :-1] = labels[w_start, index, 1:]
                    if self.iou([t-self.K, t+1], featstamps) >= self.iou_threshold:
                        labels[w_start, index, -1] = 1
            unrolled[j, 0:w_end-w_start, :] = features[w_start:w_end:1, :]

        # Let's sample the maximum number of windows we can pass back.
        nbatches = min(self.max_W, nWindows)
        sample = np.random.choice(nbatches, nWindows)
        unrolled = unrolled[sample, :, :]
        masks = masks[sample, :, :]
        labels = labels[sample, :, :]
        return torch.FloatTensor(unrolled), torch.Tensor(masks), torch.Tensor(labels)

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        return len(self.video_ids)
