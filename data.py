from torch.utils.data import Dataset

import h5py
import json
import numpy as np
import os
import progressbar
import torch

class ProposalDataset(object):
    """
    All dataset parsing classes will inherit from this class.
    """

    def __init__(self, args):
        """
        args must contain the following:
            data - the file that contains the Activity Net json data.
            features - the location of where the PCA C3D 500D features are.
        """
        assert os.path.exists(args.data)
        assert os.path.exists(args.features)
        self.data = json.load(open(args.data))
        self.features = h5py.File(args.features)
        if not os.path.exists(args.labels):
            self.generate_labels(args)
        self.labels = h5py.File(args.labels)

    def generate_labels(self, args):
        """
        Overwrite based on dataset used
        """
        pass

    def iou(self, interval, featstamps, return_index=False):
        """
        Measures temporal IoU
        """
        start_i, end_i = interval[0], interval[1]
        output = 0.0
        gt_index = -1
        for i, timestamp in enumerate(featstamps):
            start, end = timestamp
            intersection = max(0, min(end, end_i) - max(start, start_i))
            union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
            overlap = float(intersection) / (union + 1e-8)
            if overlap >= output:
                output = overlap
                gt_index = i
        if return_index:
            return output, gt_index
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


class ActivityNet(ProposalDataset):
    """
    ActivityNet is responsible for parsing the raw activity net dataset and converting it into a
    format that DataSplit (defined below) can use. This level of abstraction is used so that
    DataSplit can be used with other dataset and we would only need to write a class similar
    to this one.
    """

    def __init__(self, args):
        super(self.__class__, self).__init__(args)
        self.durations = {}
        self.gt_times = {}
        for split in ['training', 'validation', 'testing']:
            setattr(self, split + '_ids', [])
        for video_id in self.data['database']:
            split = self.data['database'][video_id]['subset']
            getattr(self, split + '_ids').append(video_id)
            self.durations[video_id] = self.data['database'][video_id]['duration']
            self.gt_times[video_id] = [ann['segment'] for ann in self.data['database'][video_id]['annotations']]

    def generate_labels(self, args):
        """
        Overwriting parent class to generate action proposal labels
        """
        print "| Generating labels for action proposals"
        label_dataset = h5py.File(args.labels, 'w')
        bar = progressbar.ProgressBar(maxval=len(self.data['database'].keys())).start()
        for progress, video_id in enumerate(self.data['database']):
            features = self.features['v_' + video_id]['c3d_features']
            nfeats = features.shape[0]
            duration = self.data['database'][video_id]['duration']
            annotations = self.data['database'][video_id]['annotations']
            timestamps = [ann['segment'] for ann in annotations]
            featstamps = [self.timestamp_to_featstamp(x, nfeats, duration) for x in timestamps]

            labels = np.zeros((nfeats, args.K))
            for t in range(nfeats):
                for k in xrange(args.K):
                    if self.iou([t-k, t+1], featstamps) >= args.iou_threshold:
                        labels[t, k] = 1

            video_dataset = label_dataset.create_dataset(video_id, (nfeats, args.K), dtype='f')
            video_dataset[...] = labels
            bar.update(progress)
        bar.finish()


class DataSplit(Dataset):

    def __init__(self, video_ids, dataset, args):
        """
        video_ids - list of video ids in the split
        features - the h5py file that contain all the C3D features for all the videos
        labels - the h5py file that contain all the proposals labels (0 or 1 per time step)
        args.W - the size of the window (the number of RNN steps to use)
        args.K - The number of proposals per time step
        args.max_W - the maximum number of windows to pass to back
        args.num_samples (optional) - contains how many of the videos in the list to use
        """
        self.video_ids = video_ids
        self.features = dataset.features
        self.labels = dataset.labels
        self.durations = dataset.durations
        self.gt_times = dataset.gt_times
        self.num_samples = args.num_samples
        self.W = args.W
        self.K = args.K
        self.max_W = args.max_W

        # Precompute masks
        self.masks = np.zeros((self.max_W, self.W, self.K))
        for index in range(self.W):
            self.masks[:, index, :min(self.K, index)] = 1
        self.masks = torch.FloatTensor(self.masks)

    def __getitem__(self, index):
        """
        To be overwritten by TrainSplit versus EvaluateSplit defined below.
        """
        pass

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        return len(self.video_ids)

class TrainSplit(DataSplit):

    def __init__(self, video_ids, dataset, args):
        super(self.__class__, self).__init__(video_ids, dataset, args)

    def collate_fn(self, data):
        """
        This function will be used by the DataLoader to concatenate outputs from
        multiple called to __get__item(). It will concatanate the windows along
        the first dimension
        """
        features = [d[0] for d in data]
        masks = [d[1] for d in data]
        labels = [d[2] for d in data]
        return torch.cat(features, 0), torch.cat(masks, 0), torch.cat(labels, 0)

    def __getitem__(self, index):
        # Now let's get the video_id
        video_id = self.video_ids[index]
        features = self.features['v_' + video_id]['c3d_features']
        labels = self.labels[video_id]
        nfeats = features.shape[0]
        nWindows = max(1, nfeats - self.W + 1)

        # Let's sample the maximum number of windows we can pass back.
        sample = range(nWindows)
        if self.max_W < nWindows:
            sample = np.random.choice(nWindows, self.max_W)
            nWindows = self.max_W

        # Create the outputs
        masks = self.masks[:nWindows, :, :]
        feature_windows = np.zeros((nWindows, self.W, features.shape[1]))
        label_windows = np.zeros((nWindows, self.W, self.K))
        for j, w_start in enumerate(sample):
            w_end = min(w_start + self.W, nfeats)
            feature_windows[j, 0:w_end-w_start, :] = features[w_start:w_end, :]
            label_windows[j, 0:w_end-w_start, :] = labels[w_start:w_end, :]

        return torch.FloatTensor(feature_windows), masks, torch.Tensor(label_windows)

class EvaluateSplit(DataSplit):

    def __init__(self, video_ids, dataset, args):
        super(self.__class__, self).__init__(video_ids, dataset, args)

    def collate_fn(self, data):
        """
        This function will be used by the DataLoader to concatenate outputs from
        multiple called to __get__item(). It will concatanate the windows along
        the first dimension
        """
        features = [d[0] for d in data]
        gt_times = [d[1] for d in data]
        durations = [d[2] for d in data]
        assert(len(features) == 1)
        assert(len(gt_times) == 1)
        assert(len(durations) == 1)
        return torch.cat(features, 0), gt_times[0], durations[0]

    def __getitem__(self, index):
        # Let's get the video_id and the features and labels
        video_id = self.video_ids[index]
        features = self.features['v_' + video_id]['c3d_features']
        duration = self.durations[video_id]
        gt_times = self.gt_times[video_id]

        return torch.FloatTensor(features), gt_times, duration
