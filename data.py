import json
import os

import h5py
import numpy as np
import progressbar
import torch
from torch.utils.data import Dataset


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
        if not os.path.exists(args.labels) or not os.path.exists(args.vid_ids):
            self.generate_labels(args)
        self.labels = h5py.File(args.labels)
        self.vid_ids = json.load(open(args.vid_ids))

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
        for i, (start, end) in enumerate(featstamps):
            intersection = max(0, min(end, end_i) - max(start, start_i))
            union = min(max(end, end_i) - min(start, start_i), end - start + end_i - start_i)
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
        start = min(int(round(start / duration * nfeats)), nfeats - 1)
        end = max(int(round(end / duration * nfeats)), start + 1)
        return start, end

    def compute_proposals_stats(self, prop_captured):
        """
        Function to compute the proportion of proposals captured during labels generation.
        :param prop_captured: array of length nb_videos
        :return:
        """
        nb_videos = len(prop_captured)
        proportion = np.mean(prop_captured[prop_captured != -1])
        nb_no_proposals = (prop_captured == -1).sum()
        print "Number of videos in the dataset: {}".format(nb_videos)
        print "Proportion of videos with no proposals: {}".format(1. * nb_no_proposals / nb_videos)
        print "Proportion of action proposals captured during labels creation: {}".format(proportion)


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
        self.w1 = self.vid_ids['w1']
        for split in ['training', 'validation', 'testing']:
            setattr(self, split + '_ids', self.vid_ids[split])
            for video_id in self.vid_ids[split]:
                self.durations[video_id] = self.data['database'][video_id]['duration']
                self.gt_times[video_id] = [ann['segment'] for ann in self.data['database'][video_id]['annotations']]

    def generate_labels(self, args):
        """
        Overwriting parent class to generate action proposal labels
        """
        print "| Generating labels for action proposals"
        label_dataset = h5py.File(args.labels, 'w')
        bar = progressbar.ProgressBar(maxval=len(self.data['database'].keys())).start()
        prop_captured = []
        prop_pos_examples = []
        video_ids = self.data['database'].keys()
        split_ids = {'training': [], 'validation': [], 'testing': [],
                     'w1': []}  # maybe find a better name since w1 is not a split
        for progress, video_id in enumerate(video_ids):
            features = self.features['v_' + video_id]['c3d_features']
            nfeats = features.shape[0]
            duration = self.data['database'][video_id]['duration']
            annotations = self.data['database'][video_id]['annotations']
            timestamps = [ann['segment'] for ann in annotations]
            featstamps = [self.timestamp_to_featstamp(x, nfeats, duration) for x in timestamps]
            nb_prop = len(featstamps)
            for i in range(nb_prop):
                if (featstamps[nb_prop - i - 1][1] - featstamps[nb_prop - i - 1][0]) > args.K / args.iou_threshold:
                    # we discard these proposals since they will not be captured for this value of K 
                    del featstamps[nb_prop - i - 1]
            if len(featstamps) == 0:
                if len(timestamps) == 0:
                    # no proposals il this video
                    prop_captured += [-1.]
                else:
                    # no proposals captured in this video since all have a length above threshold
                    prop_captured += [0.]
                continue
                # we keep track of the videos kept to update ids
            split_ids[self.data['database'][video_id]['subset']] += [video_id]
            labels = np.zeros((nfeats, args.K))
            gt_captured = []
            for t in range(nfeats):
                for k in xrange(args.K):
                    iou, gt_index = self.iou([t - k, t + 1], featstamps, return_index=True)
                    if iou >= args.iou_threshold:
                        labels[t, k] = 1
                        gt_captured += [gt_index]
            prop_captured += [1. * len(np.unique(gt_captured)) / len(timestamps)]
            if self.data['database'][video_id]['subset'] == 'training':
                prop_pos_examples += [np.sum(labels, axis=0) * 1. / nfeats]
            video_dataset = label_dataset.create_dataset(video_id, (nfeats, args.K), dtype='f')
            video_dataset[...] = labels
            bar.update(progress)
        split_ids['w1'] = np.array(prop_pos_examples).mean(axis=0).tolist()  # this will be used to compute the loss
        json.dump(split_ids, open(args.vid_ids, 'w'))
        self.compute_proposals_stats(np.array(prop_captured))
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
            # in case num sample is greater than the dataset itself
            return min(self.num_samples, len(self.video_ids))
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
        if nfeats < self.W:
            masks = self.masks[:1, :, :]
            masks[:1, nfeats:, :] = 0
            feature_windows = np.zeros((1, self.W, features.shape[1]))
            label_windows = np.zeros((1, self.W, self.K))
            label_windows[0, :nfeats, :] = labels
            feature_windows[0, :nfeats, :] = features
        else:
            # Let's sample the maximum number of windows we can pass back.
            nWindows = max(1, nfeats - self.W + 1)
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
                feature_windows[j, 0:w_end - w_start, :] = features[w_start:w_end, :]
                label_windows[j, 0:w_end - w_start, :] = labels[w_start:w_end, :]
            # if label_windows[j].sum() == 0:
        # check to see how often trainin examples have all 0 labels
        #	print "No proposals!!"
        # code to sample proposals avoiding all 0 situations
        # k = 0
        # while k<=50:
        #	k += 1
        #	sample = np.random.choice(nWindows, self.max_W)
        #	nWindows = 1
        #	masks = self.masks[:nWindows, :, :]
        #	feature_windows = np.zeros((nWindows, self.W, features.shape[1]))
        #	label_windows = np.zeros((nWindows, self.W, self.K))
        #	for j, w_start in enumerate(sample):
        #		w_end = min(w_start + self.W, nfeats)
        #		feature_windows[j, 0:w_end-w_start, :] = features[w_start:w_end, :]
        #		label_windows[j, 0:w_end-w_start, :] = labels[w_start:w_end, :]
        #		if label_windows.sum()!=0:
        #			return torch.FloatTensor(feature_windows), masks, torch.Tensor(label_windows)
        # print "No labels!!!"
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
        features = data[0][0]
        gt_times = data[0][1]
        durations = data[0][2]
        labels = data[0][3]
        masks = data[0][4]
        return features.view(1, features.size(0), features.size(1)), gt_times, durations, labels, masks#, torch.cat(labels, 0) 

    def __getitem__(self, index):
        # Let's get the video_id and the features and labels
        video_id = self.video_ids[index]
        features = self.features['v_' + video_id]['c3d_features']
        duration = self.durations[video_id]
        gt_times = self.gt_times[video_id]
        labels = self.labels[video_id]
        W, K = labels.shape
        masks = np.zeros((1, W, K))
        for index in range(W):
            masks[:, index, :min(K, index)] = 1
        masks = torch.FloatTensor(masks)
        return torch.FloatTensor(features), gt_times, duration, torch.Tensor(np.array(labels)), masks
