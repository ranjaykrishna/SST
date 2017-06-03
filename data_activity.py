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
        labels = h5py.File(args.labels)
        self.proposals_labels = labels['proposals']
        self.activity_labels = labels['activity']
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
        self.gt_activities = {}
        self.w1 = self.vid_ids['w1']
        self.labels_to_idx = self.vid_ids['labels_to_idx']
        for split in ['training', 'validation', 'testing']:
            setattr(self, split + '_ids', self.vid_ids[split])
            for video_id in self.vid_ids[split]:
                self.durations[video_id] = self.data['database'][video_id]['duration']
                self.gt_times[video_id] = [ann['segment'] for ann in self.data['database'][video_id]['annotations']]
                self.gt_activities[video_id] = [self.labels_to_idx[ann['segment']] for ann in self.data['database'][video_id]['annotations']]

    def generate_labels(self, args):
        """
        Overwriting parent class to generate action proposal labels
        """
        print "| Generating labels for action proposals"
        label_dataset = h5py.File(args.labels, 'w')
        video_ids = self.data['database'].keys()
        num_videos = len(video_ids)
        proposals_group = label_dataset.create_group('proposals')
        activity_group = label_dataset.create_group('activity')
        bar = progressbar.ProgressBar(maxval=len(self.data['database'].keys())).start()
        split_ids = {'training': [], 'validation': [], 'testing': []}
        labels_to_idx = {}
        prop_pos_examples = []  # used to compute w1 in the loss
        counter = 0  # counter to keep track of the activity labels
        for progress, video_id in enumerate(video_ids):
            features = self.features['v_' + video_id]['c3d_features']
            nfeats = features.shape[0]
            duration = self.data['database'][video_id]['duration']
            annotations = self.data['database'][video_id]['annotations']
            timestamps = [ann['segment'] for ann in annotations]
            activity_annotations = [ann['label'] for ann in annotations]
            featstamps = [self.timestamp_to_featstamp(x, nfeats, duration) for x in timestamps]
            split = self.data['database'][video_id]['subset']
            for start, end in featstamps:
                if split == 'training' and (end - start) > args.K / args.iou_threshold:
                    # go to the next video since this one has proposals that would not be captured
                    continue
            # we keep track of the videos kept to update ids
            split_ids[split] += [video_id]
            proposals_labels = np.zeros((nfeats, args.K))
            activity_labels = -1 * np.ones(nfeats)  # we use -1 to recognize timesteps with no activities
            for t in range(nfeats):
                for k in xrange(args.K):
                    iou, gt_index = self.iou([t - k, t + 1], featstamps, return_index=True)
                    if iou >= args.iou_threshold:
                        proposals_labels[t, k] = 1.
                        label_ann = activity_annotations[gt_index]
                        if label_ann not in labels_to_idx.keys():
                            labels_to_idx[label_ann] = counter
                            counter += 1
                        activity_labels[t] = labels_to_idx[label_ann]
            video_prop_dataset = proposals_group.create_dataset(video_id, (nfeats, args.K), dtype='f')
            video_activity_dataset = activity_group.create_dataset(video_id, (nfeats,), dtype='f')
            video_prop_dataset[...] = proposals_labels
            video_activity_dataset[...] = activity_labels
            bar.update(progress)
            if self.data['database'][video_id]['subset'] == 'training':
                prop_pos_examples += [np.mean(proposals_labels, axis=0)]
        split_ids['w1'] = np.array(prop_pos_examples).mean(axis=0).tolist()  # this will be used to compute the loss
        split_ids['labels_to_idx'] = labels_to_idx
        json.dump(split_ids, open(args.vid_ids, 'w'))
        bar.finish()
        print "generated activity labels with {} classes\n".format(len(labels_to_idx))


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
        self.proposals_labels = dataset.proposals_labels
        self.activity_labels = dataset.activity_labels
        self.durations = dataset.durations
        self.gt_times = dataset.gt_times
        self.num_samples = args.num_samples
        self.W = args.W
        self.K = args.K
        self.max_W = args.max_W

        # Precompute masks
        self.masks = np.zeros((self.W, self.K))
        for index in range(self.W):
            self.masks[index, :min(self.K, index)] = 1
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
        multiple called to __get__item(). It will concatenate the windows along
        the first dimension
        """
        features = data[0][0]
        masks = data[0][1]
        proposals_labels = data[0][2]
        activity_labels = data[0][3]
        return features.view(1, features.size(0), features.size(1)), \
               masks.view(1, masks.size(0), masks.size(1)), \
               proposals_labels.view(1, proposals_labels.size(0), proposals_labels.size()[1]), \
               activity_labels.view(1, activity_labels.size()[0])

    def __getitem__(self, index):
        # Now let's get the video_id
        video_id = self.video_ids[index]
        features = self.features['v_' + video_id]['c3d_features']
        proposals_labels = self.proposals_labels[video_id]
        activity_labels = self.activity_labels[video_id]
        nfeats = features.shape[0]
        nWindows = max(1, 1 + nfeats - self.W)
        start = np.random.choice(nWindows)-1
        end = min(nfeats, start + self.W)
        masks = self.masks
        masks[min(nfeats, self.W):, :] = 0
        feature_windows = np.zeros((self.W, features.shape[1]))
        proposals_labels_windows = np.zeros((self.W, self.K))
        activity_labels_windows = np.zeros(self.W)
        proposals_labels_windows[start:end, :] = proposals_labels[start:end, :]
        activity_labels_windows[start:end] = activity_labels[start:end]
        feature_windows[start:end, :] = features[start:end, :]
        return torch.FloatTensor(feature_windows), masks, torch.Tensor(proposals_labels_windows), torch.Tensor(
            activity_labels_windows)


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
        proposals_labels = data[0][3]
        activities_labels = data[0][4]
        return features.view(1, features.size(0), features.size(1)), \
               gt_times, durations, proposals_labels.view(1, proposals_labels.size(0), proposals_labels.size()[1]),  \
               activities_labels

    def __getitem__(self, index):
        # Let's get the video_id and the features and labels
        video_id = self.video_ids[index]
        features = self.features['v_' + video_id]['c3d_features']
        duration = self.durations[video_id]
        gt_times = self.gt_times[video_id]
        activities_labels = self.gt_activities[video_id]
        proposals_labels = self.proposals_labels[video_id]
        return torch.FloatTensor(features), gt_times, duration, torch.Tensor(np.array(proposals_labels)), activities_labels
