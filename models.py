import torch
import torch.nn as nn


class SST(nn.Module):
    """
    Container module with 1D convolutions to generate proposals
    """

    def __init__(self,
                 video_dim=500,
                 hidden_dim=512,
                 dropout=0,
                 W=64,
                 K=64,
                 C=200,
                 rnn_type='GRU',
                 rnn_num_layers=2,
                 rnn_dropout=0.2,
                 ):
        super(SST, self).__init__()
        self.rnn = getattr(nn, rnn_type)(video_dim, hidden_dim, rnn_num_layers, batch_first=True, dropout=rnn_dropout)
        self.proposals_scores = torch.nn.Linear(hidden_dim, K)
        self.activity_scores = torch.nn.Linear(hidden_dim, C)

        # Saving arguments
        self.video_dim = video_dim
        self.W = W
        self.rnn_type = rnn_type
        self.rnn_num_layers = rnn_num_layers
        self.rnn_dropout = rnn_dropout
        self.K = K

    def eval(self):
        self.rnn.dropout = 0

    def train(self):
        self.rnn.dropout = self.rnn_dropout

    def forward(self, features):
        N, T, _ = features.size()

        rnn_output, _ = self.rnn(features)
        rnn_output = rnn_output.contiguous()
        rnn_output = rnn_output.view(rnn_output.size(0) * rnn_output.size(1), rnn_output.size(2))
        proposals_outputs = torch.sigmoid(self.proposals_scores(rnn_output))
        activity_outputs = self.activity_scores(rnn_output)
        return proposals_outputs.view(N, T, self.K), activity_outputs.view(N, T, self.C)

    def compute_slow_softmax_loss(self, activity_scores, activity_labels):
        N, W, C = activity_scores.size()
        # todo: add masking
        activity_labels = torch.autograd.Variable(activity_labels)
        criterion = torch.nn.CrossEntropyLoss(size_average=False)
        loss = torch.autograd.Variable(torch.zeros(N))
        nb_examples = 0
        for i in range(N):
            indexes = activity_labels[i] != -1
            labels = activity_labels[indexes]
            scores = activity_scores[i][indexes, :].view(-1, C)
            loss[i] = criterion(scores, labels)
            nb_examples += indexes.size()[0]
        return loss.sum() / nb_examples

    def compute_softmax_loss(self, activity_scores, activity_labels):
        N, W, C = activity_scores.size()
        # todo: add masking
        activity_labels = torch.autograd.Variable(activity_labels).view(N * W)
        activity_scores = activity_scores.view(N * W, C)
        indexes = activity_labels != -1
        labels = activity_labels[indexes]
        scores = activity_scores[indexes, :]
        criterion = torch.nn.CrossEntropyLoss(size_average=False)
        loss = criterion(scores, labels)
        return loss / indexes.size()[0]

    def compute_loss_with_BCE(self, outputs, masks, labels, w1):
        """
        Uses weighted BCE to calculate loss
        """
        w1 = torch.FloatTensor(w1).type_as(outputs.data)
        w0 = 1. - w1
        N, W, K = labels.size()
        labels = labels.mul(masks)
        weights = labels.mul(w0.expand(labels.size())) + (1. - labels).mul(w1.expand(labels.size()))
        weights = weights.view(-1)
        labels = torch.autograd.Variable(labels.view(-1))
        masks = torch.autograd.Variable(masks)
        outputs = outputs.mul(masks).view(-1)
        criterion = torch.nn.BCELoss(weight=weights, size_average=False)
        loss = criterion(outputs, labels) / (N * W)
        return loss

    def compute_loss(self, outputs, masks, labels):
        """
        Our implementation of weighted BCE loss.
        """
        labels = labels.view(-1)
        masks = masks.view(-1)
        outputs = outputs.view(-1)

        # Generate the weights
        ones = torch.sum(labels)
        total = labels.nelement()
        weights = torch.FloatTensor(outputs.size()).type_as(outputs.data)
        weights[labels.long() == 1] = 1.0 - ones / total
        weights[labels.long() == 0] = ones / total
        weights = weights.view(weights.size(0), 1).expand(weights.size(0), 2)

        # Generate the log outputs
        outputs = outputs.clamp(min=1e-8)
        log_outputs = torch.log(outputs)
        neg_outputs = 1.0 - outputs
        neg_outputs = neg_outputs.clamp(min=1e-8)
        neg_log_outputs = torch.log(neg_outputs)
        all_outputs = torch.cat((log_outputs.view(-1, 1), neg_log_outputs.view(-1, 1)), 1)

        all_values = all_outputs.mul(torch.autograd.Variable(weights))
        all_labels = torch.autograd.Variable(torch.cat((labels.view(-1, 1), (1.0 - labels).view(-1, 1)), 1))
        all_masks = torch.autograd.Variable(torch.cat((masks.view(-1, 1), masks.view(-1, 1)), 1))
        loss = -torch.sum(all_values.mul(all_labels).mul(all_masks)) / outputs.size(0)
        return loss

    def slow_compute_loss(self, outputs, masks, labels):
        """
        Used mainly for checking the actual loss function
        """
        labels = torch.autograd.Variable(labels)
        masks = torch.autograd.Variable(masks)
        outputs = outputs.view(-1, self.W, self.K)
        loss = 0.0
        print outputs.size()
        for n in range(outputs.size(0)):
            for t in range(self.W):
                w1 = torch.sum(outputs[n, t, :]) / outputs.numel()
                w0 = 1.0 - w1
                for j in range(self.K):
                    loss -= w1 * labels[n, t, j] * torch.log(outputs[n, t, j])
                    loss -= w0 * (1.0 - labels[n, t, j]) * torch.log(1.0 - outputs[n, t, j])
            print n, loss
        return loss
