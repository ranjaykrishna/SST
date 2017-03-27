from data import DataSplit
from torch.autograd import Variable
from torch.utils.data import DataLoader

import data
import argparse
import json
import models
import numpy as np
import os
import time
import torch
import torch.optim as optim

parser = argparse.ArgumentParser(description='video features to LSTM Language Model')

# Location of data
parser.add_argument('--dataset', type=str, default='ActivityNet',
                    help='Name of the data class to use from data.py')
parser.add_argument('--data', type=str, default='data/ActivityNet/activity_net.v1-3.min.json',
                    help='location of the dataset')
parser.add_argument('--features', type=str, default='data/ActivityNet/sub_activitynet_v1-3.c3d.hdf5',
                    help='location of the video features')
parser.add_argument('--labels', type=str, default='data/ActivityNet/labels.hdf5',
                    help='location of the proposal labels')
parser.add_argument('--save', type=str,  default='data/models/default',
                    help='path to folder where to save the final model and log files and corpus')
parser.add_argument('--save-every', type=int,  default=1,
                    help='Save the model every x epochs')
parser.add_argument('--clean', dest='clean', action='store_true',
                    help='Delete the models and the log files in the folder')
parser.add_argument('--W', type=int, default=128,
                    help='The rnn kernel size to use to get the proposal features')
parser.add_argument('--K', type=int, default=64,
                    help='Number of proposals')
parser.add_argument('--max-W', type=int, default=256,
                    help='maximum number of windows to return per video')
parser.add_argument('--iou-threshold', type=float, default=0.5,
                    help='threshold above which we say something is positive')

# Model options
parser.add_argument('--rnn-type', type=str, default='GRU',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--rnn-num-layers', type=int, default=2,
                    help='Number of layers in rnn')
parser.add_argument('--rnn-dropout', type=int, default=0.0,
                    help='dropout used in rnn')
parser.add_argument('--video-dim', type=int, default=500,
                    help='dimensions of video (C3D) features')
parser.add_argument('--hidden-dim', type=int, default=512,
                    help='dimensions output layer of video network')

# Training options
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout between RNN layers')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0,
                    help='SGD weight decay')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Print out debug sentences')
parser.add_argument('--num-samples', type=int, default=None,
                    help='Number of training samples to train with')
parser.add_argument('--shuffle', type=int, default=1,
                    help='whether to shuffle the data')
parser.add_argument('--nthreads', type=int, default=1,
                    help='number of worker threas used to load data')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='reload the model')
args = parser.parse_args()

# Ensure that the kernel for RNN is greated than the number of proposals
assert(args.W > args.K)

# Check if directory exists and create one if it doesn't:
if not os.path.isdir(args.save):
    os.makedirs(args.save)

# Argument hack
args.shuffle = args.shuffle != 0

# Clean the directory
if args.clean:
    for f in ['model.pth', 'train.log', 'val.log', 'test.log']:
        try:
            os.remove(os.path.join(args.save, f))
        except:
            continue

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
elif args.cuda:
    raise Exception("No GPU found, please run without --cuda")

# Save the arguments for future viewing
with open(os.path.join(args.save, 'args.json'), 'w') as f:
    f.write(json.dumps(vars(args)))

###############################################################################
# Load data
###############################################################################

print "| Loading data into corpus: %s" % args.data
dataset = getattr(data, args.dataset)(args)
train_dataset = DataSplit(dataset.training_ids, dataset.features, dataset.labels, args)
val_dataset = DataSplit(dataset.validation_ids, dataset.features, dataset.labels, args)
print "| Dataset created"
train_loader = DataLoader(train_dataset, shuffle=args.shuffle, batch_size=args.batch_size, num_workers=args.nthreads, collate_fn=train_dataset.collate_fn)
val_loader = DataLoader(val_dataset, shuffle=args.shuffle, batch_size=args.batch_size, num_workers=args.nthreads, collate_fn=val_dataset.collate_fn)
print "| Data Loaded: # training data: %d, # val data: %d" % (len(train_loader)*args.batch_size, len(val_loader)*args.batch_size)

###############################################################################
# Build the model
###############################################################################

if args.resume:
    model = torch.load(os.path.join(args.save, 'model.pth'))
else:
    model = models.SST(
        video_dim=args.video_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        W=args.W,
        K=args.K,
        rnn_type = args.rnn_type,
        rnn_num_layers = args.rnn_num_layers,
        rnn_dropout = args.rnn_dropout,
    )

if args.cuda:
    model.cuda()

###############################################################################
# Training code
###############################################################################

def calculate_stats(proposals, masks, labels, args):
    eps = 1e-8
    labels = labels.view(-1)
    masks = masks.view(-1)

    proposal = proposals.data
    proposal[proposal < args.iou_threshold] = 0
    proposal[proposal >= args.iou_threshold] = 1
    tp = torch.sum(proposal.mul(labels).mul(masks))
    fp = torch.sum(proposal.mul(1-labels).mul(masks))
    fn = torch.sum(labels.mul(1-proposal).mul(masks))
    total = torch.sum(masks)
    assert(tp+fn == labels.mul(masks).sum())
    if args.debug:
        print ">> # of true pos: %f, # of false pos: %f, # of false neg: %f" % (tp, fp, fn)
    return float(tp)/(total+eps), float(tp)/(tp+fp+eps), float(tp)/(tp+fn+eps)


def evaluate(data_loader, maximum=None):
    total = len(data_loader)*args.batch_size
    if maximum is not None:
        total = min(total, maximum)
    accs = np.zeros(total)
    recall = np.zeros(total)
    precision = np.zeros(total)
    for batch_idx, (features, masks, labels) in enumerate(data_loader):
        if maximum is not None and batch_idx >= maximum:
            break
        if args.cuda:
            features = features.cuda()
            masks = masks.cuda()
            labels = labels.cuda()
        features = Variable(features)
        proposals = model(features)
        accs[batch_idx], precision[batch_idx], recall[batch_idx] = calculate_stats(proposals, masks, labels, args)
    return np.mean(accs), np.mean(precision), np.mean(recall)

def train(epoch):
    total_loss = []
    model.train()
    start_time = time.time()
    for batch_idx, (features, masks, labels) in enumerate(train_loader):
        if args.cuda:
            features = features.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
        features = Variable(features)
        optimizer.zero_grad()
        proposals = model(features)
        loss = model.compute_loss(proposals, masks, labels)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.data[0])

        # Debugging training samples
        if args.debug:
            acc, precision, recall = evaluate(train_loader)
            log_entry = ('| accuracy: {:2.4f}\% | precision: {:2.4f}\% ' \
                '| recall: {:2.4f}\%'.format(acc, precision, recall))
            print log_entry

        # Print out training loss every interval in the batch
        if batch_idx % args.log_interval == 0:# and batch_idx > 0:
            cur_loss = total_loss[-1]
            elapsed = time.time() - start_time
            log_entry = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | ' \
                'loss {:5.6f}'.format(
                epoch, batch_idx, len(train_loader), args.lr,
                elapsed * 1000 / args.log_interval, cur_loss*1000)
            print log_entry
            with open(os.path.join(args.save, 'train.log'), 'a') as f:
                f.write(log_entry)
                f.write('\n')
            start_time = time.time()

# Loop over epochs.
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train(epoch)
    acc, precision, recall = evaluate(val_loader)
    print('-' * 89)
    log_entry = ('| end of epoch {:3d} | time: {:5.2f}s | val accuracy: {:2.2f}\% | val precision: {:2.2f}\% ' \
            '| val recall: {:2.2f}\%'.format(
        epoch, (time.time() - epoch_start_time), acc, precision, recall))
    print log_entry
    print('-' * 89)
    with open(os.path.join(args.save, 'val.log'), 'a') as f:
        f.write(log_entry)
        f.write('\n')
    if args.save != '' and epoch % args.save_every == 0 and epoch > 0:
        torch.save(model, os.path.join(args.save, 'model_' + str(epoch) + '.pth'))

# Run on test data and save the model.
print "| Testing model on test set"
test_dataset = DataSplit(dataset.testing_ids, dataset.features, dataset.labels, args)
test_loader = DataLoader(test_dataset, shuffle=args.shuffle, batch_size=args.batch_size, num_workers=args.nthreads, collate_fn=test_dataset.collate_fn)
test_acc, test_precision, test_recall = evaluate(test_loader)
print('=' * 89)
print('| End of training | test acc {:2.2f}\% | test precision {:2.2f}\% | test recall {:2.2f}\%'.format(
    test_acc, test_precision, test_recall))
print('=' * 89)
if args.save != '':
    torch.save(model, os.path.join(args.save, 'model.pth'))
