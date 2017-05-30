import argparse
import random
import os
import subprocess

parser = argparse.ArgumentParser(description='Run the retrieval model with varying parameters.')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--nruns', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--model-dir', type=str, default='/data/chami/ActivityNet/models')
parser.add_argument('--iou-threshold', type=int, default=0.9)
parser.add_argument('--data', type=str, default='/data/chami/ActivityNet/activity_net.v1-3.min.json')
parser.add_argument('--labels', type=str, default='/data/chami/ActivityNet/labels.W256.K128.hdf5')
parser.add_argument('--features', type=str, default='/data/chami/ActivityNet/sub_activitynet_v1-3.c3d.hdf5')
parser.add_argument('--vid-ids', type=str, default='/data/chami/ActivityNet/video_ids.W256.K128.json')
args = parser.parse_args()

for _ in range(args.nruns):
    params = {
        'lr': '%.4f' % random.uniform(0.1, 10.0),
        'dropout': '%.1f' % random.uniform(0.0, 0.5),
        'weight-decay': '%.1f' % random.uniform(0.0001, 0.001),
    }
    model_name = os.path.join(args.model_dir, 'cap_' + '_'.join([k + str(params[k]) for k in params]))
    arguments = ' '.join(['--' + k + ' ' + str(params[k]) for k in params])
    train = 'CUDA_VISIBLE_DEVICES='  + args.gpu + ' python train.py --cuda --debug --save ' + model_name + ' --labels ' + args.labels + ' --data ' + args.data + ' --features ' + args.features + ' --vid-ids ' + str(args.vid_ids) + ' --epochs ' + str(args.epochs) + ' --batch-size 128 --max-W 1 --nthreads 1 --K 128 --W 256 --num-proposals 1000 --iou-threshold ' + str(args.iou_threshold) + ' --log-interval 50 --dropout ' + params['dropout'] + ' --lr ' + params['lr'] + ' --weight-decay ' + params['weight-decay']
    print '^'*89
    print train
    subprocess.call(train, shell=True)
