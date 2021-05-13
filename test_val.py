"""
Test on the PIPAL validation dataset.
To get the specific validation scores, you can submit the results to the following challenge website:
https://competitions.codalab.org/competitions/28050#participate-submit_results

Date: 2021/5/7
"""

from argparse import ArgumentParser
import torch
from scipy import stats
from torch import nn
import torch.nn.functional as F
from PIL import Image
from main import RandomCropPatches, NonOverlappingCropPatches
import numpy as np
from model.WResNet import *
# from model.RADN import *
import h5py
import os
import time

if __name__ == "__main__":
    parser = ArgumentParser(description='Test on the PIPAL validation dataset')
    parser.add_argument("--dist_dir", type=str, default='/mnt/data/ssw/PIPAL/Val_Distort',
                        help="distorted images dir.")
    parser.add_argument("--ref_dir", type=str, default='/mnt/data/ssw/PIPAL/Val_Ref',
                        help="reference image path.")
    parser.add_argument("--model_file", type=str, default='checkpoints/WResNet-lr=0.0001-bs=2',
                        help="model file path.")
    parser.add_argument("--model", type=str, default='WResNet')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'WResNet':
        model = WResNet().to(device)
    elif args.model == 'RADN':
        model = RADN().to(device)
    print(args.model)
    model.load_state_dict(torch.load(args.model_file), False)
    print('Model {} loaded！'.format(args.model_file))

    # get the file list of the distorted images
    l = os.listdir(args.dist_dir)
    l.sort()

    model.eval()
    scores = []
    f = open('./results/output.txt', 'w+')
    with torch.no_grad():
        for i in range(len(l)):
            im_name = l[i]
            ref_name = im_name[:5] + im_name[-4:]   # get names of the reference images
            # print(im_name, ref_name)
            print('{} / {}'.format(i, len(l)))

            im = Image.open(os.path.join(args.dist_dir, im_name)).convert('RGB')
            ref = Image.open(os.path.join(args.ref_dir, ref_name)).convert('RGB')
            data = NonOverlappingCropPatches(im, ref)

            dist_patches = data[0].unsqueeze(0).to(device)
            ref_patches = data[1].unsqueeze(0).to(device)
            t1 = time.time()
            score = model((dist_patches, ref_patches))
            using_time = time.time()-t1

            # print and output the results
            res = '{},{}'.format(im_name, score.item())
            print(res)
            print('using time:', using_time)
            f.write(res)
            if i != len(l)-1:
                f.write('\n')

    f.close()
