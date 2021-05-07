"""
We split the PIPAL training dataset into 2 parts for training and validation, respectively.
Codes in this file are used for testing on the validation part of the PIPAL training dataset.

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


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch test on the validation part of the PIPAL training dataset')
    parser.add_argument("--dist_dir", type=str, default=None,
                        help="distorted images dir.")
    parser.add_argument("--ref_dir", type=str, default=None,
                        help="reference image path.")
    parser.add_argument("--names_info", type=str, default=None,
                        help=".mat file that includes image names in the dataset.")
    parser.add_argument("--model_file", type=str, default='checkpoints/WResNet-lr=0.0001-bs=2-1.6360')
    parser.add_argument("--save_path", type=str, default='score/live2tid')
    parser.add_argument("--modelname", type=str, default='WResNet')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.modelname)
    if args.modelname == 'WResNet':
        model = WResNet().to(device)
    elif args.modelname == 'RADN':
        model = RADN().to(device)
    model.load_state_dict(torch.load(args.model_file), False)

    Info = h5py.File(args.names_info, 'r')
    index = Info['index']
    index = index[:, 0]
    ref_ids = Info['ref_ids'][0, :]
    K = 10
    k = 10
    testindex = index[int((k-1)/K * len(index)):int((k)/K * len(index))]
    test_index = []
    for i in range(len(ref_ids)):
        if ref_ids[i] in testindex:
            test_index.append(i)
    scale = Info['subjective_scores'][0, :].max()
    mos = Info['subjective_scores'][0, test_index]/scale  #
    im_names = [Info[Info['image_name'][0, :][i]][()].tobytes()[::2].decode() for i in test_index]
    ref_names = [Info[Info['ref_names'][0, :][i]][()].tobytes()\
                        [::2].decode() for i in (ref_ids[test_index]-1).astype(int)]

    #print(im_names)
    model.eval()
    scores = []   
    with torch.no_grad():
        for i in range(len(im_names)):
            print('{} / {}'.format(i, len(im_names)))
            im = Image.open(os.path.join(args.dist_dir, im_names[i])).convert('RGB')
            ref = Image.open(os.path.join(args.ref_dir, ref_names[i])).convert('RGB')
            data = NonOverlappingCropPatches(im, ref)
            
            dist_patches = data[0].unsqueeze(0).to(device)
            ref_patches = data[1].unsqueeze(0).to(device)
            score = model((dist_patches, ref_patches))
            print('{},{}'.format(im_names[i], score.item()))
            scores.append(score.item())
    srocc = stats.spearmanr(mos, scores)[0]
    plcc = stats.pearsonr(mos, scores)[0]
    print("Test Results - SROCC: {:.4f} PLCC: {:.4f}".format(srocc, plcc))
    np.save(args.save_path, (srocc, plcc))