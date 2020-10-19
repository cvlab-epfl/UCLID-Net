import torch
import numpy as np
import os, sys
import torch.nn as nn
import torch.nn.functional as F
import argparse, time
import datetime
import random
import string
import json
import imageio

from models import image2depth
import dataloaders.auxiliary_nets as dataset
import tqdm

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--num_views', dest='num_views',
                      type=int, default=36,
                      help='number of views per instance (default=all views)')
parser.add_argument('--workers', dest='workers',
                    help='workers',
                    default=6, type=int)
parser.add_argument('--model', dest='model',
                    help='path to the trained model',
                    required=True, type=str)
parser.add_argument('--output_dir', dest='output_dir',
                    help='path to output directory',
                    default='data/newly_inferred_depth', type=str)
parser.add_argument('--train_split', type=str,
                    default = 'data/splits/cars_train.json',
                    help='training split')
parser.add_argument('--test_split', type=str,
                    default = 'data/splits/cars_test.json',
                    help='testing split')

opt = parser.parse_args()
print(opt)
# ========================================================== #

# ===================CREATE DATASET================================= #
with open(opt.train_split, "r") as f:
  train_split = json.load(f)
with open(opt.test_split, "r") as f:
  test_split = json.load(f)

dataset_train = dataset.Image_AllViews(split=train_split)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1,
                                                shuffle=True, num_workers=int(opt.workers),
                                                pin_memory=True)

dataset_test = dataset.Image_AllViews(split=test_split)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                              shuffle=False, num_workers=int(opt.workers),
                                              pin_memory=True)

print(f'Training set has {len(dataset_train)} samples.')
print(f'Testing set has {len(dataset_test)} samples.')
# ========================================================== #

# ===================CREATE network================================= #
network = image2depth.image2depth()

network = network.cuda()  # move network to GPU
# Load trained model
try:
  network.load_state_dict(torch.load(opt.model))
  print('Trained net weights loaded')
except:
  print('ERROR: Failed to load net weights')
  exit()
network.eval()
# ========================================================== #

# ===================MAIN LOOP================================= #
def inference(dataloader, length):
  pbar = tqdm.tqdm(total=length)
  with torch.no_grad():
    for data in dataloader:
      pbar.update()
      images, mesh_name = data
      mesh_name = mesh_name[0]
      images = images[0].cuda()

      ## Forward pass
      z_fake = network(images)

      # Output generated depth maps
      dir_name = os.path.join(opt.output_dir, mesh_name[9:], 'easy')

      if not os.path.exists(dir_name):
        os.makedirs(dir_name)
      for i in range(opt.num_views):
        file_name = os.path.join(dir_name, '{0:02d}'.format(i) + '.exr')
        # De-normalize and revert axis
        z_pred = dataset.Image_DepthMaps.f_to_z(z_fake[i, 0]).cpu().numpy()
        imageio.imwrite(file_name, np.repeat(z_pred[...,None], 3, 2))
  pbar.close()

print('Infer depth maps for the whole training set:...')
inference(dataloader_train, len(dataset_train))
print('Done!')

print('Infer depth maps for the whole testing set:...')
inference(dataloader_test, len(dataset_test))
print('Done!')
