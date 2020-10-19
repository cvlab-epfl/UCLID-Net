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

from models import image2cam
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
                    default='data/newly_inferred_camera', type=str)
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

dataset_train = dataset.Image_AllViews_intrinsic(split=train_split)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1,
                                                shuffle=True, num_workers=int(opt.workers),
                                                pin_memory=True)

dataset_test = dataset.Image_AllViews_intrinsic(split=test_split)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                              shuffle=False, num_workers=int(opt.workers),
                                              pin_memory=True)

print(f'Training set has {len(dataset_train)} samples.')
print(f'Testing set has {len(dataset_test)} samples.')
# ========================================================== #

# ===================CREATE network================================= #
network = image2cam.image2cam()

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

      images, intrinsics, mesh_name = data
      intrinsics = intrinsics[0]
      mesh_name = mesh_name[0]
      images = images[0].cuda()
      B_size = images.shape[0]

      # Forward pass
      R_pred, t_pred = network(images)

      # Assemble extrinsics
      extrinsics = torch.zeros(B_size, 3, 4).cuda()
      extrinsics[:,0:3,0:3] = R_pred
      extrinsics[:,0:3,3] = t_pred.squeeze(1)

      # Output generated R and t
      dir_name = os.path.join(opt.output_dir, mesh_name[9:], 'easy')
      if not os.path.exists(dir_name):
        os.makedirs(dir_name)
      file_name = os.path.join(dir_name, 'predicted_cameras.npz')
      np.savez(file_name, intrinsic=intrinsics.cpu().numpy(),
                          extrinsic=extrinsics.cpu().numpy(),
                          R=R_pred.cpu().numpy(),
                          t=t_pred.cpu().numpy())


  pbar.close()

print('Infer cameras for the whole training set:...')
inference(dataloader_train, len(dataset_train))
print('Done!')

print('Infer cameras for the whole testing set:...')
inference(dataloader_test, len(dataset_test))
print('Done!')
