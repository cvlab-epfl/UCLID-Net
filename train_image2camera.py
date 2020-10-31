import argparse
import random
import string
import datetime
import os
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from models import image2cam
import dataloaders.auxiliary_nets as dataset
from utils import AverageValueMeter

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=22,
      help='batch size for training')
parser.add_argument('--workers', type=int, default=12,
      help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=300,
      help='number of epochs to train for')
parser.add_argument('--model', type=str, default='',
      help='optional reload model path')
parser.add_argument('--train_split', type=str, default = 'data/splits/cars_train.json',
      help='training split')
parser.add_argument('--test_split', type=str, default = 'data/splits/cars_test.json',
      help='testing split')
parser.add_argument('--output_dir', type=str, default="output/",
      help='where to log outputs')
parser.add_argument('--experiment_name', type=str, default='',
      help='Used for creating output directory and Wandb experiment')

opt = parser.parse_args()
print(opt)
# ========================================================== #

# =============OUTPUTS and LOGS======================================== #
if opt.experiment_name == '':
  # Assign random name
  experiments_name = ''.join(random.choice(string.ascii_lowercase) for i in range(6))
else:
  experiments_name = opt.experiment_name
# Give a unique experiment name
experiments_name += datetime.datetime.now().isoformat(timespec='seconds')

# Initialize wandb logs if available
try:
  import wandb
  wandb.init(project='UCLID_Net', name='image2camera_' + experiments_name)
  WANDB_LOGS = True
except:
  print('wandb module not found, or uncorrectly initialized.')
  print('Training will not be logged to wandb')
  WANDB_LOGS = False

# Create output directory
output_folder = os.path.join(opt.output_dir, experiments_name)
print("saving logs in ", output_folder)
if not os.path.exists(output_folder):
  os.makedirs(output_folder)

logfile = os.path.join(output_folder, 'log.txt')
# ========================================================== #

# ===================CREATE DATASET================================= #
# Create train/test dataloader
with open(opt.train_split, "r") as f:
  train_split = json.load(f)
with open(opt.test_split, "r") as f:
  test_split = json.load(f)

dataset_train = dataset.Image_PointClouds(split=train_split,
                                        is_train=True)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size,
                                                shuffle=True, num_workers=int(opt.workers),
                                                pin_memory=True)

dataset_test = dataset.Image_PointClouds(split=test_split,
                                      is_train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=int(opt.workers),
                                              pin_memory=True)

print(f'Training set has {len(dataset_train)} samples.')
print(f'Testing set has {len(dataset_test)} samples.')
len_dataset = len(dataset_train)
# ========================================================== #

# ===================CREATE network================================= #
network = image2cam.image2cam()

network = network.cuda()  # move network to GPU
# If needed, load existing model
if opt.model != '':
  network.load_state_dict(torch.load(opt.model))
  print('Previous net weights loaded')
# ========================================================== #

# ===================CREATE optimizer and LOSSES================================= #
lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)
loss = torch.nn.L1Loss(reduction="mean")
# ========================================================== #

# =============DEFINE stuff for logs======================================== #
# meters to record stats on learning
train_loss = AverageValueMeter()
test_loss = AverageValueMeter()
best_train_loss = 10000.
with open(logfile, 'a') as f:  # open logfile and append network's architecture
  f.write(str(network) + '\n')
# ========================================================== #

# =============PROJECTION function======================================== #
def transformation(vertices, R, t):
    '''
    Calculate projective transformation of vertices given a projection matrix
    Input parameters:
    K: batch_size * 3 * 3 intrinsic camera matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters

    Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''
    # instead of P*x we compute x'*P'
    vertices = torch.matmul(vertices, R.transpose(2,1)) + t
    return vertices
# ========================================================== #

# ===================TRAINING LOOP================================= #
for epoch in range(opt.nepoch):
  # TRAIN MODE
  train_loss.reset()
  network.train()

  # Manual learning rate schedule
  if epoch == 100:
    lrate = lrate / 10.0
    optimizer = optim.Adam(network.parameters(), lr=lrate)

    
  for i, data in enumerate(dataloader_train):
    optimizer.zero_grad()

    # Load data
    points, image, _, extrinsic, _ = data

    xyz = points.cuda()
    image = image.cuda()
    extrinsic = extrinsic.cuda()

    R = extrinsic[:,0:3,0:3]
    t = extrinsic[:,0:3,3].unsqueeze(1)

    # Forward pass
    R_pred, t_pred = network(image)

    # Loss computation
    xyz_rot = transformation(xyz, R_pred, t_pred)
    xyz_rot_gt = transformation(xyz, R, t)

    batch_loss = loss(xyz_rot, xyz_rot_gt)

    batch_loss.backward()
    train_loss.update(batch_loss.item())
    torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1) # Clip gradients
    optimizer.step()  # gradient update

    print('[%d: %d/%d] train loss:  %f' % (
      epoch, i, len_dataset / opt.batch_size, batch_loss.item()))

    # VALIDATION
  test_loss.reset()
  network.eval()
  with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
      # Load data
      points, image, _, extrinsic, _ = data

      xyz = points.cuda()
      image = image.cuda()
      extrinsic = extrinsic.cuda()

      R = extrinsic[:,0:3,0:3]
      t = extrinsic[:,0:3,3].unsqueeze(1)

      # Forward pass
      R_pred, t_pred = network(image)

      # Loss computation
      xyz_rot = transformation(xyz, R_pred, t_pred)
      xyz_rot_gt = transformation(xyz, R, t)
      batch_loss = loss(xyz_rot, xyz_rot_gt)

      test_loss.update(batch_loss.item())

      print('[%d: %d/%d] test loss :  %f' % (
      epoch, i, len(dataset_test) / opt.batch_size, batch_loss.item()))

  # Save best network
  if best_train_loss > train_loss.avg:
    print('Best train loss so far: saving net...')
    torch.save(network.state_dict(), '%s/best_network.pth' % (output_folder))
    best_train_loss = train_loss.avg

  # Log metrics to wandb if available
  if WANDB_LOGS:
    wandb.log({'Test loss': test_loss.avg,
              'Train loss': train_loss.avg})
  
  # Dump stats in log file
  log_table = {'Test loss': test_loss.avg,
              'Train loss': train_loss.avg,
              'epoch': epoch,
              'lr': lrate,
              'besttrain': best_train_loss}
  
  print(log_table)
  with open(logfile, 'a') as f:
    f.write('json_stats: ' + json.dumps(log_table) + '\n')

  # Save last network
  print('saving net...')
  torch.save(network.state_dict(), os.path.join(output_folder, 'last_network.pth'))
