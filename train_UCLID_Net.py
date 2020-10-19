import argparse
import random
import string
import datetime
import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

from models import UCLID_Net
import dataloaders.UCLID_Net as dataset
from utils import AverageValueMeter, get_target_occupancy, save_pointcloud
from extensions import dist_chamfer

distChamfer = dist_chamfer.chamferDist()

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=28,
      help='batch size for training')
parser.add_argument('--workers', type=int, default=6,
      help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=150,
      help='number of epochs to train for')
parser.add_argument('--model', type=str, default='',
      help='optional reload model path')
parser.add_argument('--num_points', type=int, default=5000,
      help='number of points for sampling GT shapes')
parser.add_argument('--train_split', type=str, default = 'data/splits/cars_train.json',
      help='training split')
parser.add_argument('--test_split', type=str, default = 'data/splits/cars_test.json',
      help='testing split')
parser.add_argument('--nb_cells', type=int, default=28,
      help='grid size of the cuboid output')
parser.add_argument('--n_2D_featuremaps', type=int, default=292,
      help='# of 2D feature maps at the bottom of the UNet-like architecture')
parser.add_argument('--output_dir', type=str, default="output/",
      help='where to log outputs')
parser.add_argument('--experiment_name', type=str, default='',
      help='Used for creating output directory and Wandb experiment')
parser.add_argument('--train_point_samples', type=int, default=10,
      help='# of points samples per voxel for training')
parser.add_argument('--test_point_samples', type=int, default=10,
      help='# of points samples per voxel for testing')

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
  wandb.init(project='UCLID_Net', name='UCLID_Net_' + experiments_name)
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

dataset_train = dataset.Image_DepthMaps_PointClouds(split=train_split,
                                                    subsample=opt.num_points,
                                                    is_train=True)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size,
                                                shuffle=True, num_workers=int(opt.workers),
                                                pin_memory=True)
# Small batch size, for the first training epoch
dataloader_small_bs = torch.utils.data.DataLoader(dataset_train, batch_size=2,
                                                  shuffle=True, num_workers=int(opt.workers),
                                                  pin_memory=True)

dataset_test = dataset.Image_DepthMaps_PointClouds(split=test_split,
                                                  subsample=opt.num_points,
                                                  is_train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=int(opt.workers),
                                              pin_memory=True)

print(f'Training set has {len(dataset_train)} samples.')
print(f'Testing set has {len(dataset_test)} samples.')
len_dataset = len(dataset_train)

# TODO: switch depth map size based on a proper option
if 'inferred' in dataset.DEPTHMAP_PATH:
  depth_map_size = 112
else:
  depth_map_size = 224

# ========================================================== #

# ===================CREATE network================================= #
network = UCLID_Net.UCLID_Net(nb_cells=opt.nb_cells,
                              n_2D_featuremaps=opt.n_2D_featuremaps,
                              depth_map_size=depth_map_size)

network = network.cuda()  # move network to GPU
# If needed, load existing model
if opt.model != '':
  network.load_state_dict(torch.load(opt.model))
  print('Previous net weights loaded')
# ========================================================== #

# ===================CREATE optimizer================================= #
lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)
# ========================================================== #

# =============DEFINE stuff for logs======================================== #
# meters to record stats on learning
total_train_loss = AverageValueMeter()
chd_train_loss = AverageValueMeter()
occ_train_loss = AverageValueMeter()
test_loss = AverageValueMeter()
best_train_loss = 10
with open(logfile, 'a') as f:  # open logfile and append network's architecture
  f.write(str(network) + '\n')
# ========================================================== #

# =============FIRST TRAINING EPOCH: occupancy only======================================== #
network.train()
for i, data in enumerate(dataloader_small_bs, 0):
  optimizer.zero_grad()

  points, img, depth_maps, _, camRt, _, _, _ = data

  points = points.cuda()
  img = img.cuda()
  depth_maps = depth_maps.cuda()
  camRt = camRt.cuda()

  target_occupancy = get_target_occupancy(points, opt.nb_cells)

  # FORWARD PASS: reconstruct points from images
  occupancy, _, _ = network(img, camRt, depth_maps, opt.train_point_samples)
  # occupancy : shape (batch, 1, nb_cells, nb_cells, nb_cells)

  loss_occ = F.binary_cross_entropy(occupancy, target_occupancy)
  loss_net = 100.0 * loss_occ
  loss_net.backward()
  optimizer.step()  # gradient update

  print('[PRETRAIN: %d/%d] train occupancy loss:  %f  ' % (
    i, len_dataset / 2, loss_occ.item()))
# ========================================================== #

# =============FULL TRAINING LOOP======================================== #
for epoch in range(opt.nepoch):
  # TRAIN MODE
  total_train_loss.reset()
  chd_train_loss.reset()
  occ_train_loss.reset()
  network.train()

  # Manual learning rate schedule
  if epoch == 100:
    lrate = lrate / 10.
    optimizer = optim.Adam(network.parameters(), lr=lrate)

  for i, data in enumerate(dataloader_train, 0):
    optimizer.zero_grad()

    points, img, depth_maps, _, camRt, _, mesh_name, _ = data
    points = points.cuda()
    B_size = points.shape[0]
    img = img.cuda()
    depth_maps = depth_maps.cuda()
    camRt = camRt.cuda()
    target_occupancy = get_target_occupancy(points, opt.nb_cells)

    # FORWARD PASS: reconstruct points from images
    occupancy, pointsReconstructed, mask = network(img, camRt, depth_maps, opt.train_point_samples)
    # occupancy : shape (batch, 1, nb_cells, nb_cells, nb_cells)
    # pointsReconstructed : shape (batch, N, points per voxels, 3)
    #   with N the max. number of occupied voxels in the batch
    # mask : shape (batch, N). Some points in pointsReconstructed correspond
    # folded patches in empty voxels, and only exist so that a single tensor
    # can be returned for the whole batch, with different number of occupied voxels.
    # The mask information allows to discard unoccupied voxels.

    # Map generated points back to original bounding box ([1,1]^3)
    pointsReconstructed = (2.0 / opt.nb_cells) * pointsReconstructed - 1.0

    # Compute masked Chamfer distance
    mask_per_pts = mask.unsqueeze(-1).repeat(1,1,opt.train_point_samples)
    mask_per_pts = mask_per_pts.reshape(B_size, -1)
    points_flat = pointsReconstructed.reshape(B_size, -1, 3)
    dist1, dist2 = distChamfer(points_flat.contiguous(), points)  # loss function

    # Now assemble loss masking out padded entries
    mean_dist1 = 0
    for k in range(B_size):
      dist1_per_batch = dist1[k]
      mask_per_batch = mask_per_pts[k, :]
      mean_dist1 += torch.mean(dist1_per_batch[mask_per_batch]) / B_size

    # Finally here is the final loss computation:
    # both sides of Chamfer + BCE on occupancy grids
    loss_ch = torch.mean(dist2) + mean_dist1
    loss_occ = F.binary_cross_entropy(occupancy, target_occupancy)
    loss_net = loss_ch + 100.0 * loss_occ

    loss_net.backward()
    optimizer.step()  # gradient update

    total_train_loss.update(loss_net.item())
    chd_train_loss.update(loss_ch.item())
    occ_train_loss.update(loss_occ.item())

    # VISUALIZE
    if i % 200 <= 0:
      print("Storing to file...")
      save_pointcloud(points[0].data.cpu(),
                      os.path.join(output_folder, f'train_GT_{epoch}_{i}.ply'))
      save_pointcloud(points_flat[0][mask_per_pts[0]].data.cpu(),
                      os.path.join(output_folder, f'train_output_{epoch}_{i}.ply'))
      save_image(img[0], os.path.join(output_folder, f'train_input_{epoch}_{i}.png'))
      
    print('[%d: %d/%d] Train Chamfer Loss:  %f, Train Occupancy Loss:  %f  ' % (
      epoch, i, len_dataset / opt.batch_size, loss_ch.item(), loss_occ.item()))

  # Testing
  test_loss.reset()

  network.eval()
  with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
      points, img, depth_maps, _, camRt, _, mesh_name = data
      points = points.cuda()
      B_size = points.shape[0]
      img = img.cuda()
      depth_maps = depth_maps.cuda()
      camRt = camRt.cuda()
      target_occupancy = get_target_occupancy(points, opt.nb_cells)

      # FORWARD PASS: reconstruct points from images
      occupancy, pointsReconstructed, mask = network(img, camRt, depth_maps, opt.train_point_samples)

      # Map generated points back to original bounding box ([1,1]^3)
      pointsReconstructed = (2.0 / opt.nb_cells) * pointsReconstructed - 1.0

      # Compute masked Chamfer distance
      mask_per_pts = mask.unsqueeze(-1).repeat(1, 1, opt.train_point_samples)
      mask_per_pts = mask_per_pts.reshape(B_size, -1)
      points_flat = pointsReconstructed.reshape(B_size, -1, 3)
      dist1, dist2 = distChamfer(points_flat.contiguous(), points)  # loss function

      # Now assemble loss masking out padded entries
      mean_dist1 = 0
      for k in range(B_size):
        dist1_per_batch = dist1[k]
        mask_per_batch = mask_per_pts[k, :]
        mean_dist1 += torch.mean(dist1_per_batch[mask_per_batch]) / B_size

      # both sides of Chamfer + BCE on occupancy grids
      loss_ch = torch.mean(dist2) + mean_dist1
      loss_occ = F.binary_cross_entropy(occupancy, target_occupancy)
      test_loss.update(loss_ch.item())

      # VISUALIZE
      if i % 200 <= 0:
        print("Storing to file...")
        save_pointcloud(points[0].data.cpu(),
                        os.path.join(output_folder, f'test_GT_{epoch}_{i}.ply'))
        save_pointcloud(points_flat[0][mask_per_pts[0]].data.cpu(),
                        os.path.join(output_folder, f'test_output_{epoch}_{i}.ply'))
        save_image(img[0], os.path.join(output_folder, f'test_input_{epoch}_{i}.png'))
      
      print('[%d: %d/%d] Test Chamfer Loss:  %f, Test Occupancy Loss:  %f  ' % (
        epoch, i, len(dataset_test) / opt.batch_size, loss_ch.item(), loss_occ.item()))

  # Save best network
  if best_train_loss > chd_train_loss.avg:
    print('Best training CHD loss so far: saving net...')
    torch.save(network.state_dict(), os.path.join(output_folder, 'best_network.pth'))
    best_train_loss = chd_train_loss.avg

  # Log metrics to wandb if available
  if WANDB_LOGS:
    wandb.log({'Training Loss total': total_train_loss.avg,
              'Test Loss CHD': test_loss.avg,
              'Best Train Loss CHD': best_train_loss,
              'Train Loss CHD': chd_train_loss.avg,
              'Training Loss OCC': occ_train_loss.avg})

  # Dump stats in log file
  log_table = {
    'train_loss': total_train_loss.avg,
    'test_loss': test_loss.avg,
    'epoch': epoch,
    'lr': lrate,
    'besttrainchd': best_train_loss,
  }
  
  print(log_table)
  with open(logfile, 'a') as f:
    f.write('json_stats: ' + json.dumps(log_table) + '\n')

  # Save last network
  print('saving net...')
  torch.save(network.state_dict(), os.path.join(output_folder, 'last_network.pth'))
