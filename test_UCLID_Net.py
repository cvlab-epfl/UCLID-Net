import argparse
import random
import string
import datetime
import os
import json
import torch
import torch.optim as optim
from torchvision.utils import save_image
from collections import defaultdict
import tqdm

from models import UCLID_Net
import dataloaders.UCLID_Net as dataset
from utils import save_pointcloud, save_pointcloud_colored
from utils import AverageValueMeter, test_chamfer, test_f_score, test_shellIoU

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=6,
      help='number of data loading workers')
parser.add_argument('--model', type=str, required=True,
      help='path to trained model')
parser.add_argument('--num_points', type=int, default=12000,
      help='number of points for sampling GT shapes')
parser.add_argument('--test_split', type=str, default = 'data/splits/all_13_classes_test.json',
      help='testing split')
parser.add_argument('--nb_cells', type=int, default=28,
      help='grid size of the cuboid output')
parser.add_argument('--n_2D_featuremaps', type=int, default=292,
      help='# of 2D feature maps at the bottom of the UNet-like architecture')
parser.add_argument('--test_point_samples', type=int, default=50,
      help='# of points samples per voxel generated')
parser.add_argument('--pts_for_chd', type=int, default=2048,
      help='number of source/target points for CHD loss computation.')
parser.add_argument('--pts_for_fscore', type=int, default=10000,
      help='number of source/target points for f-score loss computation.')
parser.add_argument('--pts_for_IoU', type=int, default=7000,
      help='number of source/target points for shell-IoU loss computation.')

opt = parser.parse_args()
print(opt)
# ========================================================== #

# ===================CREATE DATASET================================= #
# Create test dataloader
with open(opt.test_split, "r") as f:
  test_split = json.load(f)

dataset_test = dataset.Image_DepthMaps_PointClouds(split=test_split,
                                                  subsample=opt.num_points,
                                                  is_train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, # TODO: allow inference with >1 BS
                                              shuffle=False, num_workers=int(opt.workers),
                                              pin_memory=True)

print(f'Testing set has {len(dataset_test)} samples.')
len_dataset = len(dataset_test)

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
# Load trained model
try:
  network.load_state_dict(torch.load(opt.model))
  print('Trained net weights loaded')
except:
  print('ERROR: Failed to load net weights')
  exit()
network.eval()
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
# Overall average metrics
overall_chd_loss = AverageValueMeter()
overall_iou_loss = AverageValueMeter()
overall_f_score_5_percent = AverageValueMeter()

# Per shape category metrics
per_cat_items = defaultdict(lambda: 0)
per_cat_chd_loss = defaultdict(lambda: AverageValueMeter())
per_cat_iou_loss = defaultdict(lambda: AverageValueMeter())
per_cat_f_score_5_percent = defaultdict(lambda: AverageValueMeter())

if not os.path.exists(opt.model[:-4]):
  os.mkdir(opt.model[:-4])
  print(f'created dir {opt.model[:-4]}/ for saving outputs')
output_folder = opt.model[:-4]
# ========================================================== #

# =============TESTING LOOP======================================== #
# Iterate on all test data
pbar = tqdm.tqdm(total=len_dataset)
with torch.no_grad():
  for data in dataloader_test:
    pbar.update()
    gt_points, img, depth_maps, _, camRt, _, mesh_name, T = data

    gt_points = gt_points.cuda()
    img = img.cuda()
    depth_maps = depth_maps.cuda()
    camRt = camRt.cuda()

    cat = mesh_name[0].split('/')[-2]
    fn = mesh_name[0].split('/')[-1]
    per_cat_items[cat] = per_cat_items[cat] + 1
    B_size = gt_points.shape[0]

    # FORWARD PASS: reconstruct points from images
    occupancy, pointsReconstructed, _ = network(img, camRt, depth_maps, opt.test_point_samples)
    # occupancy : shape (batch, 1, nb_cells, nb_cells, nb_cells)
    # pointsReconstructed : shape (batch, N, points per voxels, 3)
    #   with N the max. number of occupied voxels in the batch
    # mask : here ignored, because no padding happens since we run with batch size 1
    # Map generated points back to original bounding box ([1,1]^3)
    pointsReconstructed = (2.0 / opt.nb_cells) * pointsReconstructed - 1.0
    # Then map back to the scaling used in DISN
    scale = T[0,0,0]
    pointsReconstructed = pointsReconstructed / scale
    gt_points = gt_points / scale

    points_flat = pointsReconstructed.reshape(B_size, -1, 3)
    
    # In case the output is empty, just randomly put 100 points
    if points_flat.shape[1] == 0:
      print(f'Error: for shape {mesh_name}, the output is empty')
      points_flat = torch.rand(1, 100, 3).cuda() - 0.5

    ##### f-score computation
    f_score_value = test_f_score(points_flat, gt_points, opt.pts_for_fscore).item()
    overall_f_score_5_percent.update(f_score_value)
    per_cat_f_score_5_percent[cat].update(f_score_value)

    ##### Chamfer loss
    chd_value = test_chamfer(points_flat, gt_points, opt.pts_for_chd).item()
    overall_chd_loss.update(chd_value)
    per_cat_chd_loss[cat].update(chd_value)

    ##### IoU computation
    iou_value = test_shellIoU(points_flat, gt_points, opt.pts_for_IoU).item()
    overall_iou_loss.update(iou_value)
    per_cat_iou_loss[cat].update(iou_value)

    # Save output point clouds for the first 10 objects per category
    if per_cat_items[cat] > 10:
      continue

    save_pointcloud(gt_points[0].data.cpu(),
                    os.path.join(output_folder, f'{fn}_GT.ply'))
    save_pointcloud_colored(points_flat[0].data.cpu(),
                            os.path.join(output_folder, f'{fn}_gen.ply'),
                            opt.test_point_samples)
    save_image(img[0], os.path.join(output_folder, f'{fn}_view.png'))

pbar.close()
## Print and save metrics
log_table = {
  "overall_chd": overall_chd_loss.avg,
  "overall_iou": overall_iou_loss.avg,
  "overall_f_score_5_percent" : overall_f_score_5_percent.avg,
}

for cat in sorted(per_cat_chd_loss):
  chd = per_cat_chd_loss[cat].avg
  iou = per_cat_iou_loss[cat].avg
  f_s = per_cat_f_score_5_percent[cat].avg

  log_table.update({f'{cat}_CHD': chd})
  log_table.update({f'{cat}_IoU': iou})
  log_table.update({f'{cat}_f5%': f_s})

print(log_table)

with open(os.path.join(output_folder, 'test_metrics.txt'), 'a') as f:
  f.write('json_stats: ' + json.dumps(log_table) + '\n')
