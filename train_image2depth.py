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

from models import image2depth
import dataloaders.auxiliary_nets as dataset
from utils import AverageValueMeter

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=82,
      help='batch size for training')
parser.add_argument('--workers', type=int, default=12,
      help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=150,
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
  wandb.init(project='UCLID_Net', name='image2depth_' + experiments_name)
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

dataset_train = dataset.Image_DepthMaps(split=train_split,
                                        is_train=True)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size,
                                                shuffle=True, num_workers=int(opt.workers),
                                                pin_memory=True)

dataset_test = dataset.Image_DepthMaps(split=test_split,
                                      is_train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=int(opt.workers),
                                              pin_memory=True)

print(f'Training set has {len(dataset_train)} samples.')
print(f'Testing set has {len(dataset_test)} samples.')
len_dataset = len(dataset_train)
# ========================================================== #

# ===================CREATE network================================= #
network = image2depth.image2depth()

network = network.cuda()  # move network to GPU
# If needed, load existing model
if opt.model != '':
  network.load_state_dict(torch.load(opt.model))
  print('Previous net weights loaded')
# ========================================================== #

# ===================CREATE optimizer and LOSSES================================= #
lrate = 0.0001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate,
                      betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)

class RMSE_log(nn.Module):
  def __init__(self):
    super(RMSE_log, self).__init__()

  def forward(self, fake, real):
    if not fake.shape == real.shape:
      _,_,H,W = real.shape
      fake = F.upsample(fake, size=(H,W), mode='bilinear')
    loss_per_batch = torch.sqrt(torch.mean(torch.abs(torch.log(real) - torch.log(fake)) ** 2, dim=[1, 2, 3]))
    return torch.mean(loss_per_batch)

class RMSE(nn.Module):
  def __init__(self):
    super(RMSE, self).__init__()

  def forward(self, fake, real):
    if not fake.shape == real.shape:
      _,_,H,W = real.shape
      fake = F.upsample(fake, size=(H,W), mode='bilinear')
    loss_per_batch = torch.sqrt( torch.mean( torch.abs(10.*real-10.*fake) ** 2, dim=[1,2,3] ) )
    return torch.mean(loss_per_batch)

class GradLoss(nn.Module):
  def __init__(self):
    super(GradLoss, self).__init__()

  # L1 norm
  def forward(self, grad_fake, grad_real):
    return torch.mean( torch.abs(grad_real-grad_fake) )

class NormalLoss(nn.Module):
  def __init__(self):
    super(NormalLoss, self).__init__()

  def forward(self, grad_fake, grad_real):
    prod = ( grad_fake[:,:,None,:] @ grad_real[:,:,:,None] ).squeeze(-1).squeeze(-1)
    fake_norm = torch.sqrt( torch.sum( grad_fake**2, dim=-1 ) )
    real_norm = torch.sqrt( torch.sum( grad_real**2, dim=-1 ) )

    return 1 - torch.mean( prod/(fake_norm*real_norm) )

def imgrad(img):
  img = torch.mean(img, 1, True)
  fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
  conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
  weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
  if img.is_cuda:
    weight = weight.cuda()
  conv1.weight = nn.Parameter(weight)
  grad_x = conv1(img)

  fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
  weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
  if img.is_cuda:
    weight = weight.cuda()
  conv2.weight = nn.Parameter(weight)
  grad_y = conv2(img)

  #     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))

  return grad_y, grad_x

def imgrad_yx(img):
  N,C,_,_ = img.size()
  grad_y, grad_x = imgrad(img)
  return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)

rmse = RMSE()
depth_criterion = RMSE_log()
grad_criterion = GradLoss()
normal_criterion = NormalLoss()
eval_metric = RMSE_log()

# ========================================================== #

# =============DEFINE stuff for logs======================================== #
# meters to record stats on learning
train_total = AverageValueMeter()
train_logRMSE = AverageValueMeter()
train_grad = AverageValueMeter()
train_normal = AverageValueMeter()
test_logRMSE = AverageValueMeter()
test_RMSE = AverageValueMeter()
best_train_loss = 10000.
with open(logfile, 'a') as f:  # open logfile and append network's architecture
  f.write(str(network) + '\n')
# ========================================================== #

# ===================TRAINING LOOP================================= #
# constants for loss balancing
grad_factor = 10.
normal_factor = 1.
for epoch in range(opt.nepoch):
  # TRAIN MODE
  train_total.reset()
  train_logRMSE.reset()
  train_grad.reset()
  train_normal.reset()
  test_logRMSE.reset()
  test_RMSE.reset()
  network.train()

  # Manual learning rate schedule
  if epoch == 100:
    lrate = lrate / 10.0
    optimizer = torch.optim.Adam(network.parameters(), lr=lrate,
                                betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)

    
  for i, data in enumerate(dataloader_train):
    optimizer.zero_grad()

    img, depth_maps, _ = data
    img = img.cuda()
    z = depth_maps.cuda()

    # FORWARD PASS:
    z_fake = network(img)

    # Prevent from reaching 0 (otherwise cannot take log)
    z_fake = torch.clamp(z_fake, min=0.001, max=1.)

    # Compute losses
    depth_loss = depth_criterion(z_fake, z)

    grad_real, grad_fake = imgrad_yx(z), imgrad_yx(z_fake)
    grad_loss = grad_criterion(grad_fake, grad_real)     * grad_factor * (epoch>3)
    normal_loss = normal_criterion(grad_fake, grad_real) * normal_factor * (epoch>7)

    loss = depth_loss + grad_loss + normal_loss
    loss.backward()
    optimizer.step()  # gradient update

    train_total.update(loss.item())
    train_logRMSE.update(depth_loss.item())
    train_grad.update(grad_loss.item())
    train_normal.update(normal_loss.item())

    # Print info
    print("[epoch %2d][iter %4d] loss: %.4f , RMSElog: %.4f , grad_loss: %.4f , normal_loss: %.4f" \
          % (epoch, i, loss, depth_loss, grad_loss, normal_loss))

    # VISUALIZE
    if i == 0:
      for idx in [0, img.shape[0]-1]:
        save_image(img[idx],    os.path.join(output_folder, f'train_input_{epoch}_{idx}.png'))
        save_image(z[idx],      os.path.join(output_folder, f'train_GT_{epoch}_{idx}.png'))
        save_image(z_fake[idx], os.path.join(output_folder, f'train_pred_{epoch}_{idx}.png'))

  # Testing
  test_logRMSE.reset()
  test_RMSE.reset()
  print('Evaluating...')
  network.eval()

  with torch.no_grad():
    for i, data in enumerate(dataloader_test):
      img, depth_maps, _ = data
      img = img.cuda()
      z = depth_maps.cuda()

      # FORWARD PASS:
      z_fake = network(img)

      # Upsample
      if not z_fake.shape == z.shape:
        _, _, H, W = z.shape
        z_fake = F.upsample(z_fake, size=(H, W), mode='bilinear')

      rmse_eval = rmse(z_fake, z)
      rmse_log = eval_metric(z_fake, z)

      test_logRMSE.update(rmse_log.item())
      test_RMSE.update(rmse_eval.item())

      # Print info
      print("TEST [epoch %2d][iter %4d] RMSE: %.4f RMSElog: %.4f" \
            % (epoch, i, rmse_eval, rmse_log))

      # VISUALIZE
      if i == 0:
        for idx in [0, img.shape[0] - 1]:
          save_image(img[idx],    os.path.join(output_folder, f'test_input_{epoch}_{idx}.png'))
          save_image(z[idx],      os.path.join(output_folder, f'test_GT_{epoch}_{idx}.png'))
          save_image(z_fake[idx], os.path.join(output_folder, f'test_pred_{epoch}_{idx}.png'))

    print("TEST [epoch %2d] RMSE_log: %.4f RMSE: %.4f" \
          % (epoch, test_logRMSE.avg, test_RMSE.avg))

  # Save best network
  if best_train_loss > train_logRMSE.avg:
    print('Best training logRMSE loss so far: saving net...')
    torch.save(network.state_dict(), os.path.join(output_folder, 'best_network.pth'))
    best_train_loss = train_logRMSE.avg

    # Log metrics to wandb if available
  if WANDB_LOGS:
    wandb.log({'Test RMSE': test_RMSE.avg,
              'Test logRMSE': test_logRMSE.avg,
              'Best Train logRMSE': best_train_loss,
              'Train logRMSE': train_logRMSE.avg,
              'Train total': train_total.avg,
              'Train grad': train_grad.avg,
              'Train normal': train_normal.avg})
  
  # Dump stats in log file
  log_table = {'Test RMSE': test_RMSE.avg,
              'Test logRMSE': test_logRMSE.avg,
              'Best Train logRMSE': best_train_loss,
              'Train logRMSE': train_logRMSE.avg,
              'Train total': train_total.avg,
              'Train grad': train_grad.avg,
              'Train normal': train_normal.avg,
              'epoch': epoch,
              'lr': lrate}

  print(log_table)
  with open(logfile, 'a') as f:
    f.write('json_stats: ' + json.dumps(log_table) + '\n')

  # Save last network
  print('saving net...')
  torch.save(network.state_dict(), os.path.join(output_folder, 'last_network.pth'))
