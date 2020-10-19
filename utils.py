import numpy as np
import torch
from extensions import dist_chamfer, func_grid_pooling
from plyfile import PlyData, PlyElement

distChamfer = dist_chamfer.chamferDist()
gridPooling = func_grid_pooling.gridPooling()

class AverageValueMeter(object):
  """
  Computes and stores the average and current value of a sequence of floats
  """
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0.0
    self.min = np.inf
    self.max = -np.inf

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    self.min = min(self.min, val)
    self.max = max(self.max, val)


def get_target_occupancy(points, grid_size):
  """
  Given a batch of point clouds, computes their target binary occupancy
  in a voxel grid of size grid_size**3

  :param points: torch tensor of size (batch, n_points, 3)
  :param grid_size: integer value, giving the number of grid cells along each xyz direction

  :return target_occupancy: binary occupancy grids, of shape
    (batch, 1, grid_size, grid_size, grid_size)
  """
  # prepare target voxel grid - scale points up to fit voxel grid
  points_for_voxelization = (grid_size + grid_size * points) / 2.0
  # create a fake feature vector for each point (composed of a single 1)
  indicator_points = torch.ones(points.shape[0], points.shape[1], 1).cuda()
  # max pooling on the grid: target_occupancy has shape (batch, 1, grid_size, grid_size, grid_size)
  # (with 1 being the size of the feature map, fake in our case..)
  target_occupancy = gridPooling(indicator_points, points_for_voxelization,
                                grid_size, grid_size, grid_size).float()
  return target_occupancy

def save_pointcloud(point_cloud, filename):
  vertex = np.array([tuple(x) for x in point_cloud.tolist()],
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
  el = PlyElement.describe(vertex, 'vertex')
  PlyData([el]).write(filename)


def save_pointcloud_colored(point_cloud, filename, points_per_patch):
  # Vertices
  vertex = np.array([tuple(x) for x in point_cloud.tolist()],
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
  # Colors: 1 random color per patch
  num_activated_voxels = vertex.shape[0] // points_per_patch
  colors = np.random.randint(0, 255, (num_activated_voxels, 3)).repeat(points_per_patch, axis=0)
  vertex_color = np.array([tuple(x) for x in colors],
                          dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

  # Fuse verts+colors arrays
  num_verts = len(vertex)
  vertex_all = np.empty(num_verts, vertex.dtype.descr + vertex_color.dtype.descr)

  for prop in vertex.dtype.names:
      vertex_all[prop] = vertex[prop]

  for prop in vertex_color.dtype.names:
      vertex_all[prop] = vertex_color[prop]

  ply = PlyData([PlyElement.describe(vertex_all, 'vertex')])
  ply.write(filename)

#### METRICS

def subsample(pointcloud, n_point):
  try:
    choice = np.random.choice(pointcloud.shape[1], n_point, replace=False)
  except:
    # Not enough samples: sample with replacement
    choice = np.random.choice(pointcloud.shape[1], n_point, replace=True)
  return choice

def test_shellIoU(x, y, n_points=7000, resolution=50):
  """
  Point clouds:
    x: B, SizeX, 3
    y: B, SizeY, 3
  both normalized to unit radius sphere
  """
  # Subsample
  choice_target = subsample(x, n_points)
  choice_source = subsample(y, n_points)
  target = x[:,choice_target,:].contiguous()
  source = y[:,choice_source,:].contiguous()
  # prepare target voxel grid - scale points up to fit voxel grid
  x_for_voxelization = (resolution + resolution * target) / 2.0
  y_for_voxelization = (resolution + resolution * source) / 2.0
  # create a fake feature vector for each point (composed of a single 1)
  x_indicator_points = torch.ones(target.shape[0], target.shape[1], 1).cuda()
  y_indicator_points = torch.ones(source.shape[0], source.shape[1], 1).cuda()
  # pooling on grid: target_occupancy has shape (batch, 1, nb_cells, nb_cells, nb_cells)
  # (with 1 being the size of the feature map, fake in our case..)
  x_grid = gridPooling(x_indicator_points, x_for_voxelization,
                                    resolution, resolution, resolution).bool()
  y_grid = gridPooling(y_indicator_points, y_for_voxelization,
                                    resolution, resolution, resolution).bool()
  # IoU
  inter = x_grid * y_grid
  union = x_grid + y_grid
  IoU = torch.sum(inter, dim=[1,2,3,4]).float() / torch.sum(union, dim=[1,2,3,4]).float()
  return IoU # Tensor of size B

def p_r_f_from_distmat(dist1, dist2, threshold_distance):
  """
  Given two directional distance matrices (dist1, dist2) and a threshold distance,
  Computes the precision (P), recall (R) and f-score (F) associated to points being
  at a distance < threshold distance to their target.
  """
  P = 100. * (dist1 < threshold_distance).sum() / (dist1.shape[0] * dist1.shape[1])
  R = 100. * (dist2 < threshold_distance).sum() / (dist2.shape[0] * dist2.shape[1])
  if P + R > 0.:
    F = 2*P*R / (P + R)
  else:
    F = torch.tensor([0.])
  return P,R,F

def test_f_score(x, y, n_points=10000, thresh=0.05):
  # Subsample  
  choice_target = subsample(y, n_points)
  choice_source = subsample(x, n_points)
  # Compute pairwise closest distances
  dist1, dist2 = distChamfer(x[:, choice_source, :].contiguous(),
                              y[:, choice_target, :].contiguous())
  # Compute f-score and return it
  f_score_value = p_r_f_from_distmat(dist1, dist2, (2 * thresh) **2)[2]
  return f_score_value

def test_chamfer(x, y, n_points=2048):
  # Subsample
  choice_target = subsample(y, n_points)
  choice_source = subsample(x, n_points)
  # Compute Chamfer and return it
  dist1, dist2 = distChamfer(x[:, choice_source, :].contiguous(), y[:, choice_target, :].contiguous())
  chd_value = (torch.mean(dist1) + torch.mean(dist2))
  return chd_value
