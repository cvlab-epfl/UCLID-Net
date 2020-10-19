import torch
import os
import time
import numpy as np
import imageio
import pickle
import dataloaders.utils as utils

## ROOT DATA DIRECTORY
DATA_LOCATION = './data'

## POINT CLOUDS PATH
POINTCLOUD_PATH = 'SurfaceSamples'

## RGB RENDERINGS PATH
RENDERING_PATH = 'renderings_rgb'

## DEPTH MAPS PATH
# Groung truth depth maps (un-clipped).
#DEPTHMAP_PATH = 'renderings_depth'
# Predicted depth maps (depth regrssion model trained on ALL CLASSES) (un-clipped)
DEPTHMAP_PATH = 'inferred_depth'

## CAMERA OPTIONS
# Load ground truth cameras?
GT_CAMERA = False
# If not, where are regressed ones:
CAMERA_PATH = 'inferred_cameras'

## SHAPENET NORMALIZATION PARAMETERS PATH
NORM_PARAMS_PATH = 'normalization_parameters.pck'


class Image_DepthMaps_PointClouds(torch.utils.data.Dataset):
  """
  Loads (posed) images + depth maps + point clouds
  Depth maps can either be ground truth or predicted, depending
   on the path defined above.
  Cameras can either be ground truth or predicted, depending
   on the boolean option defined above.
  """
  def __init__(
          self,
          split,
          subsample=5000,
          img_background='white',
          augment_data=True,
          is_train=True,
          num_views=36,
          seed=1234,
  ):
    self.subsample = subsample
    self.augment_data = augment_data
    self.is_train = is_train
    self.img_background = img_background
    self.num_views = num_views

    # Get all the instances listed in the split
    self.npyfiles = utils.get_instance_filenames(split)

    # If needed, download imageio .exr plugin, to handle depth maps
    imageio.plugins.freeimage.download()

    # POINT CLOUD PATH:
    self.pointcloud_path = os.path.join(DATA_LOCATION, POINTCLOUD_PATH)

    # IMAGE PATH:
    self.image_path = os.path.join(DATA_LOCATION, RENDERING_PATH)
    
    # DEPTH MAPS PATH
    self.depthmap_path = os.path.join(DATA_LOCATION, DEPTHMAP_PATH)

    # CAMERA PATH:
    self.inferred_cameras_path = os.path.join(DATA_LOCATION, CAMERA_PATH)

    # Print loading paths and options:
    print('RGB images loaded from: ' + self.image_path)
    print('Depth maps loaded from: ' + self.depthmap_path)
    if GT_CAMERA:
      print("Using ground truth cameras")
    else:
      print("Using inferred cameras, loaded from: " + self.inferred_cameras_path)

    # Pre-generate fixed pseudo random views for the test set
    if not self.is_train:
      np.random.seed(seed)
      self.testset_rendering_sampling = []
      for _ in range(len(self)):
        self.testset_rendering_sampling.append(np.random.randint(self.num_views))
      np.random.seed()

    # Pre-load ShapeNet normalization parameters
    norm_params_path = os.path.join(DATA_LOCATION, NORM_PARAMS_PATH)
    self.norm_params = pickle.load(open(norm_params_path, 'rb'))

  def __len__(self):
    return len(self.npyfiles)

  def __getitem__(self, idx):

    mesh_name = self.npyfiles[idx].split(".npz")[0]

    # Get point cloud
    pointcloud_filename = os.path.join(
      self.pointcloud_path, f'{mesh_name}.ply'
    )
    points, T = utils.unpack_pointcloud(pointcloud_filename, self.subsample)

    # Get view id: either random or pre-generated
    if self.is_train:
      # reset seed for data augmentation (see https://github.com/pytorch/pytorch/issues/5059)
      np.random.seed(int(time.time()) + idx)
      id = np.random.randint(0, self.num_views)
    else:
      id = self.testset_rendering_sampling[idx] # Hack here if you want a particular viewpoint
    view_id = '{0:02d}'.format(id)

    # Fetch corresponding RGB image
    image_filename = os.path.join(
      self.image_path, f'{mesh_name}/easy/{view_id}.png'
    )
    image, _ = utils.unpack_image(image_filename, self.augment_data, self.img_background)

    # Fetch cameras: either ground truth or inferred
    norm_mat = self.get_norm_matrix(mesh_name[9:])
    if GT_CAMERA:
      camera_filename = os.path.join(
        self.image_path, f'{mesh_name}/easy/rendering_metadata.txt'
      )
      intrinsic, extrinsic = utils.unpack_GT_camera(camera_filename, id, norm_mat, T)
    else:
      camera_filename = os.path.join(
        self.inferred_cameras_path, f'{mesh_name[9:]}/easy/predicted_cameras.npz'
      )
      intrinsic, extrinsic = utils.unpack_inferred_camera(camera_filename, id, norm_mat, T)

    # Return normalized pinhole camera matrix
    extrinsic = torch.from_numpy(extrinsic).float()
    intrinsic = torch.from_numpy(intrinsic).float()
    extrinsic = torch.mm(torch.tensor([[1., 0., -112.],
                                      [0., 1., -112.],
                                      [0., 0., 112.]]), torch.mm(intrinsic, extrinsic))

    # Get depth map
    depth_filename = os.path.join(
      self.depthmap_path, f'{mesh_name[9:]}/easy/{view_id}.exr'
    )
    depthmap = utils.unpack_depthmap(depth_filename)

    return points, image, depthmap, intrinsic, extrinsic, norm_mat, mesh_name, T
  
  def get_norm_matrix(self, shape_name):
    """
    Gets the normalization (scaling + offset) that was applied to go from
    ShapeNet object (from which the renderings were made) to the point clouds
    """
    norm_params = self.norm_params[shape_name]
    center, m, = norm_params[:3], norm_params[3]
    x, y, z = center[0], center[1], center[2]
    M_inv = np.asarray(
      [[m, 0., 0., 0.],
       [0., m, 0., 0.],
       [0., 0., m, 0.],
       [0., 0., 0., 1.]]
    )
    T_inv = np.asarray(
      [[1.0, 0., 0., x],
       [0., 1.0, 0., y],
       [0., 0., 1.0, z],
       [0., 0., 0., 1.]]
    )
    ret = np.matmul(T_inv, M_inv)
    return torch.from_numpy(ret).float()
