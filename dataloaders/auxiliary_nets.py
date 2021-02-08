import torch
import imageio
import numpy as np
import time
import imageio
import dataloaders.utils as utils
import os

## ROOT DATA DIRECTORY
DATA_LOCATION = './data'

## POINT CLOUDS PATH
POINTCLOUD_PATH = 'SurfaceSamples'

## RGB RENDERINGS PATH
RENDERING_PATH = 'renderings_rgb'

## DEPTH MAPS PATH
# Ground truth depth maps (un-clipped).
DEPTHMAP_PATH = 'renderings_depth'


class Image_DepthMaps(torch.utils.data.Dataset):
  """
  Loads GT images and depth maps from dataset
  """
  def __init__(
          self,
          split,
          img_background='white',
          augment_data=True,
          is_train=True,
          num_views=36,
          seed=1234,
  ):
    self.augment_data = augment_data
    self.is_train = is_train
    self.img_background = img_background
    self.num_views = num_views

    # Get all the instances listed in the split
    self.npyfiles = utils.get_instance_filenames(split)

    # If needed, download imageio .exr plugin, to handle depth maps
    imageio.plugins.freeimage.download()

    # IMAGE PATH:
    self.image_path = os.path.join(DATA_LOCATION, RENDERING_PATH)
    
    # DEPTH MAPS PATH
    self.depthmap_path = os.path.join(DATA_LOCATION, DEPTHMAP_PATH)

    # Pre-generate fixed pseudo random views for the test set
    if not self.is_train:
      np.random.seed(seed)
      self.testset_rendering_sampling = []
      for _ in range(len(self)):
        self.testset_rendering_sampling.append(np.random.randint(self.num_views))
      np.random.seed()

  def __len__(self):
    return len(self.npyfiles)

  def __getitem__(self, idx):

    mesh_name = self.npyfiles[idx].split(".npz")[0]

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
  
    # Get depth map
    depth_filename = os.path.join(
      self.depthmap_path, f'{mesh_name[9:]}/easy/{view_id}.exr'
    )
    depthmap = utils.unpack_depthmap_subsample(depth_filename)
    depthmap = self.z_to_f(depthmap).unsqueeze(0)
    # Prevent depth map from reaching zero
    depthmap[depthmap <= 0.005] = 0.005

    return image, depthmap, mesh_name
  
  @staticmethod
  def z_to_f(z, near=0.6, far=2.2):
    """
    takes depths as z (in 0-1000 range), clip it to the (near, far) range
    and normalize it to (0,1)
    """
    depth_maps = torch.clamp(z, near, far)
    depth_maps = (depth_maps - near) / (far - near)
    return depth_maps

  @staticmethod
  def f_to_z(f, near=0.6, far=2.2, infinity=1000.):
    """
    takes normalized depths as f (in 0-1 range), de-normalize it to the (near, far) range
    and put far values to infinity
    """
    depth_maps = f * (far - near) + near
    depth_maps[depth_maps > far - 0.01] = infinity
    return depth_maps


class Image_PointClouds(torch.utils.data.Dataset):
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
          subsample=8192,
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

    # POINT CLOUD PATH:
    self.pointcloud_path = os.path.join(DATA_LOCATION, POINTCLOUD_PATH)

    # IMAGE PATH:
    self.image_path = os.path.join(DATA_LOCATION, RENDERING_PATH)

    # Pre-generate fixed pseudo random views for the test set
    if not self.is_train:
      np.random.seed(seed)
      self.testset_rendering_sampling = []
      for _ in range(len(self)):
        self.testset_rendering_sampling.append(np.random.randint(self.num_views))
      np.random.seed()

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

    # Fetch cameras: ground truth one
    camera_filename = os.path.join(
      self.image_path, f'{mesh_name}/easy/rendering_metadata.txt'
    )
    intrinsic, extrinsic = utils.unpack_GT_camera_unnormalized(camera_filename, id)
    
    # Return normalized pinhole camera matrix
    extrinsic = torch.from_numpy(extrinsic).float()
    intrinsic = torch.from_numpy(intrinsic).float()

    return points, image, intrinsic, extrinsic, mesh_name


class Image_AllViews(torch.utils.data.Dataset):
  """
  Loads all rgb views
  """
  def __init__(
          self,
          split,
          img_background='white',
          num_views=36,
  ):
    self.img_background = img_background
    self.num_views = num_views

    # Get all the instances listed in the split
    self.npyfiles = utils.get_instance_filenames(split)

    # IMAGE PATH:
    self.image_path = os.path.join(DATA_LOCATION, RENDERING_PATH)

  def __len__(self):
    return len(self.npyfiles)

  def __getitem__(self, idx):
    mesh_name = self.npyfiles[idx].split(".npz")[0]
    # Iterate over all views of the given object
    images = []
    for id in range(self.num_views):
      view_id = '{0:02d}'.format(id)
      # Fetch corresponding RGB image
      image_filename = os.path.join(
        self.image_path, f'{mesh_name}/easy/{view_id}.png'
      )
      image, _ = utils.unpack_image(image_filename, self.img_background)
      images.append(image)
    
    images = torch.stack(images)

    return images, mesh_name


class Image_AllViews_intrinsic(torch.utils.data.Dataset):
  """
  Loads all rgb views + intrinsic cam param.
  """
  def __init__(
          self,
          split,
          img_background='white',
          num_views=36,
  ):
    self.img_background = img_background
    self.num_views = num_views

    # Get all the instances listed in the split
    self.npyfiles = utils.get_instance_filenames(split)

    # IMAGE PATH:
    self.image_path = os.path.join(DATA_LOCATION, RENDERING_PATH)

  def __len__(self):
    return len(self.npyfiles)

  def __getitem__(self, idx):
    mesh_name = self.npyfiles[idx].split(".npz")[0]
    # Iterate over all views of the given object
    images = []
    intrinsics = []
    for id in range(self.num_views):
      view_id = '{0:02d}'.format(id)
      # Fetch corresponding RGB image
      image_filename = os.path.join(
        self.image_path, f'{mesh_name}/easy/{view_id}.png'
      )
      image, _ = utils.unpack_image(image_filename, self.img_background)
      images.append(image)
      # Fetch intrinsic cam param
      camera_filename = os.path.join(
        self.image_path, f'{mesh_name}/easy/rendering_metadata.txt'
      )
      intrinsic, _ = utils.unpack_GT_camera_unnormalized(camera_filename, id)
      intrinsics.append(torch.from_numpy(intrinsic).float())
    
    images = torch.stack(images)
    intrinsics = torch.stack(intrinsics)

    return images, intrinsics, mesh_name

