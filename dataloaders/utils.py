import torch
from plyfile import PlyData
import numpy as np
import os
from numpy.lib import recfunctions as rfn
import imageio
import time
import trimesh
import torchvision
from PIL import Image


def get_instance_filenames(split):
  npzfiles = []
  for dataset in split:
    for class_name in split[dataset]:
      for instance_name in split[dataset][class_name]:
        instance_filename = os.path.join(
          dataset, class_name, instance_name + ".npz"
        )
        npzfiles += [instance_filename]
  return npzfiles

def unpack_pointcloud(filename, subsample=12000, bounding_box=1.):
  # 1: Load raw points and subsample
  """
  plydata = PlyData.read(filename)
  total_points = plydata['vertex'].count
  point_sampling = np.random.choice(total_points, subsample, replace=False)
  points = plydata['vertex'][point_sampling]
  points = rfn.structured_to_unstructured(points)  # remove (x,y,z) labels given by PlyData
  """
  mesh = trimesh.load(filename)
  total_points = mesh.vertices.shape[0]
  point_sampling = np.random.choice(total_points, subsample, replace=False)
  points = mesh.vertices[point_sampling]
  # 2: Normalize to desired bounding box
  # compute the translation and scaling that bring these points to desired bounding box
  translation = [-0.5 * (points[:, 0].min() + points[:, 0].max()),
                 -0.5 * (points[:, 1].min() + points[:, 1].max()),
                 -0.5 * (points[:, 2].min() + points[:, 2].max())]
  scaling = 2.0 * bounding_box / max(points[:, 0].max() - points[:, 0].min(),
                                     max(points[:, 1].max() - points[:, 1].min(),
                                         points[:, 2].max() - points[:, 2].min()))
  # now compute matrix
  T = np.array(
    [[scaling, 0.0, 0.0, scaling * translation[0]],
     [0.0, scaling, 0.0, scaling * translation[1]],
     [0.0, 0.0, scaling, scaling * translation[2]],
     [0, 0, 0, 1]])
  # and apply it
  hom_points = np.hstack([points, np.ones(len(points))[:, None]])
  hom_points_recentered = (T @ hom_points.T).T
  points = hom_points_recentered[:, :3] / hom_points_recentered[:, 3:]
  # 3: Return contiguous torch tensor + scaling matrix
  return torch.from_numpy(points).float().contiguous(), torch.from_numpy(T).float()

def unpack_image(filename, augment_data=False, background="black"):
  imageRGBA = imageio.imread(filename).astype(float) / 255.0
  image = imageRGBA[:, :, 0:3]
  silhouette = (imageRGBA[:, :, 3] > 0.0).astype(float)
  if background == "white":
    image[imageRGBA[:, :, 3] < 0.5, :] = 1.0
  if augment_data:
    # reset seed for data augmentation (see https://github.com/pytorch/pytorch/issues/5059)
    np.random.seed(int(time.time()))
    # adjust image hue
    image = np.array(torchvision.transforms.functional.adjust_hue(
                        Image.fromarray(np.uint8(image * 255)),np.random.random() - 0.5))
    image = image.astype(float) / 255.0
  return torch.tensor(image).float().permute(2, 0, 1), torch.tensor(silhouette).float()

def unpack_inferred_camera(filename, view_id, norm_mat, T):
  cams = np.load(filename)
  intrinsic = cams['intrinsic'][view_id]
  extrinsic = np.linalg.multi_dot([cams['extrinsic'][view_id], norm_mat, np.linalg.inv(T)])
  return intrinsic, extrinsic

def unpack_GT_camera(filename, view_id, norm_mat, T):
  with open(filename, 'r') as f:
    lines = f.read().splitlines()
    param_lst = read_params(lines)
    rot_mat = get_rotate_matrix(-np.pi / 2)
    az, el, distance_ratio = param_lst[view_id][0], param_lst[view_id][1], param_lst[view_id][3]
    intrinsic, RT = getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224)
    W2O_mat = get_W2O_mat((param_lst[view_id][-3], param_lst[view_id][-1], -param_lst[view_id][-2]))
    extrinsic = np.linalg.multi_dot([RT, rot_mat, W2O_mat, norm_mat, np.linalg.inv(T)])
    return intrinsic, extrinsic
    
def unpack_GT_camera_unnormalized(filename, view_id):
  with open(filename, 'r') as f:
    lines = f.read().splitlines()
    param_lst = read_params(lines)
    rot_mat = get_rotate_matrix(-np.pi / 2)
    az, el, distance_ratio = param_lst[view_id][0], param_lst[view_id][1], param_lst[view_id][3]
    intrinsic, RT = getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224)
    W2O_mat = get_W2O_mat((param_lst[view_id][-3], param_lst[view_id][-1], -param_lst[view_id][-2]))
    extrinsic = np.linalg.multi_dot([RT, rot_mat, W2O_mat])
    return intrinsic, extrinsic

def read_params(lines):
  params = []
  for line in lines:
    line = line.strip()[1:-2]
    param = np.fromstring(line, dtype=float, sep=',')
    params.append(param)
  return params

def get_rotate_matrix(rotation_angle1):
  cosval = np.cos(rotation_angle1)
  sinval = np.sin(rotation_angle1)
  rotation_matrix_x = np.array([[1, 0, 0, 0],
                                [0, cosval, -sinval, 0],
                                [0, sinval, cosval, 0],
                                [0, 0, 0, 1]])
  rotation_matrix_z = np.array([[cosval, -sinval, 0, 0],
                                [sinval, cosval, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
  scale_y_neg = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ])
  neg = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
  ])
  return np.linalg.multi_dot([neg, rotation_matrix_z, rotation_matrix_z, scale_y_neg, rotation_matrix_x])

def getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224):
  """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
  F_MM = 35.  # Focal length
  SENSOR_SIZE_MM = 32.
  PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
  RESOLUTION_PCT = 100.
  SKEW = 0.
  CAM_MAX_DIST = 1.75
  CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                        [1.0, -4.371138828673793e-08, -0.0],
                        [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])
  # Calculate intrinsic matrix.
  # 2 atan(35 / 2*32)
  scale = RESOLUTION_PCT / 100
  # print('scale', scale)
  f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
  f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
  # print('f_u', f_u, 'f_v', f_v)
  u_0 = img_w * scale / 2
  v_0 = img_h * scale / 2
  K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))
  # Calculate rotation and translation matrices.
  # Step 1: World coordinate to object coordinate.
  sa = np.sin(np.radians(-az))
  ca = np.cos(np.radians(-az))
  se = np.sin(np.radians(-el))
  ce = np.cos(np.radians(-el))
  R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                        (sa * ce, ca, sa * se),
                                        (-se, 0, ce))))
  # Step 2: Object coordinate to camera coordinate.
  R_obj2cam = np.transpose(np.matrix(CAM_ROT))
  R_world2cam = R_obj2cam * R_world2obj
  cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                         0,
                                         0)))
  T_world2cam = -1 * R_obj2cam * cam_location
  # Step 3: Fix blender camera's y and z axis direction.
  R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
  R_world2cam = R_camfix * R_world2cam
  T_world2cam = R_camfix * T_world2cam
  RT = np.hstack((R_world2cam, T_world2cam))
  return K, RT

def get_W2O_mat(shift):
  T_inv = np.asarray(
    [[1.0, 0., 0., shift[0]],
     [0., 1.0, 0., shift[1]],
     [0., 0., 1.0, shift[2]],
     [0., 0., 0., 1.]]
  )
  return T_inv

def unpack_depthmap(filename):
  depthRGBA = imageio.imread(filename).astype(float)
  depth = depthRGBA[:, :, 0] * 112.
  return torch.from_numpy(np.ascontiguousarray(depth)).float()

def unpack_depthmap_subsample(filename):
  depthRGBA = imageio.imread(filename).astype(float)
  # Subsample
  depth = depthRGBA[::2, ::2, 0]
  return torch.from_numpy(np.ascontiguousarray(depth)).float()
