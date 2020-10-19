import torch
import torch.nn as nn
import torch.nn.functional as F
import models.resnet as resnet
import numpy as np
from extensions import func_grid_pooling

gridPooling = func_grid_pooling.gridPooling()


class PointSamplerLinear(nn.Module):
  """
  Patch folder, a la AtlaNet but on a 3D grid of deformation codes
  """
  def __init__(self,
               features):
    super(PointSamplerLinear, self).__init__()
    self.features = features

    # first fold
    self.conv11 = nn.Conv2d(features + 2, features * 2, bias=False, kernel_size=1, stride=1, padding=0)
    self.bn11 = nn.BatchNorm2d(features * 2)
    self.relu11 = nn.ReLU(inplace=True)
    self.conv12 = nn.Conv2d(features * 2, features * 2, bias=False, kernel_size=1, stride=1, padding=0)
    self.bn12 = nn.BatchNorm2d(features * 2)
    self.relu12 = nn.ReLU(inplace=True)
    self.conv13 = nn.Conv2d(features * 2, features * 2, bias=False, kernel_size=1, stride=1, padding=0)
    self.bn13 = nn.BatchNorm2d(features * 2)
    self.relu13 = nn.ReLU(inplace=True)

    # second fold
    self.conv21 = nn.Conv2d(features * 2 + features, features * 2, bias=False, kernel_size=1, stride=1, padding=0)
    self.bn21 = nn.BatchNorm2d(features * 2)
    self.relu21 = nn.ReLU(inplace=True)
    self.conv22 = nn.Conv2d(features * 2, features * 2, bias=False, kernel_size=1, stride=1, padding=0)
    self.bn22 = nn.BatchNorm2d(features * 2)
    self.relu22 = nn.ReLU(inplace=True)
    self.conv23 = nn.Conv2d(features * 2, features * 2, bias=False, kernel_size=1, stride=1, padding=0)
    self.bn23 = nn.BatchNorm2d(features * 2)
    self.relu23 = nn.ReLU(inplace=True)

    # final layer
    self.conv3 = nn.Conv2d(features * 2, 3, bias=False, kernel_size=1, stride=1, padding=0)
    self.sigmoid = nn.Sigmoid()

  def forward(self, deformation, samplings=5):
    """
    Parameters
    ----------
    deformation : torch tensor of size (batch, N: number of patches, deformation code size)

    samplings : 
      desired number of points per deformed patch

    Returns
    -------
    offset : torch tensor of size (batch, N: number of patches, samplings, 3)
      3D position of the points sampled points, relatively to their corresponding voxel coords.

    """
    B = deformation.shape[0]
    # get random samplings: features define 3D deformation of 2D surface, and we sample (u,v) coordinates from it
    eps = torch.rand(B, samplings, deformation.shape[1], 2).float().cuda()

    # now repeat deformation samplings times for each cell
    deformation = deformation.unsqueeze(1).repeat(1, samplings, 1, 1)   # batch, samplings, N, features
    sampler = torch.cat((deformation, eps), -1)                         # batch, samplings, N, features + 2
    sampler = sampler.permute(0,3,1,2).contiguous()                     # batch, features + 2, samplings, N

    # first fold
    sampler = self.relu11(self.bn11(self.conv11(sampler)))
    sampler = self.relu12(self.bn12(self.conv12(sampler)))
    sampler = self.relu13(self.bn13(self.conv13(sampler)))

    # second fold
    sampler = torch.cat((deformation.permute(0,3,1,2), sampler), 1)
    sampler = self.relu21(self.bn21(self.conv21(sampler)))
    sampler = self.relu22(self.bn22(self.conv22(sampler)))
    sampler = self.relu23(self.bn23(self.conv23(sampler)))

    # map to offset
    offset = self.sigmoid(self.conv3(sampler))  # batch, 3, samplings, N
    # finally reshape back to correct size [B, N, samplings, 3]
    offset = offset.permute(0,3,2,1)
    return offset


def conv3x3x3(in_planes, out_planes, stride=1):
  """
  3x3x3 (3D) convolution with padding
  """
  return nn.Conv3d(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=stride,
    padding=1,
    bias=False)


class BasicBlockDecoder(nn.Module):
  """
  Residual 3D convolutional block
  """
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, upsample=None):
    super(BasicBlockDecoder, self).__init__()

    self.conv1 = conv3x3x3(inplanes, inplanes)
    self.bn1 = nn.BatchNorm3d(inplanes)
    self.relu = nn.ReLU(inplace=True)

    if stride > 1:
      self.conv2 = nn.ConvTranspose3d(inplanes, planes, kernel_size=2, stride=stride)
    else:
      self.conv2 = conv3x3x3(inplanes, planes, stride)

    self.bn2 = nn.BatchNorm3d(planes)
    self.upsample = upsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.upsample is not None:
      residual = self.upsample(x)

    out += residual
    out = self.relu(out)

    return out

class ResnetDecoderUNet(nn.Module):
  """
  Resnet-like decoder, with multiscale skip connections
  """
  def __init__(self,
               out_planes,
               shape,
               nb_cells,
               layers,
               block=BasicBlockDecoder):

    super(ResnetDecoderUNet, self).__init__()
    assert len(layers) == 4
    self.inplanes = shape
    self.shape = shape

    # handle variable output resolution
    spatial_resolution = 7
    stride = 2

    self.layer1 = self._make_layer(block, shape, layers[0], stride=1)

    # start upsampling
    self.layer2 = self._make_layer(block, shape // 2, layers[1], stride=stride)
    spatial_resolution = spatial_resolution * 2
    if spatial_resolution == nb_cells:
      stride = 1
    self.inplanes = self.inplanes + 32 # add the channel size of l3
    self.layer3 = self._make_layer(block, shape // 4, layers[2], stride=stride)
    spatial_resolution = spatial_resolution * 2
    if spatial_resolution == nb_cells:
      stride = 1
    self.inplanes = self.inplanes + 32 # add the channel size of l2
    self.layer4 = self._make_layer(block, shape // 4, layers[3], stride=stride)

    self.output_layer = nn.Conv3d(
      shape // 4 + 32,  # add the channel size of l1
      out_planes,
      kernel_size=1,
      stride=1,
      padding=0,
      bias=False)

  def _make_layer(self, block, planes, blocks, stride=1):
    upsample = None
    if stride != 1:# or self.inplanes != planes * block.expansion:
      upsample = nn.Sequential(
        nn.ConvTranspose3d(self.inplanes, planes * block.expansion, kernel_size=2, stride=stride, bias=False),
        nn.BatchNorm3d(planes * block.expansion))
    elif self.inplanes != planes * block.expansion:
      upsample = nn.Sequential(
        nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
        nn.BatchNorm3d(planes * block.expansion))

    layers = []
    layers.append(block(self.inplanes, planes, stride, upsample))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, l4, l3, l2, l1):
    """
    Parameters
    ----------
    l1, l2, l3, l4: torch volumetric tensors
      corresponding to multi-scale embeddings of the input
      l4: bottom of the UNet (smaller spatial res.)
      ...
      l1: top of the UNet (higher spatial res.)
    """
    x = self.layer1(l4)     # batch, fm0, 7, 7, 7
    x = self.layer2(x)      # batch, fm0 / 2, 14, 14, 14
    # concat l3
    x = torch.cat([x, l3], dim=1)
    x = self.layer3(x)      # batch, fm0 / 4, 28, 28, 28
    # concat l2
    x = torch.cat([x, l2], dim=1)
    x = self.layer4(x)      # batch, fm0 / 4, 28, 28, 28
    # concat l1
    x = torch.cat([x, l1], dim=1)
    x = self.output_layer(x)# batch, out_fm, 28, 28, 28
    return x


class UCLID_Net(nn.Module):
  def __init__(self, n_2D_featuremaps=292, nb_cells=28, nb_cells_latent=7,
              depth_map_size=112, max_depth_value=300.):
    super(UCLID_Net, self).__init__()
    # Number of 2D feature maps at the bottom of the UNet
    self.n_2D_featuremaps = n_2D_featuremaps
    # 3D grid size for backprojection at the bottom of the UNet
    self.nb_cells_latent = nb_cells_latent
    # 3D grid size for backprojection at the highest resolution (=output res.)
    self.nb_cells = nb_cells
    # Maximum depth value, for clamping depth maps
    self.max_depth_value = max_depth_value
    # Occupancy value above which a voxel is considered occupied
    self.thresh_occupancy = 0.35

    # Convolutional image encoder
    self.encoder_2D = resnet.resnet18conv()

    # 1x1 convolutions to reduce feature size before backprojecting them to 3D
    # l4: bottom of the UNet (highest number of features)
    # ...
    # l1: top of the UNet (smallest number of features)
    self.conv1x1_l4 = nn.Conv2d(512, n_2D_featuremaps - 2, kernel_size=1, stride=1,
                            padding=0, bias=True)
    self.conv1x1_l3 = nn.Conv2d(256, 32 - 2, kernel_size=1, stride=1,
                            padding=0, bias=True)
    self.conv1x1_l2 = nn.Conv2d(128, 32 - 2, kernel_size=1, stride=1,
                                padding=0, bias=True)
    self.conv1x1_l1 = nn.Conv2d(64, 32 - 2, kernel_size=1, stride=1,
                                padding=0, bias=True)

    # Volumetric 3D decoder: grid of size nb_cells**3 with 40 channels as output,
    # and n_2D_featuremaps as inputs
    self.decoder_3D = ResnetDecoderUNet(40, n_2D_featuremaps, nb_cells, [2, 2, 2, 2])

    # The first 8 channels of the grid will serve for occupancy classification,
    # after going through a 3D convolution (8 channels > 1 channel), then sigmoid
    self.decoder_occupancy = nn.Conv3d(8, 1, kernel_size=1, stride=1,
                                      padding=0, bias=True)
    self.sigmoid = nn.Sigmoid()

    # The other 32 channels serve as local codes for folding patches
    self.patch_sampler = PointSamplerLinear(32)

    ## Prepare tensors used for backprojection
    # Grid sizes at different scales
    self.grid_sizes = [self.nb_cells_latent,
                      self.nb_cells_latent * 2,
                      self.nb_cells_latent * 4] # Default: [7, 14, 28]

    # l4: bottom of the UNet (smaller spatial res.)
    # ...
    # l1: top of the UNet (higher spatial res.)
    self.l4_grid_coord = self.grid_coord_generator(self.grid_sizes[0]).reshape(-1,3)
    self.l3_grid_coord = self.grid_coord_generator(self.grid_sizes[1]).reshape(-1,3)
    self.l2_l1_grid_coord = self.grid_coord_generator(self.grid_sizes[2]).reshape(-1,3)
    
    # Compute grid coordinates for output point cloud
    xv, yv, zv = np.meshgrid(np.arange(0, self.nb_cells), np.arange(0, self.nb_cells), np.arange(0, self.nb_cells),
                             indexing='ij')
    xyz = torch.tensor(np.stack((xv, yv, zv), axis=0), requires_grad=False).float()
    self.xyz_flat = xyz.permute(1,2,3,0).view(-1, 3) # grid**3, 3

    ## For backprojecting depth maps:
    # Reverting x and z axis
    self.revert_axis = torch.tensor([ [1., 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, -1, 0],
                                      [0, 0, 0, -1]])
    # uv homogeneous coordinates on depth map images
    xv, yv = np.meshgrid(np.linspace(-1, 1, depth_map_size), np.linspace(-1, 1, depth_map_size))
    self.xv, self.yv = torch.from_numpy(xv).reshape(-1, 1).float(), torch.from_numpy(yv).reshape(-1, 1).float()

  def forward(self, x, cam_Rt, depth_maps, samplings=5):
    """
    Parameters
    ----------
    x: batch of rgb images
    cam_Rt: batch of camera roto-translation matrices, size (batch, 3, 4)
    depth_maps: batch of depth maps
    samplings: int, number of desired sampled points per patch
    """
    B_size = x.shape[0]

    ######
    ### Backproject each depth map in the batch, at each scale
    ######
    depth_grids = self.backprojection_depth(depth_maps, cam_Rt)

    ######
    ### Encode image as multiscale 2D feature maps l1, ..., l4
    ######
    x = self.encoder_2D.conv1(x)
    x = self.encoder_2D.bn1(x)
    x = self.encoder_2D.relu(x)
    x = self.encoder_2D.maxpool(x)

    l1 = self.encoder_2D.layer1(x)       # shape: batch, 64, 56, 56
    l2 = self.encoder_2D.layer2(l1)      # shape: batch, 128, 28, 28
    l3 = self.encoder_2D.layer3(l2)      # shape: batch, 256, 14, 14
    l4 = self.encoder_2D.layer4(l3)      # shape: batch, 512, 7, 7

    ######
    ### Backproject l1, ..., l4 to 3D and append binary depth grids
    ######
    l4 = self.conv1x1_l4(l4)          # shape: batch, n_2D_featuremaps - 2, 7, 7
    l4 = self.backprojection_features(l4, cam_Rt,
                         self.nb_cells_latent,
                         self.l4_grid_coord)    # shape: batch, n_2D_featuremaps - 2, 7, 7, 7
    l4 = torch.cat([l4,
                   depth_grids[0],            # voxelized depth maps are appended twice
                   depth_grids[0]], dim = 1)  # shape: batch, n_2D_featuremaps, 7, 7, 7

    l3 = self.conv1x1_l3(l3)          # shape: batch, 32 - 2, 14, 14
    l3 = self.backprojection_features(l3, cam_Rt,
                          self.nb_cells_latent*2,
                          self.l3_grid_coord)       # shape: batch, 32 - 2, 14, 14, 14
    l3 = torch.cat([l3,
                   depth_grids[1],
                   depth_grids[1]], dim = 1)  # shape: batch, 32, 14, 14, 14

    l2 = self.conv1x1_l2(l2)          # shape: batch, 32 - 2, 28, 28
    l2 = self.backprojection_features(l2, cam_Rt,
                          self.nb_cells_latent*4,
                          self.l2_l1_grid_coord)    # shape: batch, 32 - 2, 28, 28, 28
    l2 = torch.cat([l2,
                    depth_grids[2],
                    depth_grids[2]], dim=1)  # shape: batch, 32, 28, 28, 28

    l1 = self.conv1x1_l1(l1)          # shape: batch, 32 - 2, 56, 56
    l1 = self.backprojection_features(l1, cam_Rt,
                          self.nb_cells_latent * 4,
                          self.l2_l1_grid_coord)    # shape: batch, 32 - 2, 28, 28, 28
    l1 = torch.cat([l1,
                    depth_grids[2],
                    depth_grids[2]], dim=1)  # shape: batch, 32, 28, 28, 28

    ######
    ### Run 3D convolutions
    ######
    latent_grid = self.decoder_3D(l4, l3, l2, l1)

    ######
    ### Predict occupancy and folded patches
    ######
    # Split codes in occupancy and deformation parts
    # 1: Occupancy is straighforward
    occupancy = self.sigmoid(self.decoder_occupancy(latent_grid[:, :8, ...]))

    # 2: deformation codes for folding patches:
    # We filter grid of codes based on predicted occupancy
    # > only deformation codes for voxels with occupancy > self.thresh_occupancy
    # are kept and used for folding patches
    occupancy_flat = occupancy.detach().view(B_size, -1)                            # batch, grid**3
    deformation_codes = latent_grid[:, 8:, ...]
    deformations_flat = deformation_codes.permute(0,2,3,4,1).view(B_size, -1, 32)   # batch, grid**3, 32

    deformations_list = []
    xyz_list = []
    for k in range(B_size):
      indices = occupancy_flat[k] > self.thresh_occupancy
      deformations_list.append(deformations_flat[k][indices])
      xyz_list.append(self.xyz_flat[indices])

    # Pad tensors
    deformations_flat, _ = self.padding_pc(deformations_list)  # shapes batch, N_max, 32
    xyz_flat, mask_xyz = self.padding_pc(xyz_list)             # shapes batch, N_max, 3

    ## Fold Patches locally
    if deformations_flat.shape[1] > 0: # only if there exist some voxels with > occupancy
      offset = self.patch_sampler(deformations_flat, samplings)
    else:
      offset = torch.zeros(B_size, 0, samplings, 3).cuda()

    # Add 
    points = (xyz_flat.unsqueeze(2).repeat(1, 1, samplings, 1) + offset)

    return occupancy, points, mask_xyz[...,0]

  def backprojection_features(self, x, cam_Rt, cubesize, grid):
    """
    Backproject 2D feature maps x to 3D, for a single view per instance in the batch

    :param x: torch tensor of size (batch, #features, height, width)
    :param cam_Rt: torch tensor of size (batch, 3, 4)
    :param cubesize: s, vertex size of the cube to which we backproject
    :param grid: pre computed 3D coordinates of grid of size s*s*s
    :return: torch tensor of size (batch, #features, s, s, s)
    """
    batch_size = x.shape[0]
    n_features = x.shape[1]

    # Project 3D coordinates of the grid to 2D using camera matrix
    grid_coord_proj = UCLID_Net.project_3d_coordinates_to_image_plane(
      cam_Rt, grid, convert_back_to_euclidean=True
    )  # (batch, (s*s*s), 2)

    # Bilinear sampling of the 2D feature maps at these 2D locations
    grid_coord_proj = grid_coord_proj.unsqueeze(2)  # (batch, (s*s*s), 1, 2)
    volume = F.grid_sample(x, grid_coord_proj, #align_corners=True,
                                   mode='bilinear')  # (batch, n_2D_featuremaps, (s*s*s), 1)

    # Reshaping to cube
    volume = volume.view(batch_size, n_features,
                         cubesize, cubesize, cubesize)
    return volume

  def backprojection_depth(self, depth_maps, cam_Rt):
    """
    Backproject depth maps to 3D binary grids, according to camera matrices
    Each depth map is backprojected ad grid scales defined in self.grid_sizes

    :param depth_maps: torch tensor of size (batch, height, width)
    :param cam_Rt: torch tensor of size (batch, 4, 4)

    :return depth_grids: list of length len(grid_sizes).
      depth_grids[j]: tensor of size (batch, 1, grid_sizes[j], grid_sizes[j], grid_sizes[j])
      depth_grids[j][i]: binary depth grid of depth_map[i] at scale j
    """
    B_size = depth_maps.shape[0]
    device = depth_maps.device
    # Create empty grids for backprojection depth maps
    depth_grids = []
    for grid_size in self.grid_sizes:
      depth_grids.append(torch.zeros(B_size, 1, grid_size, grid_size, grid_size).cuda())

    # Sequentially Backproject each depth map in the batch, at each scale
    for i in range(B_size):
      ## 1: transform depth map into a point cloud
      # Compute inverse camera projection
      cam_hom = torch.cat([cam_Rt[i], torch.tensor([[0, 0, 0, 1.]]).to(cam_Rt.device)])
      cam_hom_inv = cam_hom.inverse()
      # Flatten depth values, and threshold them by masking out too high values
      dpt_z_flat = depth_maps[i].reshape(-1, 1)
      mask = dpt_z_flat < self.max_depth_value
      dpt_z_flat = dpt_z_flat[mask]
      # Concatenate uv positions with depth
      xv_flat = self.xv[mask]
      yv_flat = self.yv[mask]
      ones_flat = torch.ones_like(xv_flat).to(device)
      uvz_homogeneous = torch.stack([xv_flat * dpt_z_flat,
                                     yv_flat * dpt_z_flat,
                                     dpt_z_flat,
                                     ones_flat], dim=-1)
      # Backproject to 3D space
      xyz = ((torch.mm(cam_hom_inv, self.revert_axis) @ uvz_homogeneous.T).T)[:, :3]
      # Clean it by removing obvious outliers - only useful on predicted depth maps
      if xyz.shape[0] > 0:
        xyz = self.cleanPC(xyz)
      ## 2: grid-pooling - transform it into a binary voxel occupancy, at different scales
      for j, grid_size in enumerate(self.grid_sizes):
        # prepare target voxel grid - scale points up to fit voxel grid
        points_for_voxelization = ((grid_size + grid_size * xyz) / 2.0).unsqueeze(0)
        # create a fake feature vector for each point (composed of a single 1)
        indicator_points = torch.ones(1, points_for_voxelization.shape[1], 1).cuda()
        # max pooling on the grid: depth_vox has shape (1, 1, nb_cells, nb_cells, nb_cells)
        depth_vox = gridPooling(indicator_points, points_for_voxelization.contiguous(),
                                grid_size, grid_size, grid_size).float()
        depth_grids[j][i] = depth_vox.squeeze(0)
    
    return depth_grids

  def cuda(self, device=None):
    """
    Override the .cuda() method, to manually move the latent grid coord too
    """
    self = super().cuda(device)
    self.l4_grid_coord = self.l4_grid_coord.cuda(device)
    self.l3_grid_coord = self.l3_grid_coord.cuda(device)
    self.l2_l1_grid_coord = self.l2_l1_grid_coord.cuda(device)
    self.xyz_flat = self.xyz_flat.cuda(device)
    self.revert_axis = self.revert_axis.cuda(device)
    self.xv, self.yv = self.xv.cuda(device), self.yv.cuda(device)
    return self

  @staticmethod
  def padding_pc(tensor_sequence):
    """
    Padding a sequence of tensors of different sizes with zeros,
    and return a single padded tensor
    :param sequences: list of tensors
    :return: single padded tensor, and masks
    """
    num = len(tensor_sequence)
    max_len = max([s.size(0) for s in tensor_sequence])
    out_dims = (num, max_len, tensor_sequence[0].size(1))
    out_tensor = tensor_sequence[0].data.new(*out_dims).fill_(0.).cuda()
    mask = tensor_sequence[0].data.new(*out_dims).fill_(0).cuda()
    for i, tensor in enumerate(tensor_sequence):
      length = tensor.size(0)
      out_tensor[i, :length] = tensor
      mask[i, :length] = 1
    return out_tensor, mask.bool()

  @staticmethod
  def grid_coord_generator(size):
    """
    Get the normalized ([-1,1]) 3D grid coordinates of a cubic grid of size*size*size
    """
    xv, yv, zv = np.meshgrid(np.arange(0, size), np.arange(0, size), np.arange(0, size), indexing='ij')
    xyz = torch.tensor(np.stack((xv, yv, zv), axis=0), requires_grad=False).float()
    xyz = xyz.permute(1, 2, 3, 0)       # size, size, size, 3
    return (xyz * 2 / (size - 1)) - 1.  # Normalized bounding box

  @staticmethod
  def project_3d_coordinates_to_image_plane(proj_matrices, coords_3d, convert_back_to_euclidean=True):
    """Project 3D points to image plane
    Args:
        proj_matrix: torch tensor of shape (batch, 3, 4) (projection matrices)
        coords_3d: torch tensor of shape (N, 3) (3D coordinates)
        convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                        NOTE: division by zero can happen here, if z = 0
    Returns:
        torch tensor of shape (batch, N, 2): 3D points projected to image plane
                          (or (batch, N, 3) if convert_back_to_euclidean=False)
    """
    proj_matrices = proj_matrices.transpose(1,2)  # (batch, 4, 3)
    result = UCLID_Net.euclidean_to_homogeneous(coords_3d) @ proj_matrices
    if convert_back_to_euclidean:
      result = UCLID_Net.homogeneous_to_euclidean(result)
    return result

  @staticmethod
  def euclidean_to_homogeneous(coords):
    """Converts euclidean coordinates to homogeneous ones (by appending a 1 as last coordinate)
    Args:
        coords: torch tensor of shape (N, M) (or (batch, N, M+1) : N euclidean points of dimension M
                                                                  (or a batch of it)
    Returns:
        torch tensor of shape (N, M + 1) (or (batch, N, M)) : homogeneous coordinates
    """
    ones = torch.ones_like(coords)[...,0].unsqueeze(-1)
    return torch.cat([coords, ones], dim=-1)

  @staticmethod
  def homogeneous_to_euclidean(coords):
    """Converts homogeneous coordinates to euclidean ones (by dividing by last coordinate)
    Args:
        coords: torch tensor of shape (N, M+1) (or (batch, N, M+1) : N homogeneous coordinates of dimension M
                                                                    (or a batch of it)
    Returns:
        torch tensor of shape (N, M) (or (batch, N, M)) : euclidean coordinates
    """
    return coords[...,:-1] / coords[...,-1:]

  @staticmethod
  def cleanPC(xyz, k=5, dist_limit=2/28.):
    """
    Clean point cloud xyz by:
     - removing points outside of the bounding box
     - removing points whose k-th nearest neighbour is further than dist_limit
    Used for cleaning inferred depth maps point clouds
    """
    # Bounding-box criterion
    xyz = xyz[(xyz.max(dim=1)[0] <= 1) * (xyz.min(dim=1)[0] >= -1)]
    # Distance criterion
    if xyz.shape[0] > k:
      distance_matrix = torch.sqrt(xyz.pow(2).sum(1, keepdim = True) - 2 * torch.mm(xyz, xyz.t()) + xyz.pow(2).sum(1, keepdim = True).t())
      distance_matrix = torch.sort(distance_matrix, axis=1)[0]
      xyz = xyz[distance_matrix[:, k] < dist_limit]
    return xyz

