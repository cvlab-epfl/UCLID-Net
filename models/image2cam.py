import torch
import torch.nn as nn
from torch.nn import InstanceNorm2d
import functools
import math
import models.resnet

def normalize_vector(v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

def cross_product( u, v):
    batch = u.shape[0]
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)

    return out

def compute_rotation_matrix_from_ortho6d(ortho6d):
  x_raw = ortho6d[:,0:3]
  y_raw = ortho6d[:,3:6]

  x = normalize_vector(x_raw) #batch*3
  z = cross_product(x,y_raw) #batch*3
  z = normalize_vector(z)#batch*3
  y = cross_product(z,x)#batch*3

  x = x.view(-1,3,1)
  y = y.view(-1,3,1)
  z = z.view(-1,3,1)
  matrix = torch.cat((x,y,z), 2) #batch*3*3

  return matrix


class image2cam(nn.Module):
  def __init__(self, depth=26):
    super(image2cam, self).__init__()
    assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
    n = (depth - 2) // 6

    models.resnet.INSTANCENORM_AFFINE = True
    block = models.resnet.BasicBlock
    self.normlayer = functools.partial(InstanceNorm2d, affine=True)

    self.inplanes = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1,
                            bias=False)
    self.bn1 = self.normlayer(64)
    self.relu = nn.ReLU(inplace=True)
    self.layer1 = self._make_layer(block, 64, n)
    self.layer2 = self._make_layer(block, 128, n, stride=2)
    self.layer3 = self._make_layer(block, 256, n, stride=2)
    self.layer4 = self._make_layer(block, 256, n, stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    self.fc = nn.Linear(256 * block.expansion, 7)

    self.sigmoid = nn.Sigmoid()

    self.max_distance = 1.75

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      #elif isinstance(m, self.normlayer.func):
      #  import pdb
      #  pdb.set_trace()
      #  m.weight.data.fill_(1)
      #  m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
          self.normlayer(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    feature_1 = self.conv1(x)
    feature_1 = self.bn1(feature_1)
    feature_1 = self.relu(feature_1)
    feature_2 = self.layer1(feature_1)
    feature_3 = self.layer2(feature_2)
    feature_4 = self.layer3(feature_3)
    feature_out = self.layer4(feature_4)

    x = self.avgpool(feature_out)
    x = x.view(x.size(0), -1)
    global_feature = self.fc(x)

    R = compute_rotation_matrix_from_ortho6d(global_feature[:,0:6])
    t_z = self.sigmoid(global_feature[:,6:]).unsqueeze(-1)*self.max_distance
    t = torch.cat( (torch.zeros(R.shape[0],1,2).cuda(), t_z), 2)

    return R, t
