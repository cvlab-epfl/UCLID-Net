import torch
from torch import nn
from torch.autograd import Function
import grid_pooling


# Grid pooling module in CUDA, credits @edoRemelli
# GPU tensors only

class gridFunction(Function):
    @staticmethod
    def forward(ctx, feat_points, points, grid_shape):

        W = grid_shape[0]
        H = grid_shape[1]
        D = grid_shape[2]
        C = feat_points.size()[1]

        feat_cells = torch.zeros(W*H*D, C).type(torch.FloatTensor).cuda()
        indices = -1 * torch.ones(W*H*D, C).type(torch.LongTensor).cuda()

        grid_pooling.forward(points, feat_points, grid_shape, feat_cells, indices)

        ctx.save_for_backward(indices)
        ctx.N = points.size()[0]
        ctx.C = C
        ctx.grid_shape = grid_shape

        return feat_cells

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors[0]

        N  = ctx.N
        C  = ctx.C
        grid_shape = ctx.grid_shape

        grad_features = torch.zeros(N, C).type(torch.FloatTensor).cuda()

        grid_pooling.backward(grad_output, grid_shape, indices, grad_features)

        return grad_features, None, None

class gridPooling(nn.Module):
    def __init__(self):
        super(gridPooling, self).__init__()

    def forward(self, feat_points, points, W,H,D):

        assert feat_points.shape[0] == points.shape[0]
        assert feat_points.shape[1] == points.shape[1]

        batchsize = feat_points.size()[0]
        C = feat_points.size()[2]

        feat_cell = torch.zeros(batchsize, W*H*D, C).float().cuda()
        grid_shape = torch.IntTensor([W, H, D])

        for k in range(batchsize):
            feat_cell[k, :, :] = gridFunction.apply(feat_points[k, :, :], points[k, :, :], grid_shape)

        feat_cell = torch.transpose(feat_cell, 1, 2).contiguous().view(-1, C, W, H, D)
        return feat_cell
