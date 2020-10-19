#include <torch/torch.h>
#include <vector>
#include <stdio.h>

///TMP
//#include "common.h"
/// NOT TMP


int grid_pooling_cuda_forward(at::Tensor point, at::Tensor feat_points, at::Tensor shape, at::Tensor feat_cell, at::Tensor indices);


int grid_pooling_cuda_backward(at::Tensor grad_output, at::Tensor shape, at::Tensor indices, at::Tensor grad_feat_points);


int grid_pooling_forward(at::Tensor point, at::Tensor feat_points, at::Tensor shape, at::Tensor feat_cell, at::Tensor indices) {
    return grid_pooling_cuda_forward(point, feat_points, shape, feat_cell, indices);
}

int grid_pooling_backward(at::Tensor grad_output, at::Tensor shape, at::Tensor indices, at::Tensor grad_feat_points) {
    return grid_pooling_cuda_backward(grad_output, shape, indices, grad_feat_points );
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &grid_pooling_forward, "grid pooling forward (CUDA)");
  m.def("backward", &grid_pooling_backward, "grid pooling backward (CUDA)");
}
