#include <stdio.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__constant__ float grid_size=1.0;


/**
 * perform max-pooling within the cells
 * parallel over each cell and each feature dimension
 */
__global__ void grid_pooling_kernel( const float *point, const float *feat_points, float *feat_cell, long int *indices, const int n  ){

  // cell indices
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  // cell size
  // int W = gridDim.x;
  int H = gridDim.y;
  int D = gridDim.z;
  int ind = i*H*D + j*D + k;

  int c = threadIdx.x;
  int C = blockDim.x;

  for (int p=0; p<n; p++)
  {
     float px = point[p*3+0];
     float py = point[p*3+1];
     float pz = point[p*3+2];
     // if point is inside of the grid
     if (px >= i && px < i+grid_size && py >= j && py < j+grid_size && pz >= k && pz < k+grid_size)
     {
	  // max-pooling, update feat_cell if the feature is larger than the current feat_cell
	  // can be async for max operation
	  if ( feat_points[p*C + c] > feat_cell[ind*C + c] )
    {
	     feat_cell[ind*C + c] = feat_points[p*C + c];
	     indices[ind*C + c] = p;
	  }
     }
  }
}


/**
 * back-propagate the loss from the max-pooled feature to point features
 * parallel over each cell and each feature dimension
 */
__global__ void grad_grid_pooling_kernel( const float *grad_output, const long int *indices, float *grad_feat_points  ){

  // cell indices
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  // cell size
  // int W = gridDim.x;
  int H = gridDim.y;
  int D = gridDim.z;
  int ind = i*H*D + j*D + k;

  int c = threadIdx.x;
  int C = blockDim.x;

  long int p = indices[ind*C + c];
  if (p < 0) return;

  grad_feat_points[p*C + c] = grad_output[ind*C + c];

}


/*
 * Forward function, project the point features to cells, perform max pooling in every cell
 * params:
 *  	  state 	input, THCState
 *  	  point 	input, all points, Nx3
 *  	  feat_points   input, feature of all points, NxC
 *  	  shape 	input, size of the grid [W, H, D], 3
 *  	  feat_cell     output, feature of all cells, (WxHxD)xC
 *  	  indices     	output, indices of max pooling, saved for back propagation, (WxHxD)xC
 *
 */
 int grid_pooling_cuda_forward(at::Tensor point, at::Tensor feat_points, at::Tensor shape, at::Tensor feat_cell, at::Tensor indices){

  const auto W = shape.data<int>()[0];
  const auto H = shape.data<int>()[1];
  const auto D = shape.data<int>()[2];

  const auto C = feat_points.size(1);
  const auto N = point.size(0);

  dim3 dimGrid(W, H, D);
  dim3 dimBlock(C, 1, 1);

 	grid_pooling_kernel<<<dimGrid,dimBlock>>>(point.data<float>(), feat_points.data<float>(), feat_cell.data<float>(), indices.data<long>(), N);

 	return 1;


 }

/*
 * Backward function, back-propagate the loss to the point features
 * params:
 *  	  state 	input, THCState
 *  	  grad_output   	input, gradient on the output feature, WxHxC
 *  	  shape 		input, size of the grid [W, H, D], 3
 *  	  indices     		input, indices of max pooling, WxHxC
 * 	  grad_feat_points 	output, gradient on the features, NxC
 *
 */

int grid_pooling_cuda_backward(at::Tensor grad_output, at::Tensor shape, at::Tensor indices, at::Tensor grad_feat_points)
{


  const auto W = shape.data<int>()[0];
  const auto H = shape.data<int>()[1];
  const auto D = shape.data<int>()[2];
  const auto C = grad_output.size(1);

  dim3 dimGrid(W, H, D);
  dim3 dimBlock(C, 1, 1);

	grad_grid_pooling_kernel<<<dimGrid, dimBlock>>>(grad_output.data<float>(),indices.data<long>(),grad_feat_points.data<float>());

	return 1;

}
