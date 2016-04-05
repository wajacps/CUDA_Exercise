
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_WIDTH 16

// an optimized version of matrix_multiplication which eliminates redundant loads
__global__ void matrix_multiply(int *d_M, int *d_N, int *d_P, size_t width)
{
  // create shorthand names for threadIdx & blockIdx
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x,  by = blockIdx.y;

  // allocate 2D tiles in __shared__ memory
  __shared__ int s_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ int s_N[TILE_WIDTH][TILE_WIDTH];

  // calculate the row & column index of the element
  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  int result = 0;

  // loop over the tiles of the input in phases
  for(int p = 0; p < width/TILE_WIDTH; ++p)
  {
    // collaboratively load tiles into __shared__
    s_M[ty][tx] = d_M[row*width + (p*TILE_WIDTH + tx)];
    s_N[ty][tx] = d_N[(p*TILE_WIDTH + ty)*width + col];

    // wait until all data is loaded before allowing
    // any thread in this block to continue
    __syncthreads();

    // do dot product between row of s_a and column of s_b
    for(int k = 0; k < TILE_WIDTH; ++k)
    {
      result += s_M[ty][k] * s_N[k][tx];
    }

    // wait until all threads are finished with the data
    // before allowing any thread in this block to continue
    __syncthreads();
  }

  // write out this thread's result
  d_P[row*width+col] = result;
}

void MatrixMulOnHost(int* M, int* N, int* P, int Width)
{
	for (int i = 0; i < Width; ++i) {
		for (int j = 0; j < Width; ++j) {
			double sum = 0;
            for (int k = 0; k < Width; ++k) {
                double a = M[i * Width + k];
                double b = N[k * Width + j];
                sum += a * b;
            }
            P[i * Width + j] = sum;
		}
	}
}

int main(void)
{
  // create a large workload so we can easily measure the
  // performance difference of both implementations

  // note that n measures the width of the matrix, not the number of total elements
  //const size_t n = 1<<10;
  const size_t n = 1024;
  std::cout << "Total element is " << n << "\n";
  const dim3 block_size(TILE_WIDTH,TILE_WIDTH);
  const dim3 num_blocks(n / block_size.x, n / block_size.y);

  // generate random input on the host  
  std::vector<int> h_a(n*n), h_b(n*n), h_c(n*n);
  
  for(int i = 0; i < n*n; ++i)
  {
    h_a[i] = static_cast<int>(rand()) / RAND_MAX;
    h_b[i] = static_cast<int>(rand()) / RAND_MAX;
  }

  // allocate storage for the device
  int *d_a = 0, *d_b = 0, *d_c = 0;
  cudaMalloc((void**)&d_a, sizeof(int) * n * n);
  cudaMalloc((void**)&d_b, sizeof(int) * n * n);
  cudaMalloc((void**)&d_c, sizeof(int) * n * n);

  // copy input to the device
  cudaMemcpy(d_a, &h_a[0], sizeof(int) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &h_b[0], sizeof(int) * n * n, cudaMemcpyHostToDevice);

  // again, launch a single "warm-up" kernel
  matrix_multiply<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);
  
  // copy result back to the host
  cudaMemcpy(&h_c[0], d_c, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
  
  //------------------
  int* h_r;
  h_r = (int*)malloc(sizeof(int) * n * n);
  MatrixMulOnHost(&h_a[0], &h_b[0], h_r, n);
  
  for (int i=0; i<(n*n); i++) {
	if (h_r[i] != h_c[i]) {
		std::cout << "Failed at i " << i << "h_r=" << h_r[i] << ",h_c=" << h_c[i] << "\n";
		exit(1);
	}
  }
  
  std::cout << "Result is correct.";
  
  // deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
