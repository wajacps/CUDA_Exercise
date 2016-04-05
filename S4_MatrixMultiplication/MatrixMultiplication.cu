// This example demonstrates the use of shared per-block arrays
// implement an optimized dense matrix multiplication algorithm.
// Like the shared_variables.cu example, a per-block __shared__
// array acts as a "bandwidth multiplier" by eliminating redundant
// loads issued by neighboring threads.

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_WIDTH 16

// a simple version of matrix_multiply which issues redundant loads from off-chip global memory
__global__ void matrix_multiply_simple(int *a, int *b, int *ab, size_t width)
{
  // calculate the row & column index of the element

  // do dot product between row of a and column of b

  // write out this thread's result

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

  // to get accurate timings, launch a single "warm-up" kernel
  matrix_multiply_simple<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);
  
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
