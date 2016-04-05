#include <stdlib.h>
#include <stdio.h>

// --------------------------------
// Expected Output
//
//  0  0  1  1  2  2  3  3
//  0  0  1  1  2  2  3  3
//  4  4  5  5  6  6  7  7
//  4  4  5  5  6  6  7  7
//  8  8  9  9 10 10 11 11
//  8  8  9  9 10 10 11 11
// 12 12 13 13 14 14 15 15
// 12 12 13 13 14 14 15 15
//

// Kernal compute the two dimensional index of this particular
// thread in the grid
//
// TODO: Fill up the ???
__global__ void launch2DGrid(int *array)
{

  // TODO: computate x dimension:
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO: computate y dimension:
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // TODO: use the two 2D indices to compute the position
  int pindex = y * gridDim.x * blockDim.x + x;

  // TODO: use the two 2D block indices to compute a single 
  //       linear block index of value
  int pvalue = blockIdx.y * gridDim.x + blockIdx.x;

  // TODO: write out the result
  array[pindex] = pvalue;
}


int main(void)
{
  // TODO: Set total element of column
  int total_elements_in_column = 8;
  // TODO: Set total element of row
  int total_elements_in_row = 8;

  // TODO: Set total number of input bytes to be allocated 
  int num_bytes = total_elements_in_column * total_elements_in_row * sizeof(int);

  // TODO: Create a variable to store the device result
  int *device_array = 0;
  // TODO: Create a variable to store the host result
  int *host_array = 0;

  // TODO: Allocate the host memory variable for CPU 
  host_array = (int*)malloc(num_bytes);

  // TODO: Allocate the device memory variable for GPU 
  cudaMalloc((void**)&device_array, num_bytes);

  // Validate memory allocation failed
  if(host_array == 0 || device_array == 0)
  {
    printf("Failed to allocate memory\n");
    return 1;
  }

  // TODO: Create a 2x2 dimensional of thread blocks
  dim3 block_size(2,2);

  // TODO: Create a two dimensional grid
  dim3 grid_size;
  
  // TODO: Configure total of element of columns per block size of x
  grid_size.x = total_elements_in_column / block_size.x;
  
  // TODO: Configure total of element of rows per block size of y
  grid_size.y = total_elements_in_row / block_size.y;

  // TODO: Launch the kernel function
  launch2DGrid<<<grid_size,block_size>>>(device_array);

  // TODO: Copy the result from GPU to CPU
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  // Display the result element by element
  for(int row = 0; row < total_elements_in_row; ++row)
  {
    for(int col = 0; col < total_elements_in_column; ++col)
    {
      printf("%2d ", host_array[row * total_elements_in_column + col]);
    }
    printf("\n");
  }
  printf("\n");

  // TODO: Free the host memory
  free(host_array);
  
  // TODO: Free the device memory
  cudaFree(device_array);
  
  return 0;
}