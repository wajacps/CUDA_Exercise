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

  
  // TODO: computate y dimension:
  
  
  // TODO: use the two 2D indices to compute the position
  
  
  // TODO: use the two 2D block indices to compute a single 
  //       linear block index of value
  
  
  // TODO: write out the result
  
}


int main(void)
{
  // TODO: Set total element of column


  // TODO: Set total element of row

  
  // TODO: Set total number of input bytes to be allocated 
  
  
  // TODO: Create a variable to store the device result
  
  // TODO: Create a variable to store the host result
  
  
  // TODO: Allocate the host memory variable for CPU 
  
  
  // TODO: Allocate the device memory variable for GPU 
  
  
  // Validate memory allocation failed
  if(host_array == 0 || device_array == 0)
  {
    printf("Failed to allocate memory\n");
    return 1;
  }

  // TODO: Create a 2x2 dimensional of thread blocks
  
  
  // TODO: Create a two dimensional grid
  
  
  // TODO: Configure total of element of columns per block size of x
  
  
  // TODO: Configure total of element of rows per block size of y
  
  
  // TODO: Launch the kernel function
  
  
  // TODO: Copy the result from GPU to CPU
  
  
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
  
  
  // TODO: Free the device memory
  
  return 0;
 }