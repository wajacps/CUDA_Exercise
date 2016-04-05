#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// Add your kernel here
__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}

// main
int main(void)
{
	int a, b, c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);

	// Allocate memory in Device
	cudaMalloc ((void **) &d_a, size);
	cudaMalloc ((void **) &d_b, size);
	cudaMalloc ((void **) &d_c, size);

	// Initialize value
	a = 2;
	b = 7;

	// Copy data from Host to Device
	cudaMemcpy (d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy (d_b, &b, size, cudaMemcpyHostToDevice);

	// Execute
	add<<<1,1>>>(d_a, d_b, d_c);

	// Copy result back to Host
	// Take note that it will be smart enough to wait
	// until the task at device completed
	cudaMemcpy (&c, d_c, size, cudaMemcpyDeviceToHost);

	// Clean up
	cudaFree (d_a);
	cudaFree (d_b);
	cudaFree (d_c);

	printf("Task Completed: c = %d + %d = %d\n" ,a, b, c);

	return 0;
}
