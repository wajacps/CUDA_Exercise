#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// Add your kernel here
__global__ void add(int *a, int *b, int *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// main
#define N 512
int main(void)
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);
	int i;

	// Allocate memory in Host
	a = (int *) malloc (size);
	b = (int *) malloc (size);
	c = (int *) malloc (size);

	// Allocate memory in Device
	cudaMalloc ((void **) &d_a, size);
	cudaMalloc ((void **) &d_b, size);
	cudaMalloc ((void **) &d_c, size);

	// Initialize values (0 - 9)
	for(i = 0;i < N; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}


	// Copy data from Host to Device
	cudaMemcpy (d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy (d_b, b, size, cudaMemcpyHostToDevice);

	// Execute
	add<<<N,1>>>(d_a, d_b, d_c);

	// Copy result back to Host
	// Take note that it will be smart enough to wait
	// until the task at device completed
	cudaMemcpy (c, d_c, size, cudaMemcpyDeviceToHost);

	// Display the outcome
	for(i=0;i<N;i++) {
		printf("[%3d]\t%2d + %2d = %2d\n", i, a[i], b[i], c[i]);
	}

	// Clean up at Host
	free (a);
	free (b);
	free (c);

	// Clean up at Device
	cudaFree (d_a);
	cudaFree (d_b);
	cudaFree (d_c);


	return 0;
}
