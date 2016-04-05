#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// Add your kernel here
__global__ void add(int *a, int *b, int *c) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	c[index] = a[index] + b[index];
}

// main
#define N (2048*2048)
#define THREADS_PER_BLOCK 512
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
	add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

	// Copy result back to Host
	// Take note that it will be smart enough to wait
	// until the task at device completed
	cudaMemcpy (c, d_c, size, cudaMemcpyDeviceToHost);

	// Display the outcome
	for(i=N-100;i<N;i++) {
		printf("[%d]\t%2d + %2d = %2d\n", i, a[i], b[i], c[i]);
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
