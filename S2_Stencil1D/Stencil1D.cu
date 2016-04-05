#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// Add your kernel here
#define TOTAL_DATA 256
#define BLOCK_PER_THREAT 64
#define BLOCK_SIZE 32
#define RADIUS 3

__global__ void stensil_1d (int *in, int *out){
	__shared__ int temp[BLOCK_SIZE + 2*RADIUS];
	int gIdx = threadIdx.x + (blockIdx.x*blockDim.x);	// Global Index
	int lIdx = threadIdx.x + RADIUS;			// Local Index

	// Read input elements into shared memory
	temp[lIdx] = in[gIdx];
	if(threadIdx.x < RADIUS) {
		temp[lIdx - RADIUS] = in[gIdx - RADIUS];
		temp[lIdx + BLOCK_SIZE] = in[gIdx + BLOCK_SIZE];
	}


	// Make sure all the threads are syncronized
	__syncthreads();


	// Apply the stencil
	int result = 0;
	for(int offset = -RADIUS; offset <= RADIUS; offset++) {
		result += temp[lIdx + offset];
	}
	
	// Store the output
	out[gIdx] = result;
}


// main
int main(void)
{
	int *a, *b;
	int *d_a, *d_b;
	int size = TOTAL_DATA * sizeof(int);
	int i;
	
	// Allocate memory for the Host
	a = (int *) malloc (size);
	b = (int *) malloc (size);

	// Allocate memory for the Device
	cudaMalloc ((void **) &d_a, size);
	cudaMalloc ((void **) &d_b, size);

	// Initialize data (0 - 9)
	for(i=0; i<TOTAL_DATA;i++) {
		a[i] = rand() % 10;
	}	

	// Copy the data to 
	cudaMemcpy (a, d_a, size, cudaMemcpyHostToDevice);

	// Lets execute it
	stensil_1d<<<TOTAL_DATA/BLOCK_SIZE, BLOCK_SIZE>>> (d_a, d_b);

	cudaMemcpy (b, d_b, size, cudaMemcpyDeviceToHost);

	// Print the outcome
	int j;
	for(i=0;i<TOTAL_DATA;i++) {
		printf("[%3d]\t", i);
		for(j=0;j<2*RADIUS + 1;j++) printf("%d,", a[i+j]);
		printf("\t--> %d\n", b[i]);
	}


	cudaFree (d_a);
	cudaFree (d_b);
	
	free (a);
	free (b);

	cudaGetDeviceCount (&j);
	printf("Total Device = %d\n", j);

	return 0;
}
