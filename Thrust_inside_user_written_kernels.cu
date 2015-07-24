#include <stdio.h>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "TimingGPU.cuh"
#include "Utilities.cuh"

#define BLOCKSIZE_1D	256
#define BLOCKSIZE_2D_X	32
#define BLOCKSIZE_2D_Y	32

/*************************/
/* TEST KERNEL FUNCTIONS */
/*************************/
__global__ void test1(const float * __restrict__ d_data, float * __restrict__ d_results, const int Nrows, const int Ncols) {

	const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < Nrows) d_results[tid] = thrust::reduce(thrust::seq, d_data + tid * Ncols, d_data + (tid + 1) * Ncols);

}

__global__ void test2(const float * __restrict__ d_data, float * __restrict__ d_results, const int Nrows, const int Ncols) {

	const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < Nrows) d_results[tid] = thrust::reduce(thrust::device, d_data + tid * Ncols, d_data + (tid + 1) * Ncols);

}

/********/
/* MAIN */
/********/
int main() {

    const int Nrows = 64;
    const int Ncols = 2048;

    gpuErrchk(cudaFree(0));

//    size_t DevQueue;
//    gpuErrchk(cudaDeviceGetLimit(&DevQueue, cudaLimitDevRuntimePendingLaunchCount));
//    DevQueue *= 128;
//    gpuErrchk(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, DevQueue));

    float *h_data		= (float *)malloc(Nrows * Ncols * sizeof(float));
	float *h_results	= (float *)malloc(Nrows *         sizeof(float));
	float *h_results1	= (float *)malloc(Nrows *         sizeof(float));
	float *h_results2	= (float *)malloc(Nrows *         sizeof(float));
	float sum = 0.f;
    for (int i=0; i<Nrows; i++) {
		h_results[i] = 0.f;
	    for (int j=0; j<Ncols; j++) {
			h_data[i*Ncols+j] = i;
			h_results[i] = h_results[i] + h_data[i*Ncols+j];
		}
	}

	TimingGPU timerGPU;

	float *d_data;			gpuErrchk(cudaMalloc((void**)&d_data,	  Nrows * Ncols * sizeof(float)));
    float *d_results1;		gpuErrchk(cudaMalloc((void**)&d_results1, Nrows         * sizeof(float)));
    float *d_results2;		gpuErrchk(cudaMalloc((void**)&d_results2, Nrows         * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_data, h_data, Nrows * Ncols * sizeof(float), cudaMemcpyHostToDevice));

	timerGPU.StartCounter();
	test1<<<iDivUp(Nrows, BLOCKSIZE_1D), BLOCKSIZE_1D>>>(d_data, d_results1, Nrows, Ncols);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	printf("Timing approach nr. 1 = %f\n", timerGPU.GetCounter());

	gpuErrchk(cudaMemcpy(h_results1, d_results1, Nrows * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i=0; i<Nrows; i++) {
		if (h_results1[i] != h_results[i]) {
			printf("Approach nr. 1; Error at i = %i; h_results1 = %f; h_results = %f", i, h_results1[i], h_results[i]);
			return 0;
		}
	}

	timerGPU.StartCounter();
	test2<<<iDivUp(Nrows, BLOCKSIZE_1D), BLOCKSIZE_1D>>>(d_data, d_results1, Nrows, Ncols);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	printf("Timing approach nr. 2 = %f\n", timerGPU.GetCounter());

	gpuErrchk(cudaMemcpy(h_results1, d_results1, Nrows * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i=0; i<Nrows; i++) {
		if (h_results1[i] != h_results[i]) {
			printf("Approach nr. 2; Error at i = %i; h_results1 = %f; h_results = %f", i, h_results1[i], h_results[i]);
			return 0;
		}
	}

    printf("Test passed!\n");

}
