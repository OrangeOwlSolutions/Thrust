#include <cstdio>

#include "Matlab_like.cuh"
#include "Utilities.cuh"

/********/
/* MAIN */
/********/
int main()
{
	const int N = 20;

	float a = 3.87f;
	float b = 7.11f;

	float *h_arr = (float *)malloc(N * sizeof(float));
	float *d_arr = linspace(a, b, N);

	gpuErrchk(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));
 
	for(int i = 0; i < N; i++) printf("%f\n", h_arr[i]);

	return 0;

}
