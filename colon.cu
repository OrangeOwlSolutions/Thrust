#include <cstdio>

#include "Matlab_like.cuh"
#include "Utilities.cuh"

/********/
/* MAIN */
/********/
int main()
{
	float a  = 4.11f;
	float b  = 7.01f;
	float step = 0.5f;

	int N = (int)((b - a)/step) + 1;

	float *d_vec = colon(a, step, b);

	float *h_vec = (float *)malloc(N * sizeof(float));

	gpuErrchk(cudaMemcpy(h_vec, d_vec, N * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++) printf("%i %f\n", i, h_vec[i]); 

	return 0;

}
