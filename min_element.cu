#include "cuda_runtime.h"
#include "device_launch_paraMeters.h"

#include <thrust\device_vector.h>
#include <thrust/extrema.h>

/***********************/
/* CUDA ERROR CHECKING */
/***********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/********/
/* MAIN */
/********/
int main() {

	srand(time(NULL));
	
	const int N = 10;

	float *h_vec = (float *)malloc(N * sizeof(float));
	for (int i=0; i<N; i++) {
		h_vec[i] = rand() / (float)(RAND_MAX);
		printf("h_vec[%i] = %f\n", i, h_vec[i]);
	}
	
	float *d_vec; gpuErrchk(cudaMalloc((void**)&d_vec, N * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_vec);

	thrust::device_ptr<float> min_ptr = thrust::min_element(dev_ptr, dev_ptr + N);
	
	float min_value = min_ptr[0];
	printf("\nMininum value = %f\n", min_value);
	printf("Position = %i\n", &min_ptr[0] - &dev_ptr[0]);

}
