#include <stdio.h>

#include <cuda_runtime_api.h>

#include <thrust\pair.h>
#include <thrust\device_vector.h>
#include <thrust\extrema.h>

int main()
{
    const int N = 5;
    
	const float h_a[N] = { 3., 21., -2., 4., 5. };

	float *d_a;		cudaMalloc(&d_a, N * sizeof(float));
	cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

	float minel, maxel;
	thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float>> tuple;
	tuple = thrust::minmax_element(thrust::device_pointer_cast(d_a), thrust::device_pointer_cast(d_a) + N);
	minel = tuple.first[0];
	maxel = tuple.second[0];

	printf("minelement %f - maxelement %f\n", minel, maxel);

	return 0;
}
