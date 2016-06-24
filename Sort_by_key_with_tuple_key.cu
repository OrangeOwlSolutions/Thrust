#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "Utilities.cuh"

// --- Defining tuple type
typedef thrust::tuple<int, int> Tuple;

/**************************/
/* TUPLE ORDERING FUNCTOR */
/**************************/
struct TupleComp
{
	__host__ __device__ bool operator()(const Tuple& t1, const Tuple& t2)
	{
		if (t1.get<0>() < t2.get<0>())
			return true;
		if (t1.get<0>() > t2.get<0>())
			return false;
		return t1.get<1>() < t2.get<1>();
	}
};

/********/
/* MAIN */
/********/
int main()
{
	const int N = 8;

	// --- Keys and values on the host: allocation and definition
	int h_keys1[N]		= { 1, 3, 3, 3, 2, 3, 2, 1 };                                         
	int h_keys2[N]		= { 1, 5, 3, 8, 2, 8, 1, 1 };                                         
	float h_values[N]	= { 0.3, 5.1, 3.2, -0.08, 2.1, 5.2, 1.1, 0.01};

	printf("\n\n");
	printf("Original\n");
	for (int i = 0; i < N; i++) {
		printf("%i %i %f\n", h_keys1[i], h_keys2[i], h_values[i]);
	}

	// --- Keys and values on the device: allocation
	int *d_keys1;		gpuErrchk(cudaMalloc(&d_keys1, N * sizeof(int)));
	int *d_keys2;		gpuErrchk(cudaMalloc(&d_keys2, N * sizeof(int)));
	float *d_values;	gpuErrchk(cudaMalloc(&d_values, N * sizeof(float)));

	// --- Keys and values: host -> device
	gpuErrchk(cudaMemcpy(d_keys1, h_keys1, N * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_keys2, h_keys2, N * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_values, h_values, N * sizeof(float), cudaMemcpyHostToDevice));

	// --- From raw pointers to device_ptr
	thrust::device_ptr<int> dev_ptr_keys1 = thrust::device_pointer_cast(d_keys1);
	thrust::device_ptr<int> dev_ptr_keys2 = thrust::device_pointer_cast(d_keys2);
	thrust::device_ptr<float> dev_ptr_values = thrust::device_pointer_cast(d_values);

	// --- Declare outputs
	thrust::device_vector<float> d_values_output(N);
	thrust::device_vector<Tuple> d_keys_output(N);

	auto begin_keys = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_keys1, dev_ptr_keys2));
	auto end_keys = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_keys1 + N, dev_ptr_keys2 + N));

	thrust::sort_by_key(begin_keys, end_keys, dev_ptr_values, TupleComp());

	int *h_keys1_output = (int *)malloc(N * sizeof(int));
	int *h_keys2_output = (int *)malloc(N * sizeof(int));
	float *h_values_output = (float *)malloc(N * sizeof(float));

	gpuErrchk(cudaMemcpy(h_keys1_output, d_keys1, N * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_keys2_output, d_keys2, N * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_values_output, d_values, N * sizeof(float), cudaMemcpyDeviceToHost));

	printf("\n\n");
	printf("Ordered\n");
	for (int i = 0; i < N; i++) {
		printf("%i %i %f\n", h_keys1_output[i], h_keys2_output[i], h_values_output[i]);
	}

}
