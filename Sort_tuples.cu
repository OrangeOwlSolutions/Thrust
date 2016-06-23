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

	// --- Vector components on the host: allocation and definition
	int h_vector1_input[N] = { 1, 3, 3, 3, 2, 3, 2, 1 };                                         
	int h_vector2_input[N] = { 1, 5, 3, 8, 2, 8, 1, 1 };                                         

	printf("\n\n");
	printf("Original\n");
	for (int i = 0; i < N; i++) {
		printf("%i %i\n", h_vector1_input[i], h_vector2_input[i]);
	}

	// --- Vector components on the device: allocation
	int *d_vector1_input;     gpuErrchk(cudaMalloc(&d_vector1_input, N * sizeof(int)));
	int *d_vector2_input;     gpuErrchk(cudaMalloc(&d_vector2_input, N * sizeof(int)));

	// --- Vector components: host -> device
	gpuErrchk(cudaMemcpy(d_vector1_input, h_vector1_input, N * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_vector2_input, h_vector2_input, N * sizeof(int), cudaMemcpyHostToDevice));

	// --- From raw pointers to device_ptr
	thrust::device_ptr<int> dev_ptr_vector1 = thrust::device_pointer_cast(d_vector1_input);
	thrust::device_ptr<int> dev_ptr_vector2 = thrust::device_pointer_cast(d_vector2_input);

	// --- Declare output
	thrust::device_vector<Tuple> d_values_output(N);

	auto begin = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_vector1, dev_ptr_vector2));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_vector1 + N, dev_ptr_vector2 + N));

	thrust::sort(begin, end, TupleComp());

	int *h_vector1_output = (int *)malloc(N * sizeof(int));
	int *h_vector2_output = (int *)malloc(N * sizeof(int));

	gpuErrchk(cudaMemcpy(h_vector1_output, d_vector1_input, N * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_vector2_output, d_vector2_input, N * sizeof(int), cudaMemcpyDeviceToHost));

	printf("\n\n");
	printf("Ordered\n");
	for (int i = 0; i < N; i++) {
		printf("%i %i\n", h_vector1_output[i], h_vector2_output[i]);
	}

}
