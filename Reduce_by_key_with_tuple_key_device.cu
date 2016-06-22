#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "Utilities.cuh"

// --- Defining key tuple type
typedef thrust::tuple<int, int> Tuple;

typedef thrust::device_vector<Tuple>::iterator  dIter1;
typedef thrust::device_vector<float>::iterator  dIter2;

/************************************/
/* EQUALITY OPERATOR BETWEEN TUPLES */
/************************************/
struct BinaryPredicate
{
	__host__ __device__ bool operator ()
		(const Tuple& lhs, const Tuple& rhs)
	{
		return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) && (thrust::get<1>(lhs) == thrust::get<1>(rhs));
	}
};

/********/
/* MAIN */
/********/
int main()
{
	const int N = 7;

	// --- Keys and input values on the host: allocation and definition
	int h_keys1_input[N] = { 1, 3, 3, 3, 2, 2, 1 };											// --- Input keys 1 - host side
	int h_keys2_input[N] = { 1, 5, 3, 8, 2, 2, 1 };											// --- Input keys 2 - host side
	float h_input_values[N] = { 9., 8., 7., 6., 5., 4., 3. };								// --- Input values - host side

	// --- Keys and input values on the device: allocation
	int *d_keys1_input;		gpuErrchk(cudaMalloc(&d_keys1_input, N * sizeof(int)));			// --- Input keys 1 - device side
	int *d_keys2_input;		gpuErrchk(cudaMalloc(&d_keys2_input, N * sizeof(int)));			// --- Input keys 2 - device side
	float *d_input_values;	gpuErrchk(cudaMalloc(&d_input_values, N * sizeof(float)));		// --- Input values - device side

	// --- Keys and input values: host -> device
	gpuErrchk(cudaMemcpy(d_keys1_input, h_keys1_input,	 N * sizeof(int),	cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_keys2_input, h_keys2_input,	 N * sizeof(int),	cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_input_values, h_input_values, N * sizeof(float), cudaMemcpyHostToDevice));

	// --- From raw pointers to device_ptr
	thrust::device_ptr<int> dev_ptr_keys1 = thrust::device_pointer_cast(d_keys1_input);
	thrust::device_ptr<int> dev_ptr_keys2 = thrust::device_pointer_cast(d_keys2_input);
	thrust::device_ptr<float> dev_ptr_values = thrust::device_pointer_cast(d_input_values);

	// --- Declare outputs
	thrust::device_vector<Tuple> d_keys_output(N);
	thrust::device_vector<float> d_values_output(N);

	thrust::pair<dIter1, dIter2> new_end;

	auto begin = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_keys1, dev_ptr_keys2));
	auto end   = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_keys1 + N, dev_ptr_keys2 + N));

	new_end = thrust::reduce_by_key(begin,
						            end,
									dev_ptr_values,
									d_keys_output.begin(),
									d_values_output.begin(),
									BinaryPredicate(),
									thrust::plus<float>());

	int Nkeys = new_end.first - d_keys_output.begin();

	printf("\n\n");
	for (int i = 0; i < Nkeys; i++) {
		float output = d_values_output[i];
		printf("%i; %f\n", i, output);
	}

	thrust::host_vector<Tuple> h_keys_output(d_keys_output);
	printf("\n\n");
	for (int i = 0; i < Nkeys; i++) {
		int key1 = thrust::get<0>(h_keys_output[i]);
		int key2 = thrust::get<1>(h_keys_output[i]);
		printf("%i %i\n", key1, key2);
	}

	return 0;
}
