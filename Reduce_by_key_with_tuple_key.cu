#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// --- Defining key tuple type
typedef thrust::tuple<int,int> Tuple;

typedef thrust::host_vector<Tuple>::iterator  dIter1;
typedef thrust::host_vector<float>::iterator  dIter2;

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
	
	thrust::host_vector<Tuple> keys_input(N);
	thrust::host_vector<float> values_input(N);
	
	int keys1_input[N]			= {1, 3, 3, 3, 2, 2, 1};			// input keys 1
	int keys2_input[N]			= {1, 5, 3, 8, 2, 2, 1};			// input keys 2
	float input_values[N]		= {9., 8., 7., 6., 5., 4., 3.};		// input values

	for (int i=0; i<N; i++) {
		keys_input[i] = thrust::make_tuple(keys1_input[i], keys2_input[i]);
		values_input[i] = input_values[i];
	}

	for (int i=0; i<N; i++) printf("%i %i\n", thrust::get<0>(keys_input[i]), thrust::get<1>(keys_input[i]));

	thrust::host_vector<Tuple> keys_output(N);
	thrust::host_vector<float> values_output(N);
	
	thrust::pair<dIter1, dIter2> new_end;

	new_end = thrust::reduce_by_key(keys_input.begin(), 
						            keys_input.end(), 
						            values_input.begin(), 
						            keys_output.begin(), 
						            values_output.begin(), 
						            BinaryPredicate(),
						            thrust::plus<float>());

	int Nkeys = new_end.first - keys_output.begin();
	
	printf("\n\n");
	for (int i = 0; i < Nkeys; i++) printf("%i; %f\n", i, values_output[i]);

	printf("\n\n");
	for (int i = 0; i < Nkeys; i++) printf("%i %i\n", thrust::get<0>(keys_output[i]), thrust::get<1>(keys_output[i]));

	return 0;
}
