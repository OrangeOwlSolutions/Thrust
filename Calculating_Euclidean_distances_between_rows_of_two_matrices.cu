#include <thrust\device_vector.h>
#include <thrust\transform_reduce.h>
#include <thrust\sequence.h>
#include <thrust\random.h>
#include <thrust\gather.h>
#include <thrust\extrema.h>

using namespace thrust::placeholders;

/****************************************************/
/* POWER DIFFERENCE FUNCTOR FOR EUCLIDEAN DISTANCES */
/****************************************************/
struct PowerDifference {
	__host__ __device__ float operator()(const float& a, const float& b) const { return pow(a - b, 2); }
};

/*******************/
/* EXPAND OPERATOR */
/*******************/
template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator expand(InputIterator1 first1,
                      InputIterator1 last1,
                      InputIterator2 first2,
                      OutputIterator output)
{
	typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;
  
	difference_type input_size  = thrust::distance(first1, last1);
	difference_type output_size = thrust::reduce(first1, last1);

	// scan the counts to obtain output offsets for each input element
	thrust::device_vector<difference_type> output_offsets(input_size, 0);
	thrust::exclusive_scan(first1, last1, output_offsets.begin()); 

	// scatter the nonzero counts into their corresponding output positions
	thrust::device_vector<difference_type> output_indices(output_size, 0);
	thrust::scatter_if(thrust::counting_iterator<difference_type>(0), thrust::counting_iterator<difference_type>(input_size),
					   output_offsets.begin(), first1, output_indices.begin());

	// compute max-scan over the output indices, filling in the holes
	thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin(), thrust::maximum<difference_type>());

	// gather input values according to index array (output = first2[output_indices])
	OutputIterator output_end = output; thrust::advance(output_end, output_size);
	thrust::gather(output_indices.begin(), output_indices.end(), first2, output);

	// return output + output_size
	thrust::advance(output, output_size);
  
	return output;
}

/********/
/* MAIN */
/********/
int main()
{
	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
  
	const int N		= 10;			// --- Number of vector elements
	const int Nvec	= 20;			// --- Number of vectors for each matrix

	// --- Random uniform integer distribution between 0 and 100
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist(0, 20);

	// --- Matrix allocation and initialization
	thrust::device_vector<float> d_matrix1(Nvec * N);
	thrust::device_vector<float> d_matrix2(Nvec * N);
	for (size_t i = 0; i < d_matrix1.size(); i++) d_matrix1[i] = (float)dist(rng);
	for (size_t i = 0; i < d_matrix2.size(); i++) d_matrix2[i] = (float)dist(rng);

	printf("\n\nFirst matrix\n");
	for(int i = 0; i < Nvec; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N; j++)
			std::cout << d_matrix1[i * N + j] << " ";
		std::cout << "]\n";
	}

	printf("\n\nSecond matrix\n");
	for(int i = 0; i < Nvec; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N; j++)
			std::cout << d_matrix2[i * N + j] << " ";
		std::cout << "]\n";
	}

	/****************************************************************************/
	/* CALCULATING THE EUCLIDEAN DISTANCES BETWEEN THE ROWS OF THE TWO MATRICES */
	/****************************************************************************/
	// --- Creating the indices for the reduction by key
	thrust::device_vector<int> d_sequence(Nvec);
	thrust::device_vector<int> d_indices(Nvec * N);
	thrust::device_vector<int> d_counts(Nvec, N);
	thrust::sequence(d_sequence.begin(), d_sequence.begin() + Nvec);
	expand(d_counts.begin(), d_counts.end(), d_sequence.begin(), d_indices.begin());

	printf("\n\nSecond matrix\n");
	for(int i = 0; i < Nvec; i++) {
		std::cout << " [ ";
		for(int j = 0; j < N; j++)
			std::cout << d_indices[i * N + j] << " ";
		std::cout << "]\n";
	}

	thrust::device_vector<float> d_squared_differences(Nvec * N);

	thrust::transform(d_matrix1.begin(), d_matrix1.end(), d_matrix2.begin(), d_squared_differences.begin(), PowerDifference());

	thrust::device_vector<float> d_norms(Nvec);
	thrust::reduce_by_key(d_indices.begin(), d_indices.end(), d_squared_differences.begin(), d_indices.begin(), d_norms.begin());
	
	printf("\n\ndnorms\n");
	for(int i = 0; i < Nvec; i++) {
			std::cout << d_norms[i] << " ";
	}

	return 0; 
}
