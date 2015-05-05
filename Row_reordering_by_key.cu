#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include <iterator>
#include <iostream>

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

/*************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX */
/*************************************/
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
	
	T Ncols; // --- Number of columns
  
	__host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}

	__host__ __device__ T operator()(T i) { return i / Ncols; }
};

/********/
/* MAIN */
/********/
int main(void) {
	
	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
  
	const int Nrows = 5;     // --- Number of rows
	const int Ncols = 8;     // --- Number of columns
  
	// --- Random uniform integer distribution between 10 and 99
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist(10, 99);

	// --- Matrix allocation and initialization
	thrust::device_vector<float> d_matrix(Nrows * Ncols);
	for (size_t i = 0; i < d_matrix.size(); i++) d_matrix[i] = (float)dist(rng);

	// --- Setting up the row memberships
	thrust::uniform_int_distribution<int> dist_membership(0, Nrows - 1);
	thrust::device_vector<float> d_membership(Nrows);
	for (size_t i = 0; i < d_membership.size(); i++) d_membership[i] = (float)dist_membership(rng);

	printf("Row memberships\n");
	for (int i=0; i<Nrows; i++) std::cout << d_membership[i] << "\n";

	/**************************/
	/* EXPAND THE MEMBERSHIPS */
	/**************************/
	thrust::device_vector<int> d_membership_expanded(Nrows * Ncols);
	thrust::device_vector<int> d_counts(Nrows, Ncols);
	expand(d_counts.begin(), d_counts.end(), d_membership.begin(), d_membership_expanded.begin());

	// --- Print result
	printf("\n\nExpanded memberships\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_membership_expanded[i * Ncols + j] << " ";
		std::cout << "] \n";
	}

	/****************************************************/
	/* REORDERING THE ROWS ACCORDING TO THE MEMBERSHIPS */
	/****************************************************/
	printf("\n\nOriginal matrix\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "] \n";
	}

	thrust::stable_sort_by_key(d_membership_expanded.begin(), d_membership_expanded.end(), d_matrix.begin(), thrust::less<int>());

	printf("\n\nReordered matrix\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "] \n";
	}
	
	return 0;

}
