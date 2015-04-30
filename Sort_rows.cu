#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

#include <stdio.h>
#include <iostream>

#include "Utilities.cuh"

/**************************************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX - NEEDED FOR APPROACH #1 */
/**************************************************************/
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
	
	T Ncols; // --- Number of columns
  
	__host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}

	__host__ __device__ T operator()(T i) { return i / Ncols; }
};

/********/
/* MAIN */
/********/
int main()
{
	const int Nrows = 5;     // --- Number of rows
	const int Ncols = 8;     // --- Number of columns
  
	// --- Random uniform integer distribution between 10 and 99
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist(10, 99);

	// --- Matrix allocation and initialization
	thrust::device_vector<float> d_matrix(Nrows * Ncols);
	for (size_t i = 0; i < d_matrix.size(); i++) d_matrix[i] = (float)dist(rng);
  
	// --- Print result
	printf("Original matrix\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "]\n";
	}

	/*************************/
	/* BACK-TO-BACK APPROACH */
	/*************************/
	thrust::device_vector<float> d_keys(Nrows * Ncols);

	// --- Generate row indices
	thrust::transform(thrust::make_counting_iterator(0),
				      thrust::make_counting_iterator(Nrows*Ncols),
					  thrust::make_constant_iterator(Ncols),
					  d_keys.begin(),
					  thrust::divides<int>());

	// --- Back-to-back approach
	thrust::stable_sort_by_key(d_matrix.begin(),
						       d_matrix.end(),
						       d_keys.begin(),
						       thrust::less<float>());

	thrust::stable_sort_by_key(d_keys.begin(),
						       d_keys.end(),
						       d_matrix.begin(),
						       thrust::less<int>());

	// --- Print result
	printf("\n\nSorted matrix\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "]\n";
	}

	return 0;
}
