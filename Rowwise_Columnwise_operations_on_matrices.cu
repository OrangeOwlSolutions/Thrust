#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/equal.h>

using namespace thrust::placeholders;

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
int main()
{
	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
  
	const int Nrows = 10;			// --- Number of rows
	const int Ncols =  3;			// --- Number of columns  

	// --- Random uniform integer distribution between 0 and 100
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist1(0, 100);

	// --- Random uniform integer distribution between 1 and 4
	thrust::uniform_int_distribution<int> dist2(1, 4);

	// --- Matrix allocation and initialization
	thrust::device_vector<float> d_matrix(Nrows * Ncols);
	for (size_t i = 0; i < d_matrix.size(); i++) d_matrix[i] = (float)dist1(rng);

	// --- Column vector allocation and initialization
	thrust::device_vector<float> d_column(Nrows);
	for (size_t i = 0; i < d_column.size(); i++) d_column[i] = (float)dist2(rng);

	// --- Row vector allocation and initialization
	thrust::device_vector<float> d_row(Ncols);
	for (size_t i = 0; i < d_row.size(); i++) d_row[i] = (float)dist2(rng);

	printf("\n\nOriginal matrix\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "]\n";
	}

	printf("\n\nColumn vector\n");
	for(int i = 0; i < Nrows; i++) std::cout << d_column[i] << "\n";

	printf("\n\nRow vector\n");
	for(int i = 0; i < Ncols; i++) std::cout << d_row[i] << " ";

	/*******************************************************/
	/* ADDING THE SAME COLUMN VECTOR TO ALL MATRIX COLUMNS */
	/*******************************************************/

	thrust::device_vector<float> d_matrix2(d_matrix);

	thrust::transform(d_matrix.begin(), d_matrix.end(),
					  thrust::make_permutation_iterator(
								d_column.begin(),
								thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols))),
					  d_matrix2.begin(),
					  thrust::plus<float>());
	
	printf("\n\nColumn + Matrix -> Result matrix\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix2[i * Ncols + j] << " ";
		std::cout << "]\n";
	}

	/*************************************************/
	/* ADDING THE SAME ROW VECTOR TO ALL MATRIX ROWS */
	/*************************************************/

	thrust::device_vector<float> d_matrix3(d_matrix);

	thrust::transform(thrust::make_permutation_iterator(
								d_matrix.begin(),
								thrust::make_transform_iterator(thrust::make_counting_iterator(0),(_1 % Nrows) * Ncols + _1 / Nrows)), 
					  thrust::make_permutation_iterator(
								d_matrix.begin(),
								thrust::make_transform_iterator(thrust::make_counting_iterator(0),(_1 % Nrows) * Ncols + _1 / Nrows)) + Nrows * Ncols, 					  
								thrust::make_permutation_iterator(
									d_row.begin(),
									thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Nrows))),
					  thrust::make_permutation_iterator(
								d_matrix3.begin(),
								thrust::make_transform_iterator(thrust::make_counting_iterator(0),(_1 % Nrows) * Ncols + _1 / Nrows)), 
					  thrust::plus<float>());

								
	printf("\n\nRow + Matrix -> Result matrix\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix3[i * Ncols + j] << " ";
		std::cout << "]\n";
	}

	return 0; 
}
