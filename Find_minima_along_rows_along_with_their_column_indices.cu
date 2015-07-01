#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <iostream>
 
/*************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX */
/*************************************/
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
	
	T Ncols; // --- Number of columns
  
	__host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}

	__host__ __device__ T operator()(T i) { return i / Ncols; }
};
 
typedef thrust::tuple<int,float> MinAndIndexType;
 
/****************************************/
/* CONVERT LINEAR INDEX TO COLUMN INDEX */
/****************************************/
struct linear_index_to_column_index {
	
	int Ncols; // --- Number of columns
  
	__host__ __device__ linear_index_to_column_index(int Ncols) : Ncols(Ncols) {}

	__host__ __device__ MinAndIndexType operator()(MinAndIndexType a) { thrust::get<0>(a) = thrust::get<0>(a) % Ncols; return a; }
};

/********************************************/
/* COMPARISON OPERATOR FOR INDEX/VALUE TYPE */
/********************************************/
struct MinAndIndex: public thrust::binary_function<MinAndIndexType, MinAndIndexType, MinAndIndexType> { 

	__host__ __device__ MinAndIndexType operator()(const MinAndIndexType &a, const MinAndIndexType &b) const {
		
		if (thrust::get<1>(a) < thrust::get<1>(b))  { return a; } 
		else										{ return b; }
	}
};
 
/********/
/* MAIN */
/********/
int main(void) {

	const int Nrows = 6;
    const int Ncols = 8;
 
	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/

	// --- Random uniform integer distribution between 0 and 100
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist(0, 20);

	// --- Matrix allocation and initialization
	thrust::device_vector<double> d_matrix(Nrows * Ncols);
	for (size_t i = 0; i < d_matrix.size(); i++) d_matrix[i] = (double)dist(rng);

	printf("\n\nMatrix\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << " [ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "]\n";
	}

	/**********************************************/
	/* FIND ROW MINIMA ALONG WITH THEIR LOCATIONS */
	/**********************************************/
	thrust::device_vector<MinAndIndexType> d_MinsAndIndices(Nrows);
	thrust::device_vector<int> d_indices(Nrows);          

	thrust::reduce_by_key(
					thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
					thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows * Ncols),
					thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), d_matrix.begin())),
					d_indices.begin(),
					d_MinsAndIndices.begin(),
					thrust::equal_to<int>(),
					MinAndIndex());
 
	thrust::transform(d_MinsAndIndices.begin(), d_MinsAndIndices.end(), d_MinsAndIndices.begin(), linear_index_to_column_index(Ncols));

	for (int i=0; i<Nrows; i++) {
		MinAndIndexType tmp = d_MinsAndIndices[i];
		std::cout << "Min position = " << thrust::get<0>(tmp) << "; Min value = " << thrust::get<1>(tmp) << "\n";
	}

	return 0;
}
