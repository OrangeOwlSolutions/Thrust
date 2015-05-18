#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/equal.h>

#include <cublas_v2.h>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

/**************************************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX - NEEDED FOR APPROACH #1 */
/**************************************************************/
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
	
	T Ncols; // --- Number of columns
  
	__host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}

	__host__ __device__ T operator()(T i) { return i / Ncols; }
};

/***********************/
/* RECIPROCAL OPERATOR */
/***********************/
struct Inv: public thrust::unary_function<float, float>
{
    __host__ __device__ float operator()(float x)
    {
        return 1.0f / x;
    }
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

	// --- Normalization vector allocation and initialization
	thrust::device_vector<float> d_normalization(Nrows);
	for (size_t i = 0; i < d_normalization.size(); i++) d_normalization[i] = (float)dist2(rng);

	printf("\n\nOriginal matrix\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "]\n";
	}

	printf("\n\nNormlization vector\n");
	for(int i = 0; i < Nrows; i++) std::cout << d_normalization[i] << "\n";

	TimingGPU timerGPU;
	
	/*********************************/
	/* ROW NORMALIZATION WITH THRUST */
	/*********************************/

	thrust::device_vector<float> d_matrix2(d_matrix);

	timerGPU.StartCounter();
	thrust::transform(d_matrix2.begin(), d_matrix2.end(),
					  thrust::make_permutation_iterator(
								d_normalization.begin(),
								thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols))),
					  d_matrix2.begin(),
					  thrust::divides<float>());
	std::cout << "Timing - Thrust = " << timerGPU.GetCounter() << "\n";
	
	printf("\n\nNormalized matrix - Thrust case\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix2[i * Ncols + j] << " ";
		std::cout << "]\n";
	}

	/*********************************/
	/* ROW NORMALIZATION WITH CUBLAS */
	/*********************************/
	d_matrix2 = d_matrix;

	cublasHandle_t handle;
	cublasSafeCall(cublasCreate(&handle));

	timerGPU.StartCounter();
	thrust::transform(d_normalization.begin(), d_normalization.end(), d_normalization.begin(), Inv());
	cublasSafeCall(cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, Ncols, Nrows, thrust::raw_pointer_cast(&d_matrix2[0]), Ncols, 
		           thrust::raw_pointer_cast(&d_normalization[0]), 1, thrust::raw_pointer_cast(&d_matrix2[0]), Ncols));
	std::cout << "Timing - cuBLAS = " << timerGPU.GetCounter() << "\n";

	printf("\n\nNormalized matrix - cuBLAS case\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix2[i * Ncols + j] << " ";
		std::cout << "]\n";
	}

	return 0;
}
