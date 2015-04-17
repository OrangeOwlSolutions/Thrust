#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

#include <stdio.h>
#include <iostream>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

using namespace thrust::placeholders;

// --- Required for approach #2
__device__ float *vals;

/**************************************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX - NEEDED FOR APPROACH #1 */
/**************************************************************/
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
	
	T Ncols; // --- Number of columns
  
	__host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}

	__host__ __device__ T operator()(T i) { return i / Ncols; }
};

/******************************************/
/* ROW_REDUCTION - NEEDED FOR APPROACH #2 */
/******************************************/
struct col_reduction {
  
	const int Nrows;	// --- Number of rows
	const int Ncols;	// --- Number of cols

	col_reduction(int _Nrows, int _Ncols) : Nrows(_Nrows), Ncols(_Ncols) {}

	__device__ float operator()(float& x, int& y ) {
		float temp = 0.f;
		for (int i = 0; i<Nrows; i++) {
			temp += vals[y + (i*Ncols)];
		}
		return temp;
	}
};

/**************************/
/* NEEDED FOR APPROACH #3 */
/**************************/
template<typename T>
struct MulC: public thrust::unary_function<T, T>
{
    T C;
    __host__ __device__ MulC(T c) : C(c) { }
    __host__ __device__ T operator()(T x) { return x * C; }
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
  
	TimingGPU timerGPU;

	/***************/
	/* APPROACH #1 */
	/***************/
	timerGPU.StartCounter();
	// --- Allocate space for row sums and indices
	thrust::device_vector<float> d_col_sums(Ncols);
	thrust::device_vector<int> d_col_indices(Ncols);
  
	// --- Compute row sums by summing values with equal row indices
	thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Nrows)),
						  thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Nrows)) + (Nrows*Ncols),
						  thrust::make_permutation_iterator(
								d_matrix.begin(),
								thrust::make_transform_iterator(thrust::make_counting_iterator(0),(_1 % Nrows) * Ncols + _1 / Nrows)),
						  d_col_indices.begin(),
						  d_col_sums.begin(),
						  thrust::equal_to<int>(),
						  thrust::plus<float>());

	//thrust::reduce_by_key(
 //               thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Nrows)),
 //               thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Nrows)) + (Nrows*Ncols),
 //               thrust::make_permutation_iterator(
	//				d_matrix.begin(),
	//				thrust::make_transform_iterator(thrust::make_counting_iterator(0),(_1 % Nrows) * Ncols + _1 / Nrows)),
 //               thrust::make_discard_iterator(),
 //               d_col_sums.begin());
	
	printf("Timing for approach #1 = %f\n", timerGPU.GetCounter());
	
	// --- Print result
	for(int j = 0; j < Ncols; j++) {
		std::cout << "[ ";
		for(int i = 0; i < Nrows; i++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "] = " << d_col_sums[j] << "\n";
	}

	/***************/
	/* APPROACH #2 */
	/***************/
	timerGPU.StartCounter();
	thrust::device_vector<float> d_col_sums_2(Ncols, 0);
	float *s_vals = thrust::raw_pointer_cast(&d_matrix[0]);
	gpuErrchk(cudaMemcpyToSymbol(vals, &s_vals, sizeof(float *)));
	thrust::transform(d_col_sums_2.begin(), d_col_sums_2.end(), thrust::counting_iterator<int>(0), d_col_sums_2.begin(), col_reduction(Nrows, Ncols));

	printf("Timing for approach #2 = %f\n", timerGPU.GetCounter());

	for(int j = 0; j < Ncols; j++) {
		std::cout << "[ ";
		for(int i = 0; i < Nrows; i++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "] = " << d_col_sums_2[j] << "\n";
	}

	/***************/
	/* APPROACH #3 */
	/***************/

	timerGPU.StartCounter();
	thrust::device_vector<float> d_col_sums_3(Ncols, 0);
	thrust::device_vector<float> d_temp(Nrows * Ncols);
	thrust::inclusive_scan_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Nrows)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Nrows)) + (Nrows*Ncols),
  			    thrust::make_permutation_iterator(
						d_matrix.begin(),
						thrust::make_transform_iterator(thrust::make_counting_iterator(0),(_1 % Nrows) * Ncols + _1 / Nrows)),
                d_temp.begin());
    thrust::copy(
                thrust::make_permutation_iterator(
                        d_temp.begin() + Nrows - 1,
                        thrust::make_transform_iterator(thrust::make_counting_iterator(0), MulC<int>(Nrows))),
				thrust::make_permutation_iterator(
                        d_temp.begin() + Nrows - 1,
                        thrust::make_transform_iterator(thrust::make_counting_iterator(0), MulC<int>(Nrows))) + Ncols,
                d_col_sums_3.begin());
		
	printf("Timing for approach #3 = %f\n", timerGPU.GetCounter());

	for(int j = 0; j < Ncols; j++) {
		std::cout << "[ ";
		for(int i = 0; i < Nrows; i++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "] = " << d_col_sums_3[j] << "\n";
	}

	/***************/
	/* APPROACH #4 */
	/***************/
	cublasHandle_t handle;

	timerGPU.StartCounter();
	cublasSafeCall(cublasCreate(&handle));

	thrust::device_vector<float> d_col_sums_4(Ncols);
	thrust::device_vector<float> d_ones(Nrows, 1.f);

	float alpha = 1.f;
	float beta  = 0.f;
	cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, Ncols, Nrows, &alpha, thrust::raw_pointer_cast(d_matrix.data()), Ncols, 
		                       thrust::raw_pointer_cast(d_ones.data()), 1, &beta, thrust::raw_pointer_cast(d_col_sums_4.data()), 1));

	printf("Timing for approach #4 = %f\n", timerGPU.GetCounter());

	for(int j = 0; j < Ncols; j++) {
		std::cout << "[ ";
		for(int i = 0; i < Nrows; i++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "] = " << d_col_sums_4[j] << "\n";
	}

	return 0;
}
