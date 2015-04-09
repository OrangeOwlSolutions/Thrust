#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <iostream>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

// --- Required for approach #2
__device__ int *vals;
__device__ int *keys;

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
struct row_reduction {

    const int Ncols;    // --- Number of columns

    row_reduction(int _Ncols) : Ncols(_Ncols) {}

    __device__ int operator()(int& x, int& y ) {
        int temp = 0;
        for (int i = 0; i<Ncols; i++)
            temp += vals[i + (y*Ncols)] * keys[i];
        return temp;
    }
};

/********/
/* MAIN */
/********/
int main()
{
    const int Nrows = 10;     // --- Number of rows
    const int Ncols = 100000;     // --- Number of columns

    // --- Random uniform integer distribution between 10 and 99
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(10, 99);

    // --- Matrix allocation and initialization
    thrust::device_vector<int> d_matrix(Nrows * Ncols);
    for (size_t i = 0; i < d_matrix.size(); i++) d_matrix[i] = dist(rng);

    TimingGPU timerGPU;

    /***************/
    /* APPROACH #1 */
    /***************/
    timerGPU.StartCounter();
    // --- Allocate space for row sums and indices
    thrust::device_vector<int> d_row_sums(Nrows);
    thrust::device_vector<int> d_row_indices(Nrows);

    // --- Compute row sums by summing values with equal row indices
    thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                          thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                          d_matrix.begin(),
                          d_row_indices.begin(),
                          d_row_sums.begin(),
                          thrust::equal_to<int>(),
                          thrust::plus<int>());

    printf("Timing for approach #1 = %f\n", timerGPU.GetCounter());

    // --- Print result
    //for(int i = 0; i < Nrows; i++) {
    //  std::cout << "[ ";
    //  for(int j = 0; j < Ncols; j++)
    //      std::cout << d_matrix[i * Ncols + j] << " ";
    //  std::cout << "] = " << d_row_sums[i] << "\n";
    //}

    /***************/
    /* APPROACH #2 */
    /***************/
    timerGPU.StartCounter();
    thrust::device_vector<int> d_keys(Ncols);
    thrust::fill(d_keys.begin(), d_keys.end(), 1);
    thrust::fill(d_row_sums.begin(), d_row_sums.end(), 0);
    int *s_vals = thrust::raw_pointer_cast(&d_matrix[0]);
    int *s_keys = thrust::raw_pointer_cast(&d_keys[0]);
    gpuErrchk(cudaMemcpyToSymbol(vals, &s_vals, sizeof(int *)));
    gpuErrchk(cudaMemcpyToSymbol(keys, &s_keys, sizeof(int *)));
    thrust::sequence(d_row_indices.begin(), d_row_indices.end());
    thrust::transform(d_row_sums.begin(), d_row_sums.end(), d_row_indices.begin(),  d_row_sums.begin(), row_reduction(Ncols));

    printf("Timing for approach #2 = %f\n", timerGPU.GetCounter());

    //for(int i = 0; i < Nrows; i++) {
    //  std::cout << "[ ";
    //  for(int j = 0; j < Ncols; j++)
    //      std::cout << d_matrix[i * Ncols + j] << " ";
    //  std::cout << "] = " << d_row_sums[i] << "\n";
    //}

    return 0;
}
