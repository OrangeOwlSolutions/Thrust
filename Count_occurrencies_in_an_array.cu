#include <time.h>       // --- time
#include <stdlib.h>     // --- srand, rand
#include <iostream>

#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\sort.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust\unique.h>
#include <thrust/binary_search.h>
#include <thrust\adjacent_difference.h>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

//#define VERBOSE
#define NO_HISTOGRAM

/********/
/* MAIN */
/********/
int main() {

	const int N = 1048576;
	//const int N = 20;
	//const int N = 128;

	TimingGPU timerGPU;

	// --- Initialize random seed
	srand(time(NULL));

	thrust::host_vector<int> h_code(N);

	for (int k = 0; k < N; k++) {
		// --- Generate random numbers between 0 and 9
		h_code[k] = (rand() % 10);
	}

	thrust::device_vector<int> d_code(h_code);
	//thrust::device_vector<unsigned int> d_counting(N);

	thrust::sort(d_code.begin(), d_code.end());

	h_code = d_code;
	
	timerGPU.StartCounter();

#ifdef NO_HISTOGRAM
	// --- The number of d_cumsum bins is equal to the maximum value plus one
	int num_bins = d_code.back() + 1;

	thrust::device_vector<int> d_code_unique(num_bins);
	thrust::unique_copy(d_code.begin(), d_code.end(), d_code_unique.begin());
	thrust::device_vector<int> d_counting(num_bins);
	thrust::upper_bound(d_code.begin(), d_code.end(), d_code_unique.begin(), d_code_unique.end(), d_counting.begin());	
#else
	thrust::device_vector<int> d_cumsum;

	// --- The number of d_cumsum bins is equal to the maximum value plus one
	int num_bins = d_code.back() + 1;

	// --- Resize d_cumsum storage
	d_cumsum.resize(num_bins);

	// --- Find the end of each bin of values - Cumulative d_cumsum
	thrust::counting_iterator<int> search_begin(0);
	thrust::upper_bound(d_code.begin(), d_code.end(), search_begin, search_begin + num_bins, d_cumsum.begin());

	// --- Compute the histogram by taking differences of the cumulative d_cumsum
	//thrust::device_vector<int> d_counting(num_bins);
	//thrust::adjacent_difference(d_cumsum.begin(), d_cumsum.end(), d_counting.begin());
#endif

	printf("Timing GPU = %f\n", timerGPU.GetCounter());

#ifdef VERBOSE
	thrust::host_vector<int> h_counting(d_counting);
	printf("After\n");
	for (int k = 0; k < N; k++) printf("code = %i\n", h_code[k]);
#ifndef NO_HISTOGRAM
	thrust::host_vector<int> h_cumsum(d_cumsum);
	printf("\nCounting\n");
	for (int k = 0; k < num_bins; k++) printf("element = %i; counting = %i; cumsum = %i\n", k, h_counting[k], h_cumsum[k]);
#else
	thrust::host_vector<int> h_code_unique(d_code_unique);

	printf("\nCounting\n");
	for (int k = 0; k < N; k++) printf("element = %i; counting = %i\n", h_code_unique[k], h_counting[k]);
#endif
#endif
}
