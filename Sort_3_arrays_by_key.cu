#include <time.h>       // --- time
#include <stdlib.h>     // --- srand, rand

#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\sort.h>
#include <thrust\iterator\zip_iterator.h>

#include "TimingGPU.cuh"

//#define VERBOSE
//#define COMPACT

int main() {

	const int N = 1048576;
	//const int N = 10;

	TimingGPU timerGPU;

	// --- Initialize random seed
	srand(time(NULL));

	thrust::host_vector<int> h_code(N);
	thrust::host_vector<double> h_x(N);
	thrust::host_vector<double> h_y(N);
	thrust::host_vector<double> h_z(N);

	for (int k = 0; k < N; k++) {
		// --- Generate random numbers between 0 and 9
		h_code[k] = rand() % 10 + 1;
		h_x[k] = ((double)rand() / (RAND_MAX));
		h_y[k] = ((double)rand() / (RAND_MAX));
		h_z[k] = ((double)rand() / (RAND_MAX));
	}

	thrust::device_vector<int> d_code(h_code);

	thrust::device_vector<double> d_x(h_x);
	thrust::device_vector<double> d_y(h_y);
	thrust::device_vector<double> d_z(h_z);

#ifdef VERBOSE
	printf("Before\n");
	for (int k = 0; k < N; k++) printf("code = %i; x = %f; y = %f\n", h_code[k], h_x[k], h_y[k]);
#endif

	timerGPU.StartCounter();
#ifdef COMPACT
	thrust::sort_by_key(d_code.begin(), d_code.end(), thrust::make_zip_iterator(thrust::make_tuple(d_x.begin(), d_y.begin(), d_z.begin())));
#else

	// --- Initialize indices vector to [0,1,2,..]
	thrust::counting_iterator<int> iter(0);
	thrust::device_vector<int> indices(N);
	thrust::copy(iter, iter + indices.size(), indices.begin());

	// --- First, sort the keys and indices by the keys
	thrust::sort_by_key(d_code.begin(), d_code.end(), indices.begin());

	// Now reorder the ID arrays using the sorted indices
	thrust::gather(indices.begin(), indices.end(), d_x.begin(), d_x.begin());
	thrust::gather(indices.begin(), indices.end(), d_y.begin(), d_y.begin());
	thrust::gather(indices.begin(), indices.end(), d_z.begin(), d_z.begin());
#endif

	printf("Timing GPU = %f\n", timerGPU.GetCounter());

#ifdef VERBOSE
	h_code = d_code;
	h_x = d_x;
	h_y = d_y;

	printf("After\n");
	for (int k = 0; k < N; k++) printf("code = %i; x = %f; y = %f\n", h_code[k], h_x[k], h_y[k]);
#endif
}
