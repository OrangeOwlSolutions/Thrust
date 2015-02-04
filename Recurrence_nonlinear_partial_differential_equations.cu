#include <thrust\device_vector.h>

struct Recurrence{
template <typename Tuple>
	__host__ __device__ double operator()(Tuple a) {

		// --- X[n] = A[n]*X[n-1] + B[n]*X[n-2] + C[n]
		return (thrust::get<0>(a) * thrust::get<3>(a) + thrust::get<1>(a) * thrust::get<4>(a) + thrust::get<2>(a));

	}
};


int main() {

	const int N = 10;

	thrust::device_vector<double> d_x(N, 1.f);
	thrust::device_vector<double> d_a(N, 2.f);
	thrust::device_vector<double> d_b(N, 3.f);
	thrust::device_vector<double> d_c(N, 4.f);

	thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_a.begin() + 2, d_b.begin() + 2, d_c.begin() + 2, d_x.begin() + 1,     d_x.begin())), 
		              thrust::make_zip_iterator(thrust::make_tuple(d_a.begin() + N, d_b.begin() + N, d_c.begin() + N, d_x.begin() + N - 1, d_x.begin() + N - 2)), 
					  d_x.begin() + 2, Recurrence());
	
	for (int i=2; i<N; i++) {
		double temp = d_x[i];
		printf("%i %f\n", i, temp);
	}

	return 0;
}
