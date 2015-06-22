#include <thrust\device_vector.h>
#include <thrust\transform_reduce.h>
#include <thrust/random.h>

struct square { __host__ __device__ float operator()(float x) { return x * x; } };

/********/
/* MAIN */
/********/
int main()
{
	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
  
	const int N = 20;			// --- Number of rows of matrix 1

	// --- Random uniform integer distribution between 0 and 100
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist(0, 20);

	// --- Matrix allocation and initialization
	thrust::device_vector<float> d_vec(N);
	for (size_t i = 0; i < d_vec.size(); i++) d_vec[i] = (float)dist(rng);

	/************************/
	/* CALCULATING THE NORM */
	/************************/

	float reduction = sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), square(), 0.0f, thrust::plus<float>()));

	for (int i=0; i<N; i++) std::cout << d_vec[i] << "\n";
	printf("Reduction result =  %f\n", reduction);

	return 0; 
}
