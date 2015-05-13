#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/equal.h>

/********/
/* MAIN */
/********/
int main()
{
	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
  
	const int N = 20;			// --- Number of elements
  
	// --- Random uniform integer distribution between 0 and 4
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist(0, 4);

	// --- Keys allocation and initialization
	thrust::device_vector<int> d_keys(N);
	for (size_t i = 0; i < d_keys.size(); i++) d_keys[i] = dist(rng);

	/********************/
	/* SORTING THE KEYS */
	/********************/

	thrust::device_vector<int> d_values(N, 1);
	thrust::sort(d_keys.begin(), d_keys.end());

	printf("Sorted keys\n");
	for (int i=0; i<N; i++) std::cout << d_keys[i] << "\n";
	printf("\n");

	/**********************************************/
	/* FINDING THE FIRST KEY OCCURRENCE POSITIONS */
	/**********************************************/

	thrust::device_vector<int> d_positions(N);
	thrust::device_vector<int> d_keys2(d_keys);
	thrust::sequence(d_positions.begin(), d_positions.end());
	
	thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
	new_end = thrust::unique_by_key(d_keys2.begin(), d_keys2.end(), d_positions.begin());
	
	for (int i=0; i<(new_end.first-d_keys2.begin()); i++) std::cout << d_keys[i] << " " << d_positions[i] << "\n";
	printf("\n");

	/*******************************************/
	/* FINDING THE KEY OCCURRENCES - HISTOGRAM */
	/*******************************************/

	new_end = thrust::reduce_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), d_keys.begin(), d_values.begin());
	
	printf("Keys\n");
	for (int i=0; i<(new_end.first - d_keys.begin()); i++) std::cout << d_keys[i] << " " << d_values[i] << "\n";
	printf("\n");
		

	return 0;
}
