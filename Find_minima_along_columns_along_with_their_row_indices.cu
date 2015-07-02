#include <iterator>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

using namespace thrust::placeholders;

int main()
{
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
    thrust::device_vector<float> d_minima(Ncols);
    thrust::device_vector<int> d_indices(Ncols);

    thrust::reduce_by_key(
            thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), _1 / Nrows),
            thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), _1 / Nrows) + Nrows * Ncols,
            thrust::make_zip_iterator(
					thrust::make_tuple(thrust::make_permutation_iterator(
                                    d_matrix.begin(),
                                    thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), (_1 % Nrows) * Ncols + _1 / Nrows)),
            thrust::make_transform_iterator(thrust::make_counting_iterator((int) 0), _1 % Nrows))),
            thrust::make_discard_iterator(),
            thrust::make_zip_iterator(thrust::make_tuple(d_minima.begin(), d_indices.begin())),
            thrust::equal_to<int>(),
            thrust::minimum<thrust::tuple<float, int> >()
    );

	printf("\n\n");
	for (int i=0; i<Nrows; i++) std::cout << "Min position = " << d_indices[i] << "; Min value = " << d_minima[i] << "\n";

	return 0;
}
