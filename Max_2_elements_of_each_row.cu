#include <iterator>
#include <algorithm>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};


/**************************************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX - NEEDED FOR APPROACH #1 */
/**************************************************************/
template< typename T >
struct mod_functor {
    __host__ __device__ T operator()(T a, T b) { return a % b; }
};

/********/
/* MAIN */
/********/
int main()
{
    /***********************/
    /* SETTING THE PROBLEM */
    /***********************/
    const int Nrows = 4;
    const int Ncols = 6;

    // --- Random uniform integer distribution between 10 and 99
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(10, 99);

    // --- Matrix allocation and initialization
    thrust::device_vector<float> d_matrix(Nrows * Ncols);
    for (size_t i = 0; i < d_matrix.size(); i++) d_matrix[i] = (float)dist(rng);

	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "]\n";
	}

	/******************/
    /* APPROACH NR. 2 */
    /******************/
    // --- Computing row indices vector
    thrust::device_vector<int> d_row_indices(Nrows * Ncols);
    thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(Nrows * Ncols), thrust::make_constant_iterator(Ncols), d_row_indices.begin(), thrust::divides<int>() );

    // --- Computing column indices vector
    thrust::device_vector<int> d_column_indices(Nrows * Ncols);
    thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(Nrows * Ncols), thrust::make_constant_iterator(Ncols), d_column_indices.begin(), mod_functor<int>());

    // --- int and float iterators
    typedef thrust::device_vector<int>::iterator        IntIterator;
    typedef thrust::device_vector<float>::iterator      FloatIterator;

    // --- Relevant tuples of int and float iterators
    typedef thrust::tuple<IntIterator, IntIterator>     IteratorTuple1;
    typedef thrust::tuple<FloatIterator, IntIterator>   IteratorTuple2;

    // --- zip_iterator of the relevant tuples
    typedef thrust::zip_iterator<IteratorTuple1>        ZipIterator1;
    typedef thrust::zip_iterator<IteratorTuple2>        ZipIterator2;

    // --- zip_iterator creation
    ZipIterator1 iter1(thrust::make_tuple(d_row_indices.begin(), d_column_indices.begin()));

    thrust::stable_sort_by_key(d_matrix.begin(), d_matrix.end(), iter1);

    ZipIterator2 iter2(thrust::make_tuple(d_matrix.begin(), d_column_indices.begin()));

    thrust::stable_sort_by_key(d_row_indices.begin(), d_row_indices.end(), iter2);

    typedef thrust::device_vector<int>::iterator Iterator;

    // --- Strided access to the sorted array
    //strided_range<Iterator> d_min_indices_1(d_column_indices.begin(), d_column_indices.end(), Ncols);
    //strided_range<Iterator> d_min_indices_2(d_column_indices.begin() + 1, d_column_indices.end() + 1, Ncols);
    strided_range<Iterator> d_min_indices_1(d_column_indices.begin() + Ncols - 1, d_column_indices.end() + Ncols - 1, Ncols);
    strided_range<Iterator> d_min_indices_2(d_column_indices.begin() + Ncols - 2, d_column_indices.end() + Ncols - 2, Ncols);

	printf("\n\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "]\n";
	}

	printf("\n\n");
    std::copy(d_min_indices_1.begin(), d_min_indices_1.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
	
	printf("\n\n");
    std::copy(d_min_indices_2.begin(), d_min_indices_2.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

	return 0;
}
