#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

using namespace thrust::placeholders;

// --- Defining key tuple type
typedef thrust::tuple<int,int> Tuple;

typedef thrust::device_vector<Tuple>::iterator  dIter1;
typedef thrust::device_vector<float>::iterator  dIter2;

/*******************/
/* EXPAND OPERATOR */
/*******************/
template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator expand(InputIterator1 first1,
                      InputIterator1 last1,
                      InputIterator2 first2,
                      OutputIterator output)
{
	typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;
  
	difference_type input_size  = thrust::distance(first1, last1);
	difference_type output_size = thrust::reduce(first1, last1);

	// scan the counts to obtain output offsets for each input element
	thrust::device_vector<difference_type> output_offsets(input_size, 0);
	thrust::exclusive_scan(first1, last1, output_offsets.begin()); 

	// scatter the nonzero counts into their corresponding output positions
	thrust::device_vector<difference_type> output_indices(output_size, 0);
	thrust::scatter_if(thrust::counting_iterator<difference_type>(0), thrust::counting_iterator<difference_type>(input_size),
					   output_offsets.begin(), first1, output_indices.begin());

	// compute max-scan over the output indices, filling in the holes
	thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin(), thrust::maximum<difference_type>());

	// gather input values according to index array (output = first2[output_indices])
	OutputIterator output_end = output; thrust::advance(output_end, output_size);
	thrust::gather(output_indices.begin(), output_indices.end(), first2, output);

	// return output + output_size
	thrust::advance(output, output_size);
  
	return output;
}

/************************************/
/* EQUALITY OPERATOR BETWEEN TUPLES */
/************************************/
struct BinaryPredicate
{
  __host__ __device__ bool operator () 
                      (const Tuple& lhs, const Tuple& rhs) 
  {
    return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) && (thrust::get<1>(lhs) == thrust::get<1>(rhs));
  }
};

/*************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX */
/*************************************/
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
	
	T Ncols; // --- Number of columns
  
	__host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}

	__host__ __device__ T operator()(T i) { return i / Ncols; }
};

/********************************************/
/* TRANSFORM COUPLE OF KEY INDICES TO TUPLE */
/********************************************/
struct ArraysToTuple {

    __host__ __device__ Tuple operator()(const int& o1, const int& o2) {
            
		Tuple o3 = thrust::make_tuple(o1, o2);
        return o3;
    }
};

/****************************************/
/* CONVERT LINEAR INDEX TO COLUMN INDEX */
/****************************************/
template< typename T >
struct mod_functor {
    __host__ __device__ T operator()(T a, T b) { return a % b; }
};

/********/
/* MAIN */
/********/
int main()
{
	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
  
	const int Nrows = 5;     // --- Number of rows
	const int Ncols = 8;     // --- Number of columns
  
	// --- Random uniform integer distribution between 10 and 99
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist(10, 99);

	// --- Matrix allocation and initialization
	thrust::device_vector<float> d_matrix(Nrows * Ncols);
	for (size_t i = 0; i < d_matrix.size(); i++) d_matrix[i] = (float)dist(rng);

	// --- Setting up the row memberships
	thrust::uniform_int_distribution<int> dist_membership(0, Nrows - 1);
	thrust::device_vector<float> d_membership(Nrows);
	for (size_t i = 0; i < d_membership.size(); i++) d_membership[i] = (float)dist_membership(rng);

	printf("Row memberships\n");
	for (int i=0; i<Nrows; i++) std::cout << d_membership[i] << "\n";

	/**************************/
	/* EXPAND THE MEMBERSHIPS */
	/**************************/
	thrust::device_vector<int> d_membership_expanded(Nrows * Ncols);
	thrust::device_vector<int> d_counts(Nrows, Ncols);
	expand(d_counts.begin(), d_counts.end(), d_membership.begin(), d_membership_expanded.begin());

	// --- Print result
	printf("\n\nExpanded memberships\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_membership_expanded[i * Ncols + j] << " ";
		std::cout << "] \n";
	}

	/****************************************************/
	/* REORDERING THE ROWS ACCORDING TO THE MEMBERSHIPS */
	/****************************************************/
	printf("\n\nOriginal matrix\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "] \n";
	}

	thrust::stable_sort_by_key(d_membership_expanded.begin(), d_membership_expanded.end(), d_matrix.begin(), thrust::less<int>());

	printf("\n\nReordered matrix\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_matrix[i * Ncols + j] << " ";
		std::cout << "] \n";
	}

	// --- Print result
	printf("\n\nReordered expanded memberships\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_membership_expanded[i * Ncols + j] << " ";
		std::cout << "] \n";
	}

	/*************************************************************/
	/* CREATING TUPLE KEYS FOR REDUCTION OF COLUMNS BY TUPLE KEY */
	/*************************************************************/
	thrust::device_vector<Tuple> tuple_keys(Nrows * Ncols);
	
	// --- Computing column indices vector
    thrust::device_vector<int> d_column_indices(Nrows * Ncols);
    thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(Nrows * Ncols), thrust::make_constant_iterator(Ncols), d_column_indices.begin(), mod_functor<int>());

	thrust::transform(d_membership_expanded.begin(), d_membership_expanded.end(), d_column_indices.begin(), tuple_keys.begin(), ArraysToTuple());

	thrust::host_vector<Tuple> h_tuple_keys(tuple_keys);
	printf("\n\nReordered tuples\n");
	for(int i = 0; i < Nrows; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++) {
			int p = thrust::get<0>(h_tuple_keys[i * Ncols + j]);
			int q = thrust::get<1>(h_tuple_keys[i * Ncols + j]);
			std::cout << "(" << p << ", " << q << ") ";
		}
		std::cout << "] \n";
	}

	thrust::device_vector<float> d_centroids(Nrows * Ncols);
	
	const int Nkeys = 3;

	thrust::reduce_by_key(thrust::make_permutation_iterator(tuple_keys.begin(),  thrust::make_transform_iterator(thrust::make_counting_iterator(0),(_1 % Nrows) * Ncols + _1 / Nrows)), 
						  thrust::make_permutation_iterator(tuple_keys.begin(),  thrust::make_transform_iterator(thrust::make_counting_iterator(0),(_1 % Nrows) * Ncols + _1 / Nrows)) + Nrows * Ncols, 
						  thrust::make_permutation_iterator(d_matrix.begin(),    thrust::make_transform_iterator(thrust::make_counting_iterator(0),(_1 % Nrows) * Ncols + _1 / Nrows)), 
						  thrust::make_permutation_iterator(tuple_keys.begin(),  thrust::make_transform_iterator(thrust::make_counting_iterator(0),(_1 % Nrows) * Ncols + _1 / Nrows)),
						  thrust::make_permutation_iterator(d_centroids.begin(), thrust::make_transform_iterator(thrust::make_counting_iterator(0),(_1 % Nkeys) * Ncols + _1 / Nkeys)),
						  BinaryPredicate(), 
						  thrust::plus<float>());

	printf("\n\nReduced matrix\n");
	for(int i = 0; i < Nkeys; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols; j++)
			std::cout << d_centroids[i * Ncols + j] << " ";
		std::cout << "] \n";
	}

	return 0;
}
