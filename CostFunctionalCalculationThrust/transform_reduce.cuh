#ifndef TRANSFORM_REDUCE_CUH
#define TRANSFORM_REDUCE_CUH

// --- Function pointer type
// --- Complete with your own favourite instantiations
typedef float(*pointFunction_t)(const float * __restrict__, const int);

template <class T> T transform_reduce(T *, unsigned int, pointFunction_t *);

#endif
