#include <stdio.h>

#include "Utilities.cuh"
#include "transform_reduce.cuh"

#define BLOCKSIZE 512
#define warpSize 32

// --- Host-side function pointer
pointFunction_t h_pfunc;

// --- Uncomment if you want to apply CUDA error checking to the kernel launches
//#define DEBUG

//#define EXTERNAL

/*******************************************************/
/* CALCULATING THE NEXT POWER OF 2 OF A CERTAIN NUMBER */

/*******************************************************/
unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/*************************************/
/* CHECK IF A NUMBER IS A POWER OF 2 */
/*************************************/
// --- Note: although x = 1 is a power of 2 (1 = 2^0), this routine returns 0 for x == 1.
bool isPow2(unsigned int x) {
    if (x == 1) return 0;
    else        return ((x&(x-1))==0);
}

/***************************/
/* TRANSFORMATION FUNCTION */
/***************************/
template <class T>
__host__ __device__ __forceinline__ T transformation(const T * __restrict__ x, const int i) { return ((T)100 * (x[i+1] - x[i] * x[i]) * (x[i+1] - x[i] * x[i]) + (x[i] - (T)1) * (x[i] - (T)1)) ; }

/********************/
/* REDUCTION KERNEL */
/********************/
/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reductionKernel(T *g_idata, T *g_odata, unsigned int N, pointFunction_t pPointTransformation)
{
    extern __shared__ T sdata[];

    unsigned int tid    = threadIdx.x;                              // Local thread index
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;       // Global thread index - Fictitiously double the block dimension
    unsigned int gridSize = blockSize*2*gridDim.x;

    // --- Performs the first level of reduction in registers when reading from global memory on multiple elements per thread.
    //     More blocks will result in a larger gridSize and therefore fewer elements per thread
    T mySum = 0;

    while (i < N) {
#ifdef EXTERNAL
        mySum += (*pPointTransformation)(g_idata, i);
#else
        mySum += transformation(g_idata, i);
#endif
        // --- Ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < N)
#ifdef EXTERNAL
            mySum += (*pPointTransformation)(g_idata, i+blockSize);
#else
            mySum += transformation(g_idata, i+blockSize);
#endif
        i += gridSize; }

    // --- Each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();

    // --- Reduction in shared memory. Fully unrolled loop.
    if ((blockSize >= 512) && (tid < 256)) sdata[tid] = mySum = mySum + sdata[tid + 256];
    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) sdata[tid] = mySum = mySum + sdata[tid + 128];
     __syncthreads();

    if ((blockSize >= 128) && (tid <  64)) sdata[tid] = mySum = mySum + sdata[tid +  64];
    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    // --- Single warp reduction by shuffle operations
    if ( tid < 32 )
    {
        // --- Last iteration removed from the for loop, but needed for shuffle reduction
        mySum += sdata[tid + 32];
        // --- Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) mySum += __shfl_down(mySum, offset);
        //for (int offset=1; offset < warpSize; offset *= 2) mySum += __shfl_xor(mySum, i);
    }
#else
    // --- Reduction within a single warp. Fully unrolled loop.
    if ((blockSize >=  64) && (tid < 32)) sdata[tid] = mySum = mySum + sdata[tid + 32];
    __syncthreads();

    if ((blockSize >=  32) && (tid < 16)) sdata[tid] = mySum = mySum + sdata[tid + 16];
    __syncthreads();

    if ((blockSize >=  16) && (tid <  8)) sdata[tid] = mySum = mySum + sdata[tid +  8];
    __syncthreads();

    if ((blockSize >=   8) && (tid <  4)) sdata[tid] = mySum = mySum + sdata[tid +  4];
     __syncthreads();

    if ((blockSize >=   4) && (tid <  2)) sdata[tid] = mySum = mySum + sdata[tid +  2];
    __syncthreads();

    if ((blockSize >=   2) && ( tid < 1)) sdata[tid] = mySum = mySum + sdata[tid +  1];
    __syncthreads();
#endif

    // --- Write result for this block to global memory. At the end of the kernel, global memory will contain the results for the summations of
    //     individual blocks
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

/******************************/
/* REDUCTION WRAPPER FUNCTION */
/******************************/
template <class T>
T transform_reduce_inner(T *g_idata, unsigned int N, pointFunction_t h_pfunc) {

    // --- Reduction parameters
    const int NumThreads    = (N < BLOCKSIZE) ? nextPow2(N) : BLOCKSIZE;
    const int NumBlocks     = (N + NumThreads - 1) / NumThreads;
    const int smemSize      = (NumThreads <= 32) ? 2 * NumThreads * sizeof(T) : NumThreads * sizeof(T);

    // --- Device memory space where storing the partial reduction results
    T *g_odata; gpuErrchk(cudaMalloc((void**)&g_odata, NumBlocks * sizeof(T)));

    if (isPow2(N)) {
        switch (NumThreads) {
            case 512: reductionKernel<T, 512, true><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case 256: reductionKernel<T, 256, true><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case 128: reductionKernel<T, 128, true><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case 64:  reductionKernel<T,  64, true><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case 32:  reductionKernel<T,  32, true><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case 16:  reductionKernel<T,  16, true><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case  8:  reductionKernel<T,   8, true><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case  4:  reductionKernel<T,   4, true><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case  2:  reductionKernel<T,   2, true><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case  1:  reductionKernel<T,   1, true><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
        }
#ifdef DEBUG
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
#endif
    }
    else {
        switch (NumThreads) {
            case 512: reductionKernel<T, 512, false><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case 256: reductionKernel<T, 256, false><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case 128: reductionKernel<T, 128, false><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case 64:  reductionKernel<T,  64, false><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case 32:  reductionKernel<T,  32, false><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case 16:  reductionKernel<T,  16, false><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case  8:  reductionKernel<T,   8, false><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case  4:  reductionKernel<T,   4, false><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case  2:  reductionKernel<T,   2, false><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
            case  1:  reductionKernel<T,   1, false><<< NumBlocks, NumThreads, smemSize>>>(g_idata, g_odata, N, h_pfunc); break;
        }
#ifdef DEBUG
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
#endif
    }

    // --- The last part of the reduction, which would be expensive to perform on the device, is executed on the host
    T *host_vector = (T *)malloc(NumBlocks * sizeof(T));
    gpuErrchk(cudaMemcpy(host_vector, g_odata, NumBlocks * sizeof(T), cudaMemcpyDeviceToHost));

    T sum_transformReduce = (T)0;
    for (int i=0; i<NumBlocks; i++) sum_transformReduce = sum_transformReduce + host_vector[i];

    return sum_transformReduce;
}

template <class T>
T transform_reduce(T *g_idata, unsigned int N, pointFunction_t *dev_pfunc) {
#ifdef EXTERNAL
    gpuErrchk(cudaMemcpyFromSymbol(&h_pfunc, *dev_pfunc, sizeof(pointFunction_t)));
#endif
    T customizedDeviceResult = transform_reduce_inner(g_idata, N, h_pfunc);
    return customizedDeviceResult;
}


// --- Complete with your own favourite instantiations
template float transform_reduce(float *, unsigned int, pointFunction_t *);
