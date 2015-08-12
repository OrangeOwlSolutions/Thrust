// --- Requires separate compilation

#include <stdio.h>

#include <thrust/device_vector.h>

#include "transform_reduce.cuh"
#include "Utilities.cuh"
#include "TimingCPU.h"
#include "TimingGPU.cuh"

/***************************************/
/* COST FUNCTION - GPU/CUSTOMIZED CASE */
/***************************************/
// --- Transformation function
__device__ __forceinline__ float transformation(const float * __restrict__ x, const int i) { return (100.f * (x[i+1] - x[i] * x[i]) * (x[i+1] - x[i] * x[i]) + (x[i] - 1.f) * (x[i] - 1.f)) ; }
// --- Device-side function pointer
__device__ pointFunction_t dev_pfunc = transformation;

/***********************************/
/* COST FUNCTION - GPU/THRUST CASE */
/***********************************/
struct CostFunctionStructGPU{
template <typename Tuple>
    __host__ __device__ float operator()(Tuple a) {

        float temp1 = (thrust::get<1>(a) - thrust::get<0>(a) * thrust::get<0>(a));
        float temp2 = (thrust::get<0>(a) - 1.f);

        return 100.f * temp1 * temp1 + temp2 * temp2;
    }
};

/********/
/* MAIN */
/********/
int main()
{
    const int N = 90000000;

    float *x = (float *)malloc(N * sizeof(float));
    for (int i=0; i<N; i++) x[i] = 3.f;

    float *d_x; gpuErrchk(cudaMalloc((void**)&d_x, N * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));

    /************************************************/
    /* "CUSTOMIZED" DEVICE-SIDE TRANSFORM REDUCTION */
    /************************************************/
    float customizedDeviceResult = transform_reduce(d_x, N - 1, &dev_pfunc);
    TimingGPU timerGPU;
    timerGPU.StartCounter();
    customizedDeviceResult = transform_reduce(d_x, N - 1, &dev_pfunc);
    printf("Timing for 'customized', device-side transform reduction = %f\n", timerGPU.GetCounter());
    printf("Result for 'customized', device-side transform reduction = %f\n", customizedDeviceResult);
    printf("\n\n");

    /************************************************/
    /* THRUST-BASED DEVICE-SIDE TRANSFORM REDUCTION */
    /************************************************/
    thrust::device_vector<float> d_vec(N,3.f);

    timerGPU.StartCounter();
    float ThrustResult = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(d_vec.begin(), d_vec.begin() + 1)), thrust::make_zip_iterator(thrust::make_tuple(d_vec.begin() + N - 1, d_vec.begin() + N)), CostFunctionStructGPU(), 0.f, thrust::plus<float>());
    printf("Timing for Thrust-based, device-side transform reduction = %f\n", timerGPU.GetCounter());
    printf("Result for Thrust-based, device-side transform reduction = %f\n", ThrustResult);
    printf("\n\n");

    /*********************************/
    /* HOST-SIDE TRANSFORM REDUCTION */
    /*********************************/
//  thrust::host_vector<float> h_vec(d_vec);
        //sum_host = sum_host + transformation(thrust::raw_pointer_cast(h_vec.data()), i);

    TimingCPU timerCPU;
    timerCPU.StartCounter();
    float sum_host = 0.f;
    for (int i=0; i<N-1; i++) {
        float temp = (100.f * (x[i+1] - x[i] * x[i]) * (x[i+1] - x[i] * x[i]) + (x[i] - 1.f) * (x[i] - 1.f));
        sum_host = sum_host + temp;
        //printf("%i %f %f\n", i, temp, sum_host);
    }

    printf("Timing for host-side transform reduction = %f\n", timerCPU.GetCounter());
    printf("Result for host-side transform reduction = %f\n", sum_host);
    printf("\n\n");

    sum_host = 0.f;
    float c        = 0.f;
    for (int i=0; i<N-1; i++) {
        float temp = (100.f * (x[i+1] - x[i] * x[i]) * (x[i+1] - x[i] * x[i]) + (x[i] - 1.f) * (x[i] - 1.f)) - c;
        float t    = sum_host + temp;
        c          = (t - sum_host) - temp;
        sum_host = t;
    }

    printf("Result for host-side transform reduction = %f\n", sum_host);

//  cudaDeviceReset();

}
