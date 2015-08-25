#include <thrust/sequence.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(void)
{
    int N = 20;

    // --- Filter parameters
    double alpha    = 2.7;
    double beta     = -0.3;

    // --- Defining and initializing the input vector on the device
    thrust::device_vector<double> d_input(N,alpha * 1.);
    d_input[0] = d_input[0]/alpha;

    // --- Defining the output vector on the device
    thrust::device_vector<double> d_output(d_input);

    // --- Defining the {1/beta^n} sequence
    thrust::device_vector<double> d_1_over_beta(N,1./beta);
    thrust::device_vector<double> d_1_over_beta_to_the_n(N,1./beta);
    thrust::device_vector<double> d_n(N);
    thrust::sequence(d_n.begin(), d_n.end());
    thrust::inclusive_scan(d_1_over_beta.begin(), d_1_over_beta.end(), d_1_over_beta_to_the_n.begin(), thrust::multiplies<double>());
    thrust::transform(d_1_over_beta_to_the_n.begin(), d_1_over_beta_to_the_n.end(), d_input.begin(), d_input.begin(), thrust::multiplies<double>());    
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), thrust::plus<double>());
    thrust::transform(d_output.begin(), d_output.end(), d_1_over_beta_to_the_n.begin(), d_output.begin(), thrust::divides<double>());

    for (int i=0; i<N; i++) {
        double val = d_output[i];
        printf("Device vector element number %i equal to %f\n",i,val);
    }

    // --- Defining and initializing the input vector on the host
    thrust::host_vector<double> h_input(N,1.);

    // --- Defining the output vector on the host
    thrust::host_vector<double> h_output(h_input);

    h_output[0] = h_input[0];
    for(int i=1; i<N; i++)
    {
        h_output[i] = h_input[i] * alpha + beta * h_output[i-1];
    }

    for (int i=0; i<N; i++) {
        double val = h_output[i];
        printf("Host vector element number %i equal to %f\n",i,val);
    }

    for (int i=0; i<N; i++) {
        double val = h_output[i] - d_output[i];
        printf("Difference between host and device vector element number %i equal to %f\n",i,val);
    }

    getchar();
}
