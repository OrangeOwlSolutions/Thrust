#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

using namespace thrust::placeholders;

int main(void)
{
	// --- Input data 
	float a = 2.0f;
	float x[4] = { 1, 2, 3, 4 };
	float y[4] = { 1, 1, 1, 1 };

	thrust::host_vector<float> X(x, x + 4);
	thrust::host_vector<float> Y(y, y + 4);

	thrust::transform(X.begin(), X.end(),
					  Y.begin(),         
					  Y.begin(),
					  a * _1 + _2);

	for (size_t i = 0; i < 4; i++) std::cout << a << " * " << x[i] << " + " << y[i] << " = " << Y[i] << std::endl;

	return 0;
}

