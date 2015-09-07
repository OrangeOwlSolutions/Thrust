#include <thrust/device_vector.h>
#include <cstdio>

int main()
{

   const int N = 20;

   float a = 3.87f;
   float b = 7.11f;

   float Dx = (b-a)/(float)(N-1);

   thrust::device_vector<float> myvector(N);

   thrust::transform(thrust::make_counting_iterator(a/Dx), thrust::make_counting_iterator((b+1.f)/Dx), thrust::make_constant_iterator(Dx), myvector.begin(), thrust::multiplies<float>());

   for(int i=0; i<N; i++) 
   {
      float val = myvector[i];
      printf("%d %f\n", i, val);
   }

   return 0;
}
