#include "cuda_runtime.h"
#include <device_launch_parameters.h>

extern "C" {
	__global__ void convolution(double* pricesBlock, double* sizesBlock,
		 double* pricesKernel, double* sizesKernel, double* pricesBlock2,
		 double* sizesBlock2, double* pricesKernel2, double* sizesKernel2,
		 double* resVal, int N)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < N)
		{
			resVal[i] = (pricesBlock[i] * pricesKernel[i]) + (sizesBlock[i] * sizesKernel[i]) + (pricesBlock2[i] *
				pricesKernel2[i]) + (sizesBlock2[i] * sizesKernel2[i]);
		}
	}
}

extern "C" {
	__global__ void convolution2(double* featuresBlock, double* featuresKernel, double* featuresBlock2,
        double* featuresKernel2, double* resVal, int N)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < N)
		{
			resVal[i] = (featuresBlock[i] * featuresKernel[i]) + (featuresBlock2[i] * featuresKernel2[i]);
		}
	}
}

extern "C" {
	__global__ void convolutionBackProp(double* derBlock, double* inputBlock, double* resVal, int N)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < N)
		{
			resVal[i] = (derBlock[i] * inputBlock[i]);
		}
	}
}