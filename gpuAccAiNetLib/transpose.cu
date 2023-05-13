#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

extern "C" {
	__global__ void transpose(double* idata, double* odata, int width, int height)
	{
        unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

        if (xIndex < width && yIndex < height)
        {
            unsigned int index_in = xIndex + width * yIndex;
            unsigned int index_out = yIndex + height * xIndex;
            odata[index_out] = idata[index_in];
        }
	}
}