#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#ifndef __INTELLISENSE__
#define KERNEL_ARGS2(grid, block)                 <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem)         <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

extern "C" {
	__global__ void matrixMulEx(int* a, int* b, int* c, int N)
	{
		//Calculate the global row and column for each thread
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;

		//Bounds check for matrix
		if (row < N && col < N)
		{
			//Accumulate a partial result
			int temp = 0;
			for (int i = 0; i < N; i++)
			{
				temp += a[row * N + i] * b[i * N + col];
			}

			//Write back result
			c[row * N + col] = temp;
		}
	}
}