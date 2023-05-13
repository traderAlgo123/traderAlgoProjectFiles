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
    __global__ void matrixMul(double* a, double* b, double* c, int m, int n, int k)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        double sum = 0;
        if (col < k && row < m)
        {
            for (int i = 0; i < n; i++)
            {
                sum += a[row * n + i] * b[i * k + col];
            }
            c[row * k + col] = sum;
        }
    }
}
