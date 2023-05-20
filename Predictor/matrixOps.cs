/*
* Copyright 2021-2023 - Tim Prishtina, and Luke Koch
*
* All rights reserved. No part of this software may be re-produced, re-engineered, 
* re-compiled, modified, used to create derivatives, stored in a retrieval system, 
* or transmitted in any form or by any means, whether electronic, mechanical, 
* photocopying, recording, or otherwise, without the prior written permission of 
* Tim Prishtina, and Luke Koch.
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using ManagedCuda;
using System.Collections;

namespace Predictor
{
    public class matrixOps
    {
        public double[] matrixMul(double[] inArray1, double[] inArray2, int M, int K, int N)
        {
            double[] result;
            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

            CudaDeviceVariable<double> d_in1 = inArray1;
            CudaDeviceVariable<double> d_in2 = inArray2;
            CudaDeviceVariable<double> res = new CudaDeviceVariable<double>(M * N);

            kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
            kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

            kernel.Run(d_in1.DevicePointer, d_in2.DevicePointer, res.DevicePointer, M, K, N);

            result = res;
            d_in1.Dispose();
            d_in2.Dispose();
            res.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();
            return result;
        }

        public double[] matrixMulCpu(double[] inArray1, double[] inArray2, int M, int K, int N)
        {
            double[] temp_array = new double[M * N];
            Parallel.For(0, M, (row, state) =>
            {
                Parallel.For(0, N, (col, state2) =>
                {
                    double tmp = 0;
                    Parallel.For(0, K, (i, state3) =>
                    {
                       tmp += inArray1[row * K + i] * inArray2[i * N + col];
                    });
                    temp_array[row * N + col] = tmp;
                });
            });
            return temp_array;
        }

        public double[] transposeMat(double[] inArray, int M, int K)
        {
            double[] result;
            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("transpose.ptx", "transpose");

            CudaDeviceVariable<double> d_transOut = inArray;
            CudaDeviceVariable<double> d_transOut_T = new CudaDeviceVariable<double>(M * K);

            kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
            kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

            kernel.Run(d_transOut.DevicePointer, d_transOut_T.DevicePointer, M, K);

            result = d_transOut_T;

            d_transOut.Dispose();
            d_transOut_T.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();
            return result;
        }

        public double[] conv5KernelBackProp(double[] prevConvLayerOut, double[] derivative, int startIdx, bool depth1)
        {
            double[] temp;
            double[] resVal = new double[14];
            double[] copiedDerivative = new double[1400];
            double[] allRowsPrevOut = new double[1400];

            int rowIdxAdd = 0;
            int rowIdxAdd2 = 0;
            for(int i = 0; i < 14; i++)
            {
                Array.Copy(derivative, startIdx, copiedDerivative, rowIdxAdd, 100);
                rowIdxAdd += 100;
            }

            //if(predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\copiedDerivativeConv4KernelBackProp.txt");
            //    for(int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(copiedDerivative[i].ToString());
            //    }
            //    output.Close();
            //}

            rowIdxAdd = 0;
            int stride = 0;
            if(depth1 == false)
            {
                stride = 16;
            }
            for(int i = 0; i < 14; i++)
            {
                Array.Copy(prevConvLayerOut, rowIdxAdd2 + stride, allRowsPrevOut, rowIdxAdd, 100);
                rowIdxAdd += 100;
                rowIdxAdd2 += 116;
            }

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\allRowsPrevOutConv5.txt");
            //    for (int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(allRowsPrevOut[i].ToString());
            //    }
            //    output.Close();
            //}

            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolutionBackProp");
            int N = 1400;

            CudaDeviceVariable<double> d_derivative = copiedDerivative;
            CudaDeviceVariable<double> d_prevLayerOut = allRowsPrevOut;
            CudaDeviceVariable<double> d_res = new CudaDeviceVariable<double>(N);

            kernel.BlockDimensions = 256;
            kernel.GridDimensions = (N + 255) / 256;

            kernel.Run(d_derivative.DevicePointer, d_prevLayerOut.DevicePointer, d_res.DevicePointer, N);
            temp = d_res;

            //if(predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\rawConv5DeltaOut.txt");
            //    for(int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(temp[i].ToString());
            //    }
            //    output.Close();
            //}

            int j = 0;
            for(int i = 0; i < 1400; i++)
            {
                if(i % 100 == 0 && i != 0)
                {
                    j++;
                }
                resVal[j] += temp[i];
            }

            d_derivative.Dispose();
            d_prevLayerOut.Dispose();
            d_res.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();

            return resVal;
        }

        public double[] conv4KernelBackProp(double[] prevConvLayerOut, double[] derivative, int startIdx, bool depth1)
        {
            double[] temp;
            double[] resVal = new double[14];
            double[] copiedDerivative = new double[1400];
            double[] allRowsPrevOut = new double[1400];

            int rowIdxAdd = 0;
            int rowIdxAdd2 = 0;
            for (int i = 0; i < 14; i++)
            {
                Array.Copy(derivative, startIdx, copiedDerivative, rowIdxAdd, 100);
                rowIdxAdd += 100;
            }

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\copiedDerivativeConv4KernelBackProp.txt");
            //    for (int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(copiedDerivative[i].ToString());
            //    }
            //    output.Close();
            //}

            rowIdxAdd = 0;
            int stride = 0;
            if (depth1 == false)
            {
                stride = 8;
            }
            for (int i = 0; i < 14; i++)
            {
                Array.Copy(prevConvLayerOut, rowIdxAdd2 + stride, allRowsPrevOut, rowIdxAdd, 100);
                rowIdxAdd += 100;
                rowIdxAdd2 += 108;
            }

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\allRowsPrevOutConv4.txt");
            //    for (int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(allRowsPrevOut[i].ToString());
            //    }
            //    output.Close();
            //}

            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolutionBackProp");
            int N = 1400;

            CudaDeviceVariable<double> d_derivative = copiedDerivative;
            CudaDeviceVariable<double> d_prevLayerOut = allRowsPrevOut;
            CudaDeviceVariable<double> d_res = new CudaDeviceVariable<double>(N);

            kernel.BlockDimensions = 256;
            kernel.GridDimensions = (N + 255) / 256;

            kernel.Run(d_derivative.DevicePointer, d_prevLayerOut.DevicePointer, d_res.DevicePointer, N);
            temp = d_res;

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\rawConv4DeltaOut.txt");
            //    for (int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(temp[i].ToString());
            //    }
            //    output.Close();
            //}

            int j = 0;
            for (int i = 0; i < 1400; i++)
            {
                if (i % 100 == 0 && i != 0)
                {
                    j++;
                }
                resVal[j] += temp[i];
            }

            d_derivative.Dispose();
            d_prevLayerOut.Dispose();
            d_res.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();

            return resVal;
        }

        public double[] find_conv_layer_err4(double[] transDerivative)
        {
            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolution2");

            int idx = 0;

            double[] resVal;
            double[] convVal_GPU = new double[1400];

            double[] padded_input = new double[1624];
            Array.Copy(transDerivative, 0, padded_input, 224, 1400);

            int N = 19600; //100 events * 14 depth = 1400 (with sixteen 0 events as causal padding) * 14 kernels = 19600

            kernel.BlockDimensions = 256;
            kernel.GridDimensions = (N + 255) / 256;

            double[] h_F = new double[19600];
            double[] k_F = new double[19600];

            double[] h_F2 = new double[19600];
            double[] k_F2 = new double[19600];

            int feature_idx = 0;
            for (int i = 0; i < 14; i++)
            {
                Array.Copy(padded_input, 0, h_F, feature_idx, 1400);
                Array.Copy(padded_input, 224, h_F2, feature_idx, 1400);
                feature_idx += 1400;
            }

            for (int kernel_idx = 0; kernel_idx < 14; kernel_idx++)
            {
                for (int i = 0; i < 100; i++)
                {
                    Array.Copy(predictorGui.convStructs[0].convLayer5Kernel5[kernel_idx].depth1, 0, k_F, idx, 14);
                    Array.Copy(predictorGui.convStructs[0].convLayer5Kernel5[kernel_idx].depth2, 0, k_F2, idx, 14);
                    idx += 14;
                }
            }

            CudaDeviceVariable<double> d_F = h_F;
            CudaDeviceVariable<double> dk_F = k_F;

            CudaDeviceVariable<double> d_F2 = h_F2;
            CudaDeviceVariable<double> dk_F2 = k_F2;

            CudaDeviceVariable<double> returnedVal = new CudaDeviceVariable<double>(N);
            kernel.Run(d_F.DevicePointer, dk_F.DevicePointer, d_F2.DevicePointer, dk_F2.DevicePointer, returnedVal.DevicePointer, N);

            resVal = returnedVal;

            int j = 0;
            for (int i = 0; i < 19600; i++)
            {
                if (i % 14 == 0 && i != 0)
                {
                    j++;
                }
                convVal_GPU[j] += resVal[i];
            }

            d_F.Dispose();
            dk_F.Dispose();
            d_F2.Dispose();
            dk_F2.Dispose();
            returnedVal.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();

            return convVal_GPU;
        }
        public double[] conv3KernelBackProp(double[] prevConvLayerOut, double[] derivative, int startIdx, bool depth1)
        {
            double[] temp;
            double[] resVal = new double[14];
            double[] copiedDerivative = new double[1400];
            double[] allRowsPrevOut = new double[1400];

            int rowIdxAdd = 0;
            int rowIdxAdd2 = 0;
            for (int i = 0; i < 14; i++)
            {
                Array.Copy(derivative, startIdx, copiedDerivative, rowIdxAdd, 100);
                rowIdxAdd += 100;
            }

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\copiedDerivativeConv3KernelBackProp.txt");
            //    for (int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(copiedDerivative[i].ToString());
            //    }
            //    output.Close();
            //}

            rowIdxAdd = 0;
            int stride = 0;
            if (depth1 == false)
            {
                stride = 4;
            }
            for (int i = 0; i < 14; i++)
            {
                Array.Copy(prevConvLayerOut, rowIdxAdd2 + stride, allRowsPrevOut, rowIdxAdd, 100);
                rowIdxAdd += 100;
                rowIdxAdd2 += 104;
            }

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\allRowsPrevOutConv3.txt");
            //    for (int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(allRowsPrevOut[i].ToString());
            //    }
            //    output.Close();
            //}

            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolutionBackProp");
            int N = 1400;

            CudaDeviceVariable<double> d_derivative = copiedDerivative;
            CudaDeviceVariable<double> d_prevLayerOut = allRowsPrevOut;
            CudaDeviceVariable<double> d_res = new CudaDeviceVariable<double>(N);

            kernel.BlockDimensions = 256;
            kernel.GridDimensions = (N + 255) / 256;

            kernel.Run(d_derivative.DevicePointer, d_prevLayerOut.DevicePointer, d_res.DevicePointer, N);
            temp = d_res;

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\rawConv3DeltaOut.txt");
            //    for (int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(temp[i].ToString());
            //    }
            //    output.Close();
            //}

            int j = 0;
            for (int i = 0; i < 1400; i++)
            {
                if (i % 100 == 0 && i != 0)
                {
                    j++;
                }
                resVal[j] += temp[i];
            }

            d_derivative.Dispose();
            d_prevLayerOut.Dispose();
            d_res.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();

            return resVal;
        }

        public double[] find_conv_layer_err3(double[] transDerivative)
        {
            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolution2");

            int idx = 0;

            double[] resVal;
            double[] convVal_GPU = new double[1400];

            double[] padded_input = new double[1512];
            Array.Copy(transDerivative, 0, padded_input, 112, 1400);

            int N = 19600; //100 events * 14 depth = 1400 (with sixteen 0 events as causal padding) * 14 kernels = 19600

            kernel.BlockDimensions = 256;
            kernel.GridDimensions = (N + 255) / 256;

            double[] h_F = new double[19600];
            double[] k_F = new double[19600];

            double[] h_F2 = new double[19600];
            double[] k_F2 = new double[19600];

            int feature_idx = 0;
            for (int i = 0; i < 14; i++)
            {
                Array.Copy(padded_input, 0, h_F, feature_idx, 1400);
                Array.Copy(padded_input, 112, h_F2, feature_idx, 1400);
                feature_idx += 1400;
            }

            for (int kernel_idx = 0; kernel_idx < 14; kernel_idx++)
            {
                for (int i = 0; i < 100; i++)
                {
                    Array.Copy(predictorGui.convStructs[0].convLayer4Kernel4[kernel_idx].depth1, 0, k_F, idx, 14);
                    Array.Copy(predictorGui.convStructs[0].convLayer4Kernel4[kernel_idx].depth2, 0, k_F2, idx, 14);
                    idx += 14;
                }
            }

            CudaDeviceVariable<double> d_F = h_F;
            CudaDeviceVariable<double> dk_F = k_F;

            CudaDeviceVariable<double> d_F2 = h_F2;
            CudaDeviceVariable<double> dk_F2 = k_F2;

            CudaDeviceVariable<double> returnedVal = new CudaDeviceVariable<double>(N);
            kernel.Run(d_F.DevicePointer, dk_F.DevicePointer, d_F2.DevicePointer, dk_F2.DevicePointer, returnedVal.DevicePointer, N);

            resVal = returnedVal;

            int j = 0;
            for (int i = 0; i < 19600; i++)
            {
                if (i % 14 == 0 && i != 0)
                {
                    j++;
                }
                convVal_GPU[j] += resVal[i];
            }

            d_F.Dispose();
            dk_F.Dispose();
            d_F2.Dispose();
            dk_F2.Dispose();
            returnedVal.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();

            return convVal_GPU;
        }

        public double[] conv2KernelBackProp(double[] prevConvLayerOut, double[] derivative, int startIdx, bool depth1)
        {
            double[] temp;
            double[] resVal = new double[14];
            double[] copiedDerivative = new double[1400];
            double[] allRowsPrevOut = new double[1400];

            int rowIdxAdd = 0;
            int rowIdxAdd2 = 0;
            for (int i = 0; i < 14; i++)
            {
                Array.Copy(derivative, startIdx, copiedDerivative, rowIdxAdd, 100);
                rowIdxAdd += 100;
            }

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\copiedDerivativeConv2KernelBackProp.txt");
            //    for (int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(copiedDerivative[i].ToString());
            //    }
            //    output.Close();
            //}

            rowIdxAdd = 0;
            int stride = 0;
            if (depth1 == false)
            {
                stride = 2;
            }
            for (int i = 0; i < 14; i++)
            {
                Array.Copy(prevConvLayerOut, rowIdxAdd2 + stride, allRowsPrevOut, rowIdxAdd, 100);
                rowIdxAdd += 100;
                rowIdxAdd2 += 102;
            }

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\allRowsPrevOutConv2.txt");
            //    for (int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(allRowsPrevOut[i].ToString());
            //    }
            //    output.Close();
            //}

            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolutionBackProp");
            int N = 1400;

            CudaDeviceVariable<double> d_derivative = copiedDerivative;
            CudaDeviceVariable<double> d_prevLayerOut = allRowsPrevOut;
            CudaDeviceVariable<double> d_res = new CudaDeviceVariable<double>(N);

            kernel.BlockDimensions = 256;
            kernel.GridDimensions = (N + 255) / 256;

            kernel.Run(d_derivative.DevicePointer, d_prevLayerOut.DevicePointer, d_res.DevicePointer, N);
            temp = d_res;

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\rawConv2DeltaOut.txt");
            //    for (int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(temp[i].ToString());
            //    }
            //    output.Close();
            //}

            int j = 0;
            for (int i = 0; i < 1400; i++)
            {
                if (i % 100 == 0 && i != 0)
                {
                    j++;
                }
                resVal[j] += temp[i];
            }

            d_derivative.Dispose();
            d_prevLayerOut.Dispose();
            d_res.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();

            return resVal;
        }
        public double[] find_conv_layer_err2(double[] transDerivative)
        {
            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolution2");

            int idx = 0;

            double[] resVal;
            double[] convVal_GPU = new double[1400];

            double[] padded_input = new double[1456];
            Array.Copy(transDerivative, 0, padded_input, 56, 1400);

            int N = 19600; //100 events * 14 depth = 1400 (with sixteen 0 events as causal padding) * 14 kernels = 19600

            kernel.BlockDimensions = 256;
            kernel.GridDimensions = (N + 255) / 256;

            double[] h_F = new double[19600];
            double[] k_F = new double[19600];

            double[] h_F2 = new double[19600];
            double[] k_F2 = new double[19600];

            int feature_idx = 0;
            for (int i = 0; i < 14; i++)
            {
                Array.Copy(padded_input, 0, h_F, feature_idx, 1400);
                Array.Copy(padded_input, 56, h_F2, feature_idx, 1400);
                feature_idx += 1400;
            }

            for (int kernel_idx = 0; kernel_idx < 14; kernel_idx++)
            {
                for (int i = 0; i < 100; i++)
                {
                    Array.Copy(predictorGui.convStructs[0].convLayer3Kernel3[kernel_idx].depth1, 0, k_F, idx, 14);
                    Array.Copy(predictorGui.convStructs[0].convLayer3Kernel3[kernel_idx].depth2, 0, k_F2, idx, 14);
                    idx += 14;
                }
            }

            CudaDeviceVariable<double> d_F = h_F;
            CudaDeviceVariable<double> dk_F = k_F;

            CudaDeviceVariable<double> d_F2 = h_F2;
            CudaDeviceVariable<double> dk_F2 = k_F2;

            CudaDeviceVariable<double> returnedVal = new CudaDeviceVariable<double>(N);
            kernel.Run(d_F.DevicePointer, dk_F.DevicePointer, d_F2.DevicePointer, dk_F2.DevicePointer, returnedVal.DevicePointer, N);

            resVal = returnedVal;

            int j = 0;
            for (int i = 0; i < 19600; i++)
            {
                if (i % 14 == 0 && i != 0)
                {
                    j++;
                }
                convVal_GPU[j] += resVal[i];
            }

            d_F.Dispose();
            dk_F.Dispose();
            d_F2.Dispose();
            dk_F2.Dispose();
            returnedVal.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();

            return convVal_GPU;
        }
        public double[] find_conv_layer_err1(double[] transDerivative)
        {
            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolution2");

            int idx = 0;

            double[] resVal;
            double[] convVal_GPU = new double[1400];

            double[] padded_input = new double[1428];
            Array.Copy(transDerivative, 0, padded_input, 28, 1400);

            int N = 19600; //100 events * 14 depth = 1400 (with sixteen 0 events as causal padding) * 14 kernels = 19600

            kernel.BlockDimensions = 256;
            kernel.GridDimensions = (N + 255) / 256;

            double[] h_F = new double[19600];
            double[] k_F = new double[19600];

            double[] h_F2 = new double[19600];
            double[] k_F2 = new double[19600];

            int feature_idx = 0;
            for (int i = 0; i < 14; i++)
            {
                Array.Copy(padded_input, 0, h_F, feature_idx, 1400);
                Array.Copy(padded_input, 28, h_F2, feature_idx, 1400);
                feature_idx += 1400;
            }

            for (int kernel_idx = 0; kernel_idx < 14; kernel_idx++)
            {
                for (int i = 0; i < 100; i++)
                {
                    Array.Copy(predictorGui.convStructs[0].convLayer2Kernel2[kernel_idx].depth1, 0, k_F, idx, 14);
                    Array.Copy(predictorGui.convStructs[0].convLayer2Kernel2[kernel_idx].depth2, 0, k_F2, idx, 14);
                    idx += 14;
                }
            }

            CudaDeviceVariable<double> d_F = h_F;
            CudaDeviceVariable<double> dk_F = k_F;

            CudaDeviceVariable<double> d_F2 = h_F2;
            CudaDeviceVariable<double> dk_F2 = k_F2;

            CudaDeviceVariable<double> returnedVal = new CudaDeviceVariable<double>(N);
            kernel.Run(d_F.DevicePointer, dk_F.DevicePointer, d_F2.DevicePointer, dk_F2.DevicePointer, returnedVal.DevicePointer, N);

            resVal = returnedVal;

            int j = 0;
            for (int i = 0; i < 19600; i++)
            {
                if (i % 14 == 0 && i != 0)
                {
                    j++;
                }
                convVal_GPU[j] += resVal[i];
            }

            d_F.Dispose();
            dk_F.Dispose();
            d_F2.Dispose();
            dk_F2.Dispose();
            returnedVal.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();

            return convVal_GPU;
        }
        public double[] conv1KernelBackProp(double[] prevConvLayerOut, double[] derivative, int startIdx, int depth)
        {
            double[] temp;
            double[] resVal = new double[32];
            double[] copiedDerivative = new double[3200];
            double[] allRowsPrevOut = new double[3200];

            int rowIdxAdd = 0;
            int rowIdxAdd2 = 0;
            for (int i = 0; i < 32; i++)
            {
                Array.Copy(derivative, startIdx, copiedDerivative, rowIdxAdd, 100);
                rowIdxAdd += 100;
            }

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\copiedDerivativeConv1KernelBackProp.txt");
            //    for (int i = 0; i < 1400; i++)
            //    {
            //        output.WriteLine(copiedDerivative[i].ToString());
            //    }
            //    output.Close();
            //}

            rowIdxAdd = 0;
            int stride = 0;
            if (depth == 3 || depth == 4)
            {
                stride = 1;
            }
            for (int i = 0; i < 32; i++)
            {
                Array.Copy(prevConvLayerOut, rowIdxAdd2 + stride, allRowsPrevOut, rowIdxAdd, 100);
                rowIdxAdd += 100;
                rowIdxAdd2 += 101;
            }

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\allRowsPrevOutConv1.txt");
            //    for (int i = 0; i < 3200; i++)
            //    {
            //        output.WriteLine(allRowsPrevOut[i].ToString());
            //    }
            //    output.Close();
            //}

            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolutionBackProp");
            int N = 3200;

            CudaDeviceVariable<double> d_derivative = copiedDerivative;
            CudaDeviceVariable<double> d_prevLayerOut = allRowsPrevOut;
            CudaDeviceVariable<double> d_res = new CudaDeviceVariable<double>(N);

            kernel.BlockDimensions = 256;
            kernel.GridDimensions = (N + 255) / 256;

            kernel.Run(d_derivative.DevicePointer, d_prevLayerOut.DevicePointer, d_res.DevicePointer, N);
            temp = d_res;

            //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\rawConv1DeltaOut.txt");
            //    for (int i = 0; i < 3200; i++)
            //    {
            //        output.WriteLine(temp[i].ToString());
            //    }
            //    output.Close();
            //}

            int j = 0;
            for (int i = 0; i < 3200; i++)
            {
                if (i % 100 == 0 && i != 0)
                {
                    j++;
                }
                resVal[j] += temp[i];
            }

            d_derivative.Dispose();
            d_prevLayerOut.Dispose();
            d_res.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();

            return resVal;
        }
    }
}
