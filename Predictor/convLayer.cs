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
using ManagedCuda.BasicTypes;
using System.Diagnostics;
using System.Runtime;

namespace Predictor
{
    public class convLayer
    {
        //conv layer takes in 100 events from the data scraper and performs a convolution
        //operation using a kernel of size 2 on each event. This then returns a pointer to a matrix of convolved values
        //of equal size to the previous matrix. padding will be introduced to the input matrix before
        //convolution. conceptually each element in the full matrix is treated as a neuron, therefore we will then pump
        //these values through a relu function to obtain the input nodes for the next convolution layer.

        public static double SeLU_lambda = 1.0507009873554804934193349852946;
        public static double SeLU_alpha = 1.6732632423543772848170429916717;

        //public static double SeLU_lambda = 1.1507009873554804934193349852946;
        //public static double SeLU_alpha = 1.8732632423543772848170429916717;

        matrixOps matOps = new matrixOps();
        public void convolution1D(int events, int networkNum, bool trainingActive)
        {
            if (trainingActive == true)
            {
                Parallel.For(0, 32, (k, state) =>
                {
                    //initialize events array with passed in events param
                    //we will leave first event empty as our causal padding
                    //for (int i = 0; i < events + 1; i++)
                    //{
                    //    predictorGui.totalEventsArray[k].eventsArray[i] = new Event();
                    //}

                    //each tensor is 32 lines long, iterate through tensorIn
                    //to retrieve
                    int j = 1;
                    int x = 0;
                    for (int i = 0; i < (32 * events); i++)
                    {
                        if (i % 32 == 0 && i != 0)
                        {
                            j++;
                            x = 0;
                        }
                        predictorGui.totalEventsArray[k].eventsArray[j].prices[x] = predictorGui.tensorIn[k].price[i];
                        predictorGui.totalEventsArray[k].eventsArray[j].sizes[x] = predictorGui.tensorIn[k].size[i];
                        if (j == 1)
                        {
                            //populate causal padding with copy of the first event, this is for replication padding
                            predictorGui.totalEventsArray[k].eventsArray[j - 1].prices[x] = predictorGui.tensorIn[k].price[i];
                            predictorGui.totalEventsArray[k].eventsArray[j - 1].sizes[x] = predictorGui.tensorIn[k].size[i];
                        }
                        x++;
                    }
                });

                Parallel.For(0, 32, (n, state) =>
                {
                    conv_full_layer1(networkNum, n);
                    conv_full_layer2(networkNum, n);
                    conv_full_layer3(networkNum, n);
                    conv_full_layer4(networkNum, n);
                    conv_full_layer5(networkNum, n);

                    if (predictorGui.predictorGui1.preluSelect.Checked == true || predictorGui.predictorGui1.mishSelect.Checked == true)
                    {
                        layerNorm(networkNum, n);
                    }
                    else if (predictorGui.predictorGui1.seluSelect.Checked == true)
                    {
                        Array.Copy(predictorGui.networkArray[networkNum].convStructs[n].convLayer5Output, 0, predictorGui.networkArray[networkNum].convStructs[n].convLayer5OutputNorm, 0, 1400);
                        for (int i = 0; i < 1400; i++)
                        {
                            predictorGui.networkArray[networkNum].convStructs[n].convLayer5OutputNorm[i] *= 100;
                        }
                    }

                    double[] temp = matOps.transposeMatCpu(predictorGui.networkArray[networkNum].convStructs[n].convLayer5OutputNorm, 100, 14);
                    Array.Copy(temp, 0, predictorGui.networkArray[networkNum].convStructs[n].convLayer5OutputNorm, 0, 1400);

                    temporalEncodingConCat(networkNum, n);
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[n].temporalEncodedNormOutput, 0, predictorGui.networkArray[networkNum].transStructs[n].transformerInput, 0, 1500);
                });
            }
            else
            {
                //initialize events array with passed in events param
                //we will leave first event empty as our causal padding
                //for (int i = 0; i < events + 1; i++)
                //{
                //    predictorGui.totalEventsArray[0].eventsArray[i] = new Event();
                //}

                //each tensor is 32 lines long, iterate through tensorIn
                //to retrieve
                int j = 1;
                int x = 0;
                for (int i = 0; i < (32 * events); i++)
                {
                    if (i % 32 == 0 && i != 0)
                    {
                        j++;
                        x = 0;
                    }
                    predictorGui.totalEventsArray[0].eventsArray[j].prices[x] = predictorGui.tensorIn[0].price[i];
                    predictorGui.totalEventsArray[0].eventsArray[j].sizes[x] = predictorGui.tensorIn[0].size[i];
                    if (j == 1)
                    {
                        //populate causal padding with copy of the first event, this is for replication padding
                        predictorGui.totalEventsArray[0].eventsArray[j - 1].prices[x] = predictorGui.tensorIn[0].price[i];
                        predictorGui.totalEventsArray[0].eventsArray[j - 1].sizes[x] = predictorGui.tensorIn[0].size[i];
                    }
                    x++;
                }
                //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                //{
                    //StreamWriter write = File.AppendText(@"X:\eventsTensor.txt");
                    //for (int i = 0; i < events + 1; i++)
                    //{
                    //    for (int k = 0; k < 32; k++)
                    //    {
                    //        write.WriteLine(predictorGui.totalEventsArray[0].eventsArray[i].prices[k].ToString() + " " + predictorGui.totalEventsArray[0].eventsArray[i].sizes[k].ToString());
                    //    }
                    //}
                    //write.Close();
                //}

                conv_full_layer1(networkNum, 0);
                conv_full_layer2(networkNum, 0);
                conv_full_layer3(networkNum, 0);
                conv_full_layer4(networkNum, 0);
                conv_full_layer5(networkNum, 0);

                if (predictorGui.predictorGui1.preluSelect.Checked == true || predictorGui.predictorGui1.mishSelect.Checked == true)
                {
                    layerNorm(networkNum, 0);
                }
                else if (predictorGui.predictorGui1.seluSelect.Checked == true)
                {
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[0].convLayer5Output, 0, predictorGui.networkArray[networkNum].convStructs[0].convLayer5OutputNorm, 0, 1400);
                    for (int i = 0; i < 1400; i++)
                    {
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer5OutputNorm[i] *= 100;
                    }
                }

                double[] temp = matOps.transposeMatCpu(predictorGui.networkArray[networkNum].convStructs[0].convLayer5OutputNorm, 100, 14);
                Array.Copy(temp, 0, predictorGui.networkArray[networkNum].convStructs[0].convLayer5OutputNorm, 0, 1400);

                temporalEncodingConCat(networkNum, 0);
                Array.Copy(predictorGui.networkArray[networkNum].convStructs[0].temporalEncodedNormOutput, 0, predictorGui.networkArray[networkNum].transStructs[0].transformerInput, 0, 1500);
            }
            
            
            /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output2 = File.AppendText(@"X:\normFeatureMap.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output2.WriteLine("Normalized Feature map[" + i.ToString() + "] = " + predictorGui.convStructs[0].convLayer5OutputNorm[i].ToString());
                }
                output2.WriteLine();
                output2.Close();
            }*/



            /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output2 = File.AppendText(@"X:\normFeatureMap_transposed.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output2.WriteLine("Normalized Feature map transposed[" + i.ToString() + "] = " + predictorGui.convStructs[0].convLayer5OutputNorm[i].ToString());
                }
                output2.WriteLine();
                output2.Close();
            }*/

            /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output1 = File.AppendText(@"X:\featureMap.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output1.WriteLine("Feature map[" + i.ToString() + "] = " + predictorGui.convStructs[0].convLayer5Output[i].ToString());
                }
                output1.WriteLine();
                output1.Close();
            }*/
        }

        public void conv_full_layer1(int networkNum, int exampleNum)
        {
            //CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            //CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolution");

            int idx = 0;

            double[] resVal;
            double[] convVal_GPU = new double[1400];
            double[] convVal_CPU = new double[1400];

            int N = 44800; //100 events * 32 depth = 3200 (with one 0 event as causal padding) * 14 kernels = 44800

            //kernel.BlockDimensions = 256;
            //kernel.GridDimensions = (N + 255) / 256;

            double[] h_P = new double[44800];
            double[] k_P = new double[44800];
            double[] h_S = new double[44800];
            double[] k_S = new double[44800];

            double[] h_P2 = new double[44800];
            double[] k_P2 = new double[44800];
            double[] h_S2 = new double[44800];
            double[] k_S2 = new double[44800];

            for (int kernel_idx = 0; kernel_idx < 14; kernel_idx++)
            {
                for (int i = 0; i < 100; i++)
                {
                    Array.Copy(predictorGui.totalEventsArray[exampleNum].eventsArray[i].prices, 0, h_P, idx, 32);
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Kernel1[kernel_idx].depth1, 0, k_P, idx, 32);
                    Array.Copy(predictorGui.totalEventsArray[exampleNum].eventsArray[i].sizes, 0, h_S, idx, 32);
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Kernel1[kernel_idx].depth2, 0, k_S, idx, 32);
                    Array.Copy(predictorGui.totalEventsArray[exampleNum].eventsArray[i + 1].prices, 0, h_P2, idx, 32);
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Kernel1[kernel_idx].depth3, 0, k_P2, idx, 32);
                    Array.Copy(predictorGui.totalEventsArray[exampleNum].eventsArray[i + 1].sizes, 0, h_S2, idx, 32);
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Kernel1[kernel_idx].depth4, 0, k_S2, idx, 32);
                    idx += 32;
                }
            }

            /*
            StreamWriter debug2 = File.AppendText(@"X:\debugOutput\h_P" + networkNum + "-" + exampleNum + ".txt");
            for (int i = 0; i < 44800; i++)
            {
                debug2.WriteLine(h_P[i].ToString());
            }
            debug2.Close();

            StreamWriter debug6 = File.AppendText(@"X:\debugOutput\h_S" + networkNum + "-" + exampleNum + ".txt");
            for (int i = 0; i < 44800; i++)
            {
                debug6.WriteLine(h_S[i].ToString());
            }
            debug6.Close();

            StreamWriter debug3 = File.AppendText(@"X:\debugOutput\k_P" + networkNum + "-" + exampleNum + ".txt");
            for (int i = 0; i < 44800; i++)
            {
                debug3.WriteLine(k_P[i].ToString());
            }
            debug3.Close();

            StreamWriter debug7 = File.AppendText(@"X:\debugOutput\k_S" + networkNum + "-" + exampleNum + ".txt");
            for (int i = 0; i < 44800; i++)
            {
                debug7.WriteLine(k_P[i].ToString());
            }
            debug7.Close();

            StreamWriter debug4 = File.AppendText(@"X:\debugOutput\h_P2" + networkNum + "-" + exampleNum + ".txt");
            for (int i = 0; i < 44800; i++)
            {
                debug4.WriteLine(h_P2[i].ToString());
            }
            debug4.Close();

            StreamWriter debug8 = File.AppendText(@"X:\debugOutput\h_S2" + networkNum + "-" + exampleNum + ".txt");
            for (int i = 0; i < 44800; i++)
            {
                debug8.WriteLine(h_P2[i].ToString());
            }
            debug8.Close();

            StreamWriter debug5 = File.AppendText(@"X:\debugOutput\k_P2" + networkNum + "-" + exampleNum + ".txt");
            for (int i = 0; i < 44800; i++)
            {
                debug5.WriteLine(k_P2[i].ToString());
            }
            debug5.Close();

            StreamWriter debug9 = File.AppendText(@"X:\debugOutput\k_S2" + networkNum + "-" + exampleNum + ".txt");
            for (int i = 0; i < 44800; i++)
            {
                debug9.WriteLine(k_P2[i].ToString());
            }
            debug9.Close();
            */
            /*CudaDeviceVariable<double> d_P = h_P;
            CudaDeviceVariable<double> d_S = h_S;
            CudaDeviceVariable<double> dk_P = k_P;
            CudaDeviceVariable<double> dk_S = k_S;

            CudaDeviceVariable<double> d_P2 = h_P2;
            CudaDeviceVariable<double> d_S2 = h_S2;
            CudaDeviceVariable<double> dk_P2 = k_P2;
            CudaDeviceVariable<double> dk_S2 = k_S2;

            CudaDeviceVariable<double> returnedVal = new CudaDeviceVariable<double>(N);*/
            /*
            output.Write("h_P[] = ");
            for (int i = 0; i < 32; i++)
            {
                output.Write(h_P[i].ToString() + " ");
            }
            output.WriteLine();
            output.Write("h_S[] = ");
            for (int i = 0; i < 32; i++)
            {
                output.Write(h_S[i].ToString() + " ");
            }
            output.WriteLine();
            output.Write("h_P2[] = ");
            for (int i = 0; i < 32; i++)
            {
                output.Write(h_P2[i].ToString() + " ");
            }
            output.WriteLine();
            output.Write("h_S2[] = ");
            for (int i = 0; i < 32; i++)
            {
                output.Write(h_S2[i].ToString() + " ");
            }
            output.WriteLine();
            output.Write("k_P[] = ");
            for (int i = 0; i < 32; i++)
            {
                output.Write(k_P[i].ToString() + " ");
            }
            output.WriteLine();
            output.Write("k_S[] = ");
            for (int i = 0; i < 32; i++)
            {
                output.Write(k_S[i].ToString() + " ");
            }
            output.WriteLine();
            output.Write("k_P2[] = ");
            for (int i = 0; i < 32; i++)
            {
                output.Write(k_P2[i].ToString() + " ");
            }
            output.WriteLine();
            output.Write("k_S2[] = ");
            for (int i = 0; i < 32; i++)
            {
                output.Write(k_S2[i].ToString() + " ");
            }
            output.WriteLine();
            */
            /*kernel.Run(d_P.DevicePointer, d_S.DevicePointer, dk_P.DevicePointer, dk_S.DevicePointer,
                d_P2.DevicePointer, d_S2.DevicePointer, dk_P2.DevicePointer, dk_S2.DevicePointer,
                returnedVal.DevicePointer, N);

            resVal = returnedVal;*/
            resVal = matOps.convolutionLayer1OnlyCpu(h_P, h_S, k_P, k_S, h_P2, h_S2, k_P2, k_S2, N);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\rawConvGPUOperation1.txt");
                for (int i = 0; i < N; i++)
                {
                    output.WriteLine(resVal[i].ToString());
                }
                output.Close();
            }

            int j = 0;
            for (int i = 0; i < 44800; i++)
            {
                if (i % 32 == 0 && i != 0)
                {
                    j++;
                }
                convVal_GPU[j] += resVal[i];
            }

            //CPU implementation of convolution operation running on GPU
            if (predictorGui.predictorGui1.confirmCPU.Checked == true)
            {
                int cpu_idx = 0;
                /*for (int m = 0; m < 14; m++)
                {
                    for (int k = 0; k < 100; k++)
                    {
                        for (int i = 0; i < 32; i++)
                        {
                            convVal_CPU[cpu_idx] += (predictorGui.eventsArray[k].prices[i] * predictorGui.convStructs[0].convLayer1Kernel1[m].depth1[i]) +
                                (predictorGui.eventsArray[k].sizes[i] * predictorGui.convStructs[0].convLayer1Kernel1[m].depth2[i]) +
                                (predictorGui.eventsArray[k + 1].prices[i] * predictorGui.convStructs[0].convLayer1Kernel1[m].depth3[i]) +
                                (predictorGui.eventsArray[k + 1].sizes[i] * predictorGui.convStructs[0].convLayer1Kernel1[m].depth4[i]);
                        }
                        cpu_idx++;
                    }
                }*/
                for (int i = 0; i < 1400; i++)
                {
                    if (Math.Round(convVal_GPU[i], 5) == Math.Round(convVal_CPU[i], 5))
                    {
                        predictorGui.predictorGui1.label3.Text = "Confirmed accurate using CPU calculation.";
                    }
                    else
                    {
                        predictorGui.predictorGui1.label3.Text = "Error in GPU calculated values detected: convVal_GPU[" + i.ToString() + "] != convVal_CPU[" + i.ToString() + "]";
                    }
                }
            }

            Array.Copy(convVal_GPU, predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output, 1400);
            if (predictorGui.predictorGui1.preluSelect.Checked == true)
            {
                convLayer1_PReLU_and_add_bias(networkNum, exampleNum);
            }
            else if (predictorGui.predictorGui1.seluSelect.Checked == true)
            {
                convLayer1_SeLU_and_add_bias(networkNum, exampleNum);
            }
            else if (predictorGui.predictorGui1.mishSelect.Checked == true)
            {
                convLayer1_Mish_and_add_bias(networkNum, exampleNum);
            }

            //StreamWriter debug = File.AppendText(@"X:\debugOutput\convLayer1Out" + networkNum + "-" + exampleNum + ".txt");
            //for(int i = 0; i < 1400; i++)
            //{
            //    debug.WriteLine(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i].ToString());
            //}
            //debug.Close();

            /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\convolutionOutput.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output.WriteLine("CPU convolved value = " + convVal_CPU[i].ToString() + "   GPU convolved value = " + convVal_GPU[i].ToString() +
                        "  Saved Conv Layer 1 Output = " + predictorGui.convStructs[0].convLayer1Output[i].ToString());
                }
                output.WriteLine();
                output.Close();
            }*/

            /*d_P.Dispose();
            d_S.Dispose();
            dk_P.Dispose();
            dk_S.Dispose();
            d_P2.Dispose();
            d_S2.Dispose();
            dk_P2.Dispose();
            dk_S2.Dispose();
            returnedVal.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();*/
        }

        public void conv_full_layer2(int networkNum, int exampleNum)
        {
            //CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            //CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolution2");

            int idx = 0;

            double[] resVal;
            double[] convVal_GPU = new double[1400];
            double[] convVal_CPU = new double[1400];
            double[] temp;

            //we transpose the output of convLayer1, this is to take care of the issue that my feature vectors were scrambled otherwise
            temp = matOps.transposeMatCpu(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output, 100, 14);

            double[] padded_input = new double[1428];
            Array.Copy(temp, 0, padded_input, 28, 1400);
            //implement replication padding
            Array.Copy(temp, 0, padded_input, 0, 28);
            Array.Copy(padded_input, 0, predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1OutPadded, 0, 1428);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter padded_input_array = File.AppendText(@"X:\padded_input_array_convLayer2.txt");
                for(int i = 0; i < 1428; i++)
                {
                    padded_input_array.WriteLine(padded_input[i]);
                }
                padded_input_array.Close();
            }

            int N = 19600; //100 events * 14 depth = 1400 (with two 0 events as causal padding) * 14 kernels = 19600

            //kernel.BlockDimensions = 256;
            //kernel.GridDimensions = (N + 255) / 256;

            double[] h_F = new double[19600];
            double[] k_F = new double[19600];

            double[] h_F2 = new double[19600];
            double[] k_F2 = new double[19600];

            int feature_idx = 0;

            //there is a bug here and in the next conv layers that has to do with how the feature vectors are being convolved on the GPU
            //UPDATE: added transposition before padding in order to handle convolution correctly.
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
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Kernel2[kernel_idx].depth1, 0, k_F, idx, 14);
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Kernel2[kernel_idx].depth2, 0, k_F2, idx, 14);
                    idx += 14;
                }
            }

            //StreamWriter feature_arrays = File.AppendText(@"X:\feature_arrays.txt");
            //for(int i = 0; i < 19600; i++)
            //{
            //    feature_arrays.WriteLine(h_F[i].ToString() + " " + h_F2[i].ToString() + "   " + k_F[i].ToString() + " " + k_F2[i].ToString());
            //}
            //feature_arrays.Close();

            /*CudaDeviceVariable<double> d_F = h_F;
            CudaDeviceVariable<double> dk_F = k_F;

            CudaDeviceVariable<double> d_F2 = h_F2;
            CudaDeviceVariable<double> dk_F2 = k_F2;

            CudaDeviceVariable<double> returnedVal = new CudaDeviceVariable<double>(N);
            kernel.Run(d_F.DevicePointer, dk_F.DevicePointer, d_F2.DevicePointer, dk_F2.DevicePointer, returnedVal.DevicePointer, N);*/

            //resVal = returnedVal;
            resVal = matOps.convolutionLayer2thru5Cpu(h_F, k_F, h_F2, k_F2, N);

            int j = 0;
            for (int i = 0; i < 19600; i++)
            {
                if (i % 14 == 0 && i != 0)
                {
                    j++;
                }
                convVal_GPU[j] += resVal[i];
            }

            /*CPU implementation of convolution operation running on GPU
            if (predictorGui.predictorGui1.confirmCPU.Checked == true)
            {
                int cpu_idx = 0;
                for (int m = 0; m < 14; m++)
                {
                    for (int k = 0; k < 100; k++)
                    {
                        for (int i = 0; i < 14; i++)
                        {
                            convVal_CPU[cpu_idx] += (predictorGui.eventsArray[k].prices[i] * convLayer2Kernel2[m].depth1[i]) +
                                (predictorGui.eventsArray[k].sizes[i] * convLayer2Kernel2[m].depth2[i]);
                        }
                        cpu_idx++;
                    }
                }
                for (int i = 0; i < 1400; i++)
                {
                    if (convVal_GPU[i] == convVal_CPU[i])
                    {
                        predictorGui.predictorGui1.label3.Text = "Confirmed accurate using CPU calculation.";
                    }
                    else
                    {
                        predictorGui.predictorGui1.label3.Text = "Error in GPU calculated values detected: convVal_GPU[" + i.ToString() + "] != convVal_CPU[" + i.ToString() + "]";
                    }
                }
            }
            */

            Array.Copy(convVal_GPU, predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output, 1400);
            if (predictorGui.predictorGui1.preluSelect.Checked == true)
            {
                convLayer2_PReLU_and_add_bias(networkNum, exampleNum);
            }
            else if (predictorGui.predictorGui1.seluSelect.Checked == true)
            {
                convLayer2_SeLU_and_add_bias(networkNum, exampleNum);
            }
            else if (predictorGui.predictorGui1.mishSelect.Checked == true)
            {
                convLayer2_Mish_and_add_bias(networkNum, exampleNum);
            }

            //StreamWriter debug = File.AppendText(@"X:\debugOutput\convLayer2Out" + networkNum + "-" + exampleNum + ".txt");
            //for (int i = 0; i < 1400; i++)
            //{
            //    debug.WriteLine(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i].ToString());
            //}
            //debug.Close();

            /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            { 
                StreamWriter output = File.AppendText(@"X:\convolutionOutput2.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output.WriteLine("CPU convolved value = " + convVal_CPU[i].ToString() + "   GPU convolved value = " + convVal_GPU[i].ToString() +
                        "  Saved Conv Layer 2 Output = " + predictorGui.convStructs[0].convLayer2Output[i].ToString());
                }
                output.WriteLine();
                output.Close();
            }*/

            /*d_F.Dispose();
            dk_F.Dispose();
            d_F2.Dispose();
            dk_F2.Dispose();
            returnedVal.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();*/
        }

        public void conv_full_layer3(int networkNum, int exampleNum)
        {
            //CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            //CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolution2");

            int idx = 0;

            double[] resVal = new double[19600];
            double[] convVal_GPU = new double[1400];
            double[] convVal_CPU = new double[1400];
            double[] temp;

            //we transpose the output of convLayer2, this is to take care of the issue that my feature vectors were scrambled otherwise
            temp = matOps.transposeMatCpu(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output, 100, 14);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\convolutionOutput2_transposed.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output.WriteLine(temp[i].ToString());
                }
                output.WriteLine();
                output.Close();
            }

            double[] padded_input = new double[1456];
            Array.Copy(temp, 0, padded_input, 56, 1400);
            //implement replication padding
            Array.Copy(temp, 0, padded_input, 0, 56);
            Array.Copy(padded_input, 0, predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2OutPadded, 0, 1456);

            //StreamWriter padded_input_array = File.AppendText(@"X:\padded_input_array.txt");
            //for(int i = 0; i < 1428; i++)
            //{
            //    padded_input_array.WriteLine(padded_input[i]);
            //}
            //padded_input_array.Close();

            int N = 19600; //100 events * 14 depth = 1400 (with four 0 events as causal padding) * 14 kernels = 19600

            //int threads = 32;
            //int blocks = (N + threads - 1) / threads;
            //kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(threads, threads, 1);
            //kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3(blocks, blocks, 1);
            //kernel.BlockDimensions = 256;
            //kernel.GridDimensions = (N + 255) / 256;

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
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Kernel3[kernel_idx].depth1, 0, k_F, idx, 14);
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Kernel3[kernel_idx].depth2, 0, k_F2, idx, 14);
                    idx += 14;
                }
            }

            //StreamWriter feature_arrays = File.AppendText(@"X:\feature_arrays.txt");
            //for(int i = 0; i < 19600; i++)
            //{
            //    feature_arrays.WriteLine(h_F[i].ToString() + " " + h_F2[i].ToString() + "   " + k_F[i].ToString() + " " + k_F2[i].ToString());
            //}
            //feature_arrays.Close();

            /*CudaDeviceVariable<double> d_F = h_F;
            CudaDeviceVariable<double> dk_F = k_F;

            CudaDeviceVariable<double> d_F2 = h_F2;
            CudaDeviceVariable<double> dk_F2 = k_F2;

            CudaDeviceVariable<double> returnedVal = new CudaDeviceVariable<double>(N);
            kernel.Run(d_F.DevicePointer, dk_F.DevicePointer, d_F2.DevicePointer, dk_F2.DevicePointer, returnedVal.DevicePointer, N);

            resVal = returnedVal;*/
            resVal = matOps.convolutionLayer2thru5Cpu(h_F, k_F, h_F2, k_F2, N);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\rawConvGPUOperation3.txt");
                for(int i = 0; i < N; i++)
                {
                    output.WriteLine(resVal[i].ToString());
                }
                output.Close();
            }

            int j = 0;
            for (int i = 0; i < 19600; i++)
            {
                if (i % 14 == 0 && i != 0)
                {
                    j++;
                }
                convVal_GPU[j] += resVal[i];
            }

            /*CPU implementation of convolution operation running on GPU
            if (predictorGui.predictorGui1.confirmCPU.Checked == true)
            {
                int cpu_idx = 0;
                for (int m = 0; m < 14; m++)
                {
                    for (int k = 0; k < 100; k++)
                    {
                        for (int i = 0; i < 14; i++)
                        {
                            convVal_CPU[cpu_idx] += (predictorGui.eventsArray[k].prices[i] * convLayer3Kernel3[m].depth1[i]) +
                                (predictorGui.eventsArray[k].sizes[i] * convLayer3Kernel3[m].depth2[i]);
                        }
                        cpu_idx++;
                    }
                }
                for (int i = 0; i < 1400; i++)
                {
                    if (convVal_GPU[i] == convVal_CPU[i])
                    {
                        predictorGui.predictorGui1.label3.Text = "Confirmed accurate using CPU calculation.";
                    }
                    else
                    {
                        predictorGui.predictorGui1.label3.Text = "Error in GPU calculated values detected: convVal_GPU[" + i.ToString() + "] != convVal_CPU[" + i.ToString() + "]";
                    }
                }
            }*/

            Array.Copy(convVal_GPU, predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output, 1400);
            if (predictorGui.predictorGui1.preluSelect.Checked == true)
            {
                convLayer3_PReLU_and_add_bias(networkNum, exampleNum);
            }
            else if (predictorGui.predictorGui1.seluSelect.Checked == true)
            {
                convLayer3_SeLU_and_add_bias(networkNum, exampleNum);
            }
            else if (predictorGui.predictorGui1.mishSelect.Checked == true)
            {
                convLayer3_Mish_and_add_bias(networkNum, exampleNum);
            }

            //StreamWriter debug = File.AppendText(@"X:\debugOutput\convLayer3Out" + networkNum + "-" + exampleNum + ".txt");
            //for (int i = 0; i < 1400; i++)
            //{
            //    debug.WriteLine(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i].ToString());
            //}
            //debug.Close();

            /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\convolutionOutput3.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output.WriteLine("CPU convolved value = " + convVal_CPU[i].ToString() + "   GPU convolved value = " + convVal_GPU[i].ToString() +
                        "  Saved Conv Layer 3 Output = " + predictorGui.convStructs[0].convLayer3Output[i].ToString());
                }
                output.WriteLine();
                output.Close();
            }*/

            /*d_F.Dispose();
            dk_F.Dispose();
            d_F2.Dispose();
            dk_F2.Dispose();
            returnedVal.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();*/
        }

        public void conv_full_layer4(int networkNum, int exampleNum)
        {
            //CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            //CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolution2");

            int idx = 0;

            double[] resVal;
            double[] convVal_GPU = new double[1400];
            double[] convVal_CPU = new double[1400];

            double[] temp;

            //we transpose the output of convLayer3, this is to take care of the issue that my feature vectors were scrambled otherwise
            temp = matOps.transposeMatCpu(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output, 100, 14);

            double[] padded_input = new double[1512];
            Array.Copy(temp, 0, padded_input, 112, 1400);
            //implement replication padding
            Array.Copy(temp, 0, padded_input, 0, 112);
            Array.Copy(padded_input, 0, predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3OutPadded, 0, 1512);

            //StreamWriter padded_input_array = File.AppendText(@"X:\padded_input_array.txt");
            //for(int i = 0; i < 1428; i++)
            //{
            //    padded_input_array.WriteLine(padded_input[i]);
            //}
            //padded_input_array.Close();

            int N = 19600; //100 events * 14 depth = 1400 (with eight 0 events as causal padding) * 14 kernels = 19600

            //kernel.BlockDimensions = 256;
            //kernel.GridDimensions = (N + 255) / 256;

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
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Kernel4[kernel_idx].depth1, 0, k_F, idx, 14);
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Kernel4[kernel_idx].depth2, 0, k_F2, idx, 14);
                    idx += 14;
                }
            }

            //StreamWriter feature_arrays = File.AppendText(@"X:\feature_arrays.txt");
            //for(int i = 0; i < 19600; i++)
            //{
            //    feature_arrays.WriteLine(h_F[i].ToString() + " " + h_F2[i].ToString() + "   " + k_F[i].ToString() + " " + k_F2[i].ToString());
            //}
            //feature_arrays.Close();

            /*CudaDeviceVariable<double> d_F = h_F;
            CudaDeviceVariable<double> dk_F = k_F;

            CudaDeviceVariable<double> d_F2 = h_F2;
            CudaDeviceVariable<double> dk_F2 = k_F2;

            CudaDeviceVariable<double> returnedVal = new CudaDeviceVariable<double>(N);
            kernel.Run(d_F.DevicePointer, dk_F.DevicePointer, d_F2.DevicePointer, dk_F2.DevicePointer, returnedVal.DevicePointer, N);

            resVal = returnedVal;*/
            resVal = matOps.convolutionLayer2thru5Cpu(h_F, k_F, h_F2, k_F2, N);

            int j = 0;
            for (int i = 0; i < 19600; i++)
            {
                if (i % 14 == 0 && i != 0)
                {
                    j++;
                }
                convVal_GPU[j] += resVal[i];
            }

            /*CPU implementation of convolution operation running on GPU
            if (predictorGui.predictorGui1.confirmCPU.Checked == true)
            {
                int cpu_idx = 0;
                for (int m = 0; m < 14; m++)
                {
                    for (int k = 0; k < 100; k++)
                    {
                        for (int i = 0; i < 14; i++)
                        {
                            convVal_CPU[cpu_idx] += (predictorGui.eventsArray[k].prices[i] * convLayer1Kernel1[m].depth1[i]) +
                                (predictorGui.eventsArray[k].sizes[i] * convLayer1Kernel1[m].depth2[i]) +
                                (predictorGui.eventsArray[k + 2].prices[i] * convLayer1Kernel1[m].depth3[i]) +
                                (predictorGui.eventsArray[k + 2].sizes[i] * convLayer1Kernel1[m].depth4[i]);
                        }
                        cpu_idx++;
                    }
                }
                for (int i = 0; i < 1400; i++)
                {
                    if (convVal_GPU[i] == convVal_CPU[i])
                    {
                        predictorGui.predictorGui1.label3.Text = "Confirmed accurate using CPU calculation.";
                    }
                    else
                    {
                        predictorGui.predictorGui1.label3.Text = "Error in GPU calculated values detected: convVal_GPU[" + i.ToString() + "] != convVal_CPU[" + i.ToString() + "]";
                    }
                }
            }*/

            Array.Copy(convVal_GPU, predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output, 1400);
            if (predictorGui.predictorGui1.preluSelect.Checked == true)
            {
                convLayer4_PReLU_and_add_bias(networkNum, exampleNum);
            }
            else if (predictorGui.predictorGui1.seluSelect.Checked == true)
            {
                convLayer4_SeLU_and_add_bias(networkNum, exampleNum);
            }
            else if (predictorGui.predictorGui1.mishSelect.Checked == true)
            {
                convLayer4_Mish_and_add_bias(networkNum, exampleNum);
            }

            //StreamWriter debug = File.AppendText(@"X:\debugOutput\convLayer4Out" + networkNum + "-" + exampleNum + ".txt");
            //for (int i = 0; i < 1400; i++)
            //{
            //    debug.WriteLine(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i].ToString());
            //}
            //debug.Close();

            /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\convolutionOutput4.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output.WriteLine("CPU convolved value = " + convVal_CPU[i].ToString() + "   GPU convolved value = " + convVal_GPU[i].ToString() +
                        "  Saved Conv Layer 4 Output = " + predictorGui.convStructs[0].convLayer4Output[i].ToString());
                }
                output.WriteLine();
                output.Close();
            }*/

            /*d_F.Dispose();
            dk_F.Dispose();
            d_F2.Dispose();
            dk_F2.Dispose();
            returnedVal.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();*/
        }

        public void conv_full_layer5(int networkNum, int exampleNum)
        {
            //CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            //CudaKernel kernel = ctx.LoadKernel("convolution.ptx", "convolution2");

            int idx = 0;

            double[] resVal;
            double[] convVal_GPU = new double[1400];
            double[] convVal_CPU = new double[1400];

            double[] temp;

            //we transpose the output of convLayer4, this is to take care of the issue that my feature vectors were scrambled otherwise
            temp = matOps.transposeMatCpu(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output, 100, 14);

            double[] padded_input = new double[1624];
            Array.Copy(temp, 0, padded_input, 224, 1400);
            //implement replication padding
            Array.Copy(temp, 0, padded_input, 0, 224);
            Array.Copy(padded_input, 0, predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4OutPadded, 0, 1624);
            
            //StreamWriter padded_input_array = File.AppendText(@"X:\padded_input_array.txt");
            //for(int i = 0; i < 1428; i++)
            //{
            //    padded_input_array.WriteLine(padded_input[i]);
            //}
            //padded_input_array.Close();

            int N = 19600; //100 events * 14 depth = 1400 (with sixteen 0 events as causal padding) * 14 kernels = 19600

            //kernel.BlockDimensions = 256;
            //kernel.GridDimensions = (N + 255) / 256;

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
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Kernel5[kernel_idx].depth1, 0, k_F, idx, 14);
                    Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Kernel5[kernel_idx].depth2, 0, k_F2, idx, 14);
                    idx += 14;
                }
            }

            //StreamWriter feature_arrays = File.AppendText(@"X:\feature_arrays.txt");
            //for(int i = 0; i < 19600; i++)
            //{
            //    feature_arrays.WriteLine(h_F[i].ToString() + " " + h_F2[i].ToString() + "   " + k_F[i].ToString() + " " + k_F2[i].ToString());
            //}
            //feature_arrays.Close();

            /*CudaDeviceVariable<double> d_F = h_F;
            CudaDeviceVariable<double> dk_F = k_F;

            CudaDeviceVariable<double> d_F2 = h_F2;
            CudaDeviceVariable<double> dk_F2 = k_F2;

            CudaDeviceVariable<double> returnedVal = new CudaDeviceVariable<double>(N);
            kernel.Run(d_F.DevicePointer, dk_F.DevicePointer, d_F2.DevicePointer, dk_F2.DevicePointer, returnedVal.DevicePointer, N);*/

            //resVal = returnedVal;

            resVal = matOps.convolutionLayer2thru5Cpu(h_F, k_F, h_F2, k_F2, N);

            int j = 0;
            for (int i = 0; i < 19600; i++)
            {
                if (i % 14 == 0 && i != 0)
                {
                    j++;
                }
                convVal_GPU[j] += resVal[i];
            }

            /*CPU implementation of convolution operation running on GPU
            if (predictorGui.predictorGui1.confirmCPU.Checked == true)
            {
                int cpu_idx = 0;
                for (int m = 0; m < 14; m++)
                {
                    for (int k = 0; k < 100; k++)
                    {
                        for (int i = 0; i < 14; i++)
                        {
                            convVal_CPU[cpu_idx] += (predictorGui.eventsArray[k].prices[i] * convLayer1Kernel1[m].depth1[i]) +
                                (predictorGui.eventsArray[k].sizes[i] * convLayer1Kernel1[m].depth2[i]) +
                                (predictorGui.eventsArray[k + 2].prices[i] * convLayer1Kernel1[m].depth3[i]) +
                                (predictorGui.eventsArray[k + 2].sizes[i] * convLayer1Kernel1[m].depth4[i]);
                        }
                        cpu_idx++;
                    }
                }
                for (int i = 0; i < 1400; i++)
                {
                    if (convVal_GPU[i] == convVal_CPU[i])
                    {
                        predictorGui.predictorGui1.label3.Text = "Confirmed accurate using CPU calculation.";
                    }
                    else
                    {
                        predictorGui.predictorGui1.label3.Text = "Error in GPU calculated values detected: convVal_GPU[" + i.ToString() + "] != convVal_CPU[" + i.ToString() + "]";
                    }
                }
            }
            */
            Array.Copy(convVal_GPU, predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output, 1400);
            if (predictorGui.predictorGui1.preluSelect.Checked == true)
            {
                convLayer5_PReLU_and_add_bias(networkNum, exampleNum);
            }
            else if (predictorGui.predictorGui1.seluSelect.Checked == true)
            {
                convLayer5_SeLU_and_add_bias(networkNum, exampleNum);
            }
            else if (predictorGui.predictorGui1.mishSelect.Checked == true)
            {
                convLayer5_Mish_and_add_bias(networkNum, exampleNum);
            }

            //StreamWriter debug = File.AppendText(@"X:\debugOutput\convLayer5Out" + networkNum + "-" + exampleNum + ".txt");
            //for (int i = 0; i < 1400; i++)
            //{
            //    debug.WriteLine(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i].ToString());
            //}
            //debug.Close();

            /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\convolutionOutput5.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output.WriteLine("CPU convolved value = " + convVal_CPU[i].ToString() + "   GPU convolved value = " + convVal_GPU[i].ToString() +
                        "  Saved Conv Layer 5 Output = " + predictorGui.convStructs[0].convLayer5Output[i].ToString());
                }
                output.WriteLine();
                output.Close();
            }*/

            /*d_F.Dispose();
            dk_F.Dispose();
            d_F2.Dispose();
            dk_F2.Dispose();
            returnedVal.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();*/
        }

        public void layerNorm(int networkNum, int exampleNum)
        {
            double var_summation = 0;
            double feature_summation = 0;
            int meanVarianceIdx = 0;
            predictorGui.networkArray[networkNum].convStructs[exampleNum].epsilon = backProp.layerNormEpsilon; // 1 * (10 ^ -5) APPARENTLY C# doesn't know how to do scientific notation
                             //one times ten to the -5 dumbass, apparently it should be 1e-05

            //needs to be redesigned to execute normalization across each feature row, instead of how it is currently applying
            //normalization across the entire layer mean and variance (NOTE: Done on 5/25/2022 for conv module forward pass)
            for (int i = 0; i <= 1400; i++)
            {
                if(i % 100 == 0 && i != 0)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].mean[meanVarianceIdx] = feature_summation / 100;
                    feature_summation = 0;
                    meanVarianceIdx++;
                    if(i == 1400)
                    {
                        break;
                    }
                }
                feature_summation += predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i];
            }

            meanVarianceIdx = 0;
            for (int i = 0; i <= 1400; i++)
            {
                if (i % 100 == 0 && i != 0)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].variance[meanVarianceIdx] = var_summation / 100;
                    var_summation = 0;
                    meanVarianceIdx++;
                    if (i == 1400)
                    {
                        break;
                    }
                }
                var_summation += Math.Pow((predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] - predictorGui.networkArray[networkNum].convStructs[exampleNum].mean[meanVarianceIdx]), 2);
            }

            meanVarianceIdx = 0;
            for (int i = 0; i <= 1400; i++)
            {
                if(i % 100 == 0 && i != 0)
                {
                    meanVarianceIdx++;
                    if (i == 1400)
                    {
                        break;
                    }
                }
                predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5OutputNorm[i] = (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] - predictorGui.networkArray[networkNum].convStructs[exampleNum].mean[meanVarianceIdx]) / 
                                                        (Math.Sqrt(predictorGui.networkArray[networkNum].convStructs[exampleNum].variance[meanVarianceIdx] + predictorGui.networkArray[networkNum].convStructs[exampleNum].epsilon));
            }
            scaleAndShift(networkNum, exampleNum);

            /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\layerNormVariables.txt");
                for (int i = 0; i < 14; i++)
                {
                    output.WriteLine("mean = " + predictorGui.convStructs[0].mean[i].ToString());
                    output.WriteLine("variance = " + predictorGui.convStructs[0].variance[i].ToString() + "   epsilon = " + predictorGui.convStructs[0].epsilon.ToString());
                }
                output.Close();
            }*/

            //StreamWriter debug = File.AppendText(@"X:\debugOutput\convLayer5OutNorm" + networkNum + "-" + exampleNum + ".txt");
            //for (int i = 0; i < 1400; i++)
            //{
            //    debug.WriteLine(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5OutputNorm[i].ToString());
            //}
            //debug.Close();
        }

        public void scaleAndShift(int networkNum, int exampleNum)
        {
            for(int i = 0; i < 1400; i++)
            {
                predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5OutputNorm[i] = (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5OutputNormGamma[i] * predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5OutputNorm[i]) +
                                                            predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5OutputNormBeta[i];
            }
        }

        public void temporalEncodingConCat(int networkNum, int exampleNum)
        {
            int sourceIdx = 0;
            int destIdx = 0;
            int vecLen = 14;

            for (int i = 0; i < 100; i++)
            {
                Array.Copy(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5OutputNorm, sourceIdx, predictorGui.networkArray[networkNum].convStructs[exampleNum].temporalEncodedNormOutput, destIdx, vecLen);
                destIdx += 15;
                predictorGui.networkArray[networkNum].convStructs[exampleNum].temporalEncodedNormOutput[destIdx - 1] = ((2.0 / (100.0 - 1.0)) * i - 1.0);
                sourceIdx += 14;
            }
            /*if(predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter temporalEncodingOutput = File.AppendText(@"X:\temporalEncodedFeatureMap.txt");
                for(int i = 0; i < 1500; i++)
                {
                    temporalEncodingOutput.WriteLine("Temporal Encoded Feature map[" + i.ToString() + "] = " + predictorGui.networkArray[networkNum].convStructs[exampleNum].temporalEncodedNormOutput[i].ToString());
                }
                temporalEncodingOutput.Close();
            }*/

            //StreamWriter debug = File.AppendText(@"X:\debugOutput\temporalEncodedNormOut" + networkNum + "-" + exampleNum + ".txt");
            //for (int i = 0; i < 1500; i++)
            //{
            //    debug.WriteLine(predictorGui.networkArray[networkNum].convStructs[exampleNum].temporalEncodedNormOutput[i].ToString());
            //}
            //debug.Close();
        }

        public void convLayer1_PReLU_and_add_bias(int networkNum, int exampleNum)
        {
            for(int i = 0; i < 1400; i++)
            {
                if((predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Bias[i]) > 0.0F)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i] += predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Bias[i];
                }
                else
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i] = 0;
                }
            }
        }

        public void convLayer2_PReLU_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                if ((predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Bias[i]) > 0.0F)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i] += predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Bias[i];
                }
                else
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i] = 0; //(convLayer2Output[i] + convLayer2Bias[i]) * convLayer2PReLUParam[i];
                }
            }
        }

        public void convLayer3_PReLU_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                if ((predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Bias[i]) > 0.0F)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i] += predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Bias[i];
                }
                else
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i] = 0;
                }
            }
        }

        public void convLayer4_PReLU_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                if ((predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Bias[i]) > 0.0F)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i] += predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Bias[i];
                }
                else
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i] = 0;
                }
            }
        }

        public void convLayer5_PReLU_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                if ((predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Bias[i]) > 0.0F)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] += predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Bias[i];
                }
                else
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] = 0;
                }
            }
        }

        public void convLayer1_SeLU_and_add_bias(int networkNum, int exampleNum)
        {
            for(int i = 0; i < 1400; i++)
            {
                if((predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Bias[i]) > 0.0F)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i] = SeLU_lambda * (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Bias[i]);
                }
                else
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i] = SeLU_lambda * (SeLU_alpha * Math.Exp(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i] + 
                                                                        predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Bias[i]) - SeLU_alpha);
                }
            }
        }

        public void convLayer2_SeLU_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                if ((predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Bias[i]) > 0.0F)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i] = SeLU_lambda * (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Bias[i]);
                }
                else
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i] = SeLU_lambda * (SeLU_alpha * Math.Exp(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i] + 
                                                                        predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Bias[i]) - SeLU_alpha);
                }
            }
        }

        public void convLayer3_SeLU_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                if ((predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Bias[i]) > 0.0F)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i] = SeLU_lambda * (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Bias[i]);
                }
                else
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i] = SeLU_lambda * (SeLU_alpha * Math.Exp(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i] + 
                                                                        predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Bias[i]) - SeLU_alpha);
                }
            }
        }

        public void convLayer4_SeLU_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                if ((predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Bias[i]) > 0.0F)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i] = SeLU_lambda * (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Bias[i]);
                }
                else
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i] = SeLU_lambda * (SeLU_alpha * Math.Exp(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i] + 
                                                                        predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Bias[i]) - SeLU_alpha);
                }
            }
        }

        public void convLayer5_SeLU_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                if ((predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Bias[i]) > 0.0F)
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] = SeLU_lambda * (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Bias[i]);
                }
                else
                {
                    predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] = SeLU_lambda * (SeLU_alpha * Math.Exp(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] + 
                                                                        predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Bias[i]) - SeLU_alpha);
                }
            }
        }

        public void convLayer1_Mish_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i] = (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Bias[i]) * 
                    Math.Tanh(softplus(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer1Bias[i]));
            }
        }

        public void convLayer2_Mish_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i] = (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Bias[i]) * 
                    Math.Tanh(softplus(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer2Bias[i]));
            }
        }

        public void convLayer3_Mish_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i] = (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Bias[i]) * 
                    Math.Tanh(softplus(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer3Bias[i]));
            }
        }

        public void convLayer4_Mish_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i] = (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Bias[i]) * 
                    Math.Tanh(softplus(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer4Bias[i]));
            }
        }

        public void convLayer5_Mish_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 1400; i++)
            {
                predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] = (predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Bias[i]) *
                    Math.Tanh(softplus(predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Output[i] + predictorGui.networkArray[networkNum].convStructs[exampleNum].convLayer5Bias[i]));
            }
        }
        public double softplus(double x)
        {
            double temp;
            temp = Math.Log(1 + Math.Exp(x));
            return temp;
        }

        public void convLayerBiases_init()
        {
            if (!File.Exists(@"X:\convLayerBias1FlatFile.txt"))
            {
                StreamWriter output = File.AppendText(@"X:\convLayerBias1FlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\convLayerBias2FlatFile.txt");
                StreamWriter output3 = File.AppendText(@"X:\convLayerBias3FlatFile.txt");
                StreamWriter output4 = File.AppendText(@"X:\convLayerBias4FlatFile.txt");
                StreamWriter output5 = File.AppendText(@"X:\convLayerBias5FlatFile.txt");
                for (int i = 0; i < 1400; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer1Bias[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Bias[i]);
                    predictorGui.networkArray[0].convStructs[0].convLayer2Bias[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Bias[i]);
                    predictorGui.networkArray[0].convStructs[0].convLayer3Bias[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output3.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Bias[i]);
                    predictorGui.networkArray[0].convStructs[0].convLayer4Bias[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output4.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Bias[i]);
                    predictorGui.networkArray[0].convStructs[0].convLayer5Bias[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output5.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Bias[i]);
                }
                output.Close();
                output2.Close();
                output3.Close();
                output4.Close();
                output5.Close();
            }
            else
            {
                string[] arr = File.ReadAllLines(@"X:\convLayerBias1FlatFile.txt");
                string[] arr2 = File.ReadAllLines(@"X:\convLayerBias2FlatFile.txt");
                string[] arr3 = File.ReadAllLines(@"X:\convLayerBias3FlatFile.txt");
                string[] arr4 = File.ReadAllLines(@"X:\convLayerBias4FlatFile.txt");
                string[] arr5 = File.ReadAllLines(@"X:\convLayerBias5FlatFile.txt");
                for (int i = 0; i < 1400; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer1Bias[i] = Convert.ToDouble(arr[i]);
                    predictorGui.numOfLearnableParams++;
                    predictorGui.networkArray[0].convStructs[0].convLayer2Bias[i] = Convert.ToDouble(arr2[i]);
                    predictorGui.numOfLearnableParams++;
                    predictorGui.networkArray[0].convStructs[0].convLayer3Bias[i] = Convert.ToDouble(arr3[i]);
                    predictorGui.numOfLearnableParams++;
                    predictorGui.networkArray[0].convStructs[0].convLayer4Bias[i] = Convert.ToDouble(arr4[i]);
                    predictorGui.numOfLearnableParams++;
                    predictorGui.networkArray[0].convStructs[0].convLayer5Bias[i] = Convert.ToDouble(arr5[i]);
                    predictorGui.numOfLearnableParams++;
                }
            }
        }

        public void convLayerPReLUParams_init()
        {
            if (!File.Exists(@"X:\convLayerPReLUParams1FlatFile.txt"))
            {
                StreamWriter output = File.AppendText(@"X:\convLayerPReLUParams1FlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\convLayerPReLUParams2FlatFile.txt");
                StreamWriter output3 = File.AppendText(@"X:\convLayerPReLUParams3FlatFile.txt");
                StreamWriter output4 = File.AppendText(@"X:\convLayerPReLUParams4FlatFile.txt");
                StreamWriter output5 = File.AppendText(@"X:\convLayerPReLUParams5FlatFile.txt");
                for (int i = 0; i < 1400; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer1PReLUParam[i] = 0.02F;
                    predictorGui.numOfLearnableParams++;
                    output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1PReLUParam[i]);
                    predictorGui.networkArray[0].convStructs[0].convLayer2PReLUParam[i] = 0.02F;
                    predictorGui.numOfLearnableParams++;
                    output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2PReLUParam[i]);
                    predictorGui.networkArray[0].convStructs[0].convLayer3PReLUParam[i] = 0.02F;
                    predictorGui.numOfLearnableParams++;
                    output3.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3PReLUParam[i]);
                    predictorGui.networkArray[0].convStructs[0].convLayer4PReLUParam[i] = 0.02F;
                    predictorGui.numOfLearnableParams++;
                    output4.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4PReLUParam[i]);
                    predictorGui.networkArray[0].convStructs[0].convLayer5PReLUParam[i] = 0.02F;
                    predictorGui.numOfLearnableParams++;
                    output5.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5PReLUParam[i]);
                }
                output.Close();
                output2.Close();
                output3.Close();
                output4.Close();
                output5.Close();
            }
            else
            {
                string[] arr = File.ReadAllLines(@"X:\convLayerPReLUParams1FlatFile.txt");
                string[] arr2 = File.ReadAllLines(@"X:\convLayerPReLUParams2FlatFile.txt");
                string[] arr3 = File.ReadAllLines(@"X:\convLayerPReLUParams3FlatFile.txt");
                string[] arr4 = File.ReadAllLines(@"X:\convLayerPReLUParams4FlatFile.txt");
                string[] arr5 = File.ReadAllLines(@"X:\convLayerPReLUParams5FlatFile.txt");
                for (int i = 0; i < 1400; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer1PReLUParam[i] = Convert.ToDouble(arr[i]);
                    predictorGui.numOfLearnableParams++;
                    predictorGui.networkArray[0].convStructs[0].convLayer2PReLUParam[i] = Convert.ToDouble(arr2[i]);
                    predictorGui.numOfLearnableParams++;
                    predictorGui.networkArray[0].convStructs[0].convLayer3PReLUParam[i] = Convert.ToDouble(arr3[i]);
                    predictorGui.numOfLearnableParams++;
                    predictorGui.networkArray[0].convStructs[0].convLayer4PReLUParam[i] = Convert.ToDouble(arr4[i]);
                    predictorGui.numOfLearnableParams++;
                    predictorGui.networkArray[0].convStructs[0].convLayer5PReLUParam[i] = Convert.ToDouble(arr5[i]);
                    predictorGui.numOfLearnableParams++;
                }
            }
        }

        public void convLayerNormGammaBetaInit()
        {
            if (!File.Exists(@"X:\convLayerNormGammaFlatFile.txt"))
            {
                StreamWriter output = File.AppendText(@"X:\convLayerNormGammaFlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\convLayerNormBetaFlatFile.txt");
                for (int i = 0; i < 1400; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormGamma[i] = 1;
                    predictorGui.numOfLearnableParams++;
                    output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormGamma[i].ToString());
                    predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormBeta[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormBeta[i].ToString());
                }
                output.Close();
                output2.Close();
            }
            else
            {
                string[] arr = File.ReadAllLines(@"X:\convLayerNormGammaFlatFile.txt");
                string[] arr2 = File.ReadAllLines(@"X:\convLayerNormBetaFlatFile.txt");
                for (int i = 0; i < 1400; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormGamma[i] = Convert.ToDouble(arr[i]);
                    predictorGui.numOfLearnableParams++;
                    predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormBeta[i] = Convert.ToDouble(arr2[i]);
                    predictorGui.numOfLearnableParams++;
                }
            }
        }

        public void kaiming_he_init_layer(int layerNum)
        {
            if (layerNum == 1)
            {
                //number of input nodes from previous layer
                int n = 32 * 100 * 2;//6400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(2.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i] = new inputTensorKernel();
                    if (!File.Exists(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        StreamWriter output3 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth3FlatFile.txt");
                        StreamWriter output4 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth4FlatFile.txt");
                        for (int j = 0; j < 32; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth2[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth3[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output3.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth3[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth4[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output4.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth4[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                        output3.Close();
                        output4.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        string[] arr3 = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth3FlatFile.txt");
                        string[] arr4 = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth4FlatFile.txt");
                        for (int j = 0; j < 32; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth3[j] = Convert.ToDouble(arr3[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth4[j] = Convert.ToDouble(arr4[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if(layerNum == 2)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(2.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if (layerNum == 3)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(2.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if (layerNum == 4)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(2.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if (layerNum == 5)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(2.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
        }

        public void lecun_normal_init_layer(int layerNum)
        {
            if (layerNum == 1)
            {
                //number of input nodes from previous layer
                int n = 32 * 100 * 2;//6400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(1.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i] = new inputTensorKernel();
                    if (!File.Exists(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        StreamWriter output3 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth3FlatFile.txt");
                        StreamWriter output4 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth4FlatFile.txt");
                        for (int j = 0; j < 32; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth2[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth3[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output3.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth3[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth4[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output4.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth4[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                        output3.Close();
                        output4.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        string[] arr3 = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth3FlatFile.txt");
                        string[] arr4 = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth4FlatFile.txt");
                        for (int j = 0; j < 32; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth3[j] = Convert.ToDouble(arr3[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth4[j] = Convert.ToDouble(arr4[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if (layerNum == 2)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(1.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if (layerNum == 3)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(1.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if (layerNum == 4)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(1.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if (layerNum == 5)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(1.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
        }

        public void xavier_init_layer(int layerNum)
        {
            if (layerNum == 1)
            {
                double fan_in = 6400;
                double fan_out = 1400;
                double upper = 1 / Math.Sqrt(fan_in);
                double lower = -(1 / Math.Sqrt(fan_in));

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i] = new inputTensorKernel();
                    if (!File.Exists(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        StreamWriter output3 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth3FlatFile.txt");
                        StreamWriter output4 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth4FlatFile.txt");
                        for (int j = 0; j < 32; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth1[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth2[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth2[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth3[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output3.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth3[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth4[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output4.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth4[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                        output3.Close();
                        output4.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        string[] arr3 = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth3FlatFile.txt");
                        string[] arr4 = File.ReadAllLines(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth4FlatFile.txt");
                        for (int j = 0; j < 32; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth3[j] = Convert.ToDouble(arr3[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth4[j] = Convert.ToDouble(arr4[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if (layerNum == 2)
            {
                double fan_in = 1400;
                double fan_out = 1400;
                double upper = 1 / Math.Sqrt(fan_in);
                double lower = -(1 / Math.Sqrt(fan_in));

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth1[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth2[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if (layerNum == 3)
            {
                double fan_in = 1400;
                double fan_out = 1400;
                double upper = 1 / Math.Sqrt(fan_in);
                double lower = -(1 / Math.Sqrt(fan_in));

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth1[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth2[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if (layerNum == 4)
            {
                double fan_in = 1400;
                double fan_out = 1400;
                double upper = 1 / Math.Sqrt(fan_in);
                double lower = -(1 / Math.Sqrt(fan_in));

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth1[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth2[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
            else if (layerNum == 5)
            {
                double fan_in = 1400;
                double fan_out = 1400;
                double upper = 1 / Math.Sqrt(fan_in);
                double lower = -(1 / Math.Sqrt(fan_in));
                
                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i] = new hiddenTensorKernel();
                    if (!File.Exists(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                    {
                        StreamWriter output = File.AppendText(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        StreamWriter output2 = File.AppendText(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth1[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth1[j].ToString());
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth2[j] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                            predictorGui.numOfLearnableParams++;
                            output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth2[j].ToString());
                        }
                        output.Close();
                        output2.Close();
                    }
                    else
                    {
                        string[] arr = File.ReadAllLines(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                        string[] arr2 = File.ReadAllLines(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth1[j] = Convert.ToDouble(arr[j]);
                            predictorGui.numOfLearnableParams++;
                            predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth2[j] = Convert.ToDouble(arr2[j]);
                            predictorGui.numOfLearnableParams++;
                        }
                    }
                }
            }
        }

        public void lecun_normal_init_layer_GA(int networkNum, int layerNum)
        {
            if (layerNum == 1)
            {
                //number of input nodes from previous layer
                int n = 32 * 100 * 2;//6400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(1.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[networkNum].convStructs[0].convLayer1Kernel1[i] = new inputTensorKernel();
                    for (int j = 0; j < 32; j++)
                    {
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer1Kernel1[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer1Kernel1[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer1Kernel1[i].depth3[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer1Kernel1[i].depth4[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                    }
                }

                for(int k = 1; k < 32; k++)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        predictorGui.networkArray[networkNum].convStructs[k].convLayer1Kernel1[i] = new inputTensorKernel();
                        for (int j = 0; j < 32; j++)
                        {
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer1Kernel1[i].depth1[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer1Kernel1[i].depth1[j];
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer1Kernel1[i].depth2[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer1Kernel1[i].depth2[j];
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer1Kernel1[i].depth3[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer1Kernel1[i].depth3[j];
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer1Kernel1[i].depth4[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer1Kernel1[i].depth4[j];
                        }
                    }
                }
            }
            else if (layerNum == 2)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(1.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[networkNum].convStructs[0].convLayer2Kernel2[i] = new hiddenTensorKernel();

                    for (int j = 0; j < 14; j++)
                    {
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer2Kernel2[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer2Kernel2[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                    }
                }

                for (int k = 1; k < 32; k++)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        predictorGui.networkArray[networkNum].convStructs[k].convLayer2Kernel2[i] = new hiddenTensorKernel();

                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer2Kernel2[i].depth1[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer2Kernel2[i].depth1[j];
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer2Kernel2[i].depth2[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer2Kernel2[i].depth2[j];
                        }
                    }
                }
            }
            else if (layerNum == 3)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(1.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[networkNum].convStructs[0].convLayer3Kernel3[i] = new hiddenTensorKernel();
                    for (int j = 0; j < 14; j++)
                    {
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer3Kernel3[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer3Kernel3[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                    }
                }

                for (int k = 1; k < 32; k++)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        predictorGui.networkArray[networkNum].convStructs[k].convLayer3Kernel3[i] = new hiddenTensorKernel();

                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer3Kernel3[i].depth1[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer3Kernel3[i].depth1[j];
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer3Kernel3[i].depth2[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer3Kernel3[i].depth2[j];
                        }
                    }
                }
            }
            else if (layerNum == 4)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(1.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[networkNum].convStructs[0].convLayer4Kernel4[i] = new hiddenTensorKernel();
                    for (int j = 0; j < 14; j++)
                    {
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer4Kernel4[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer4Kernel4[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                    }
                }

                for (int k = 1; k < 32; k++)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        predictorGui.networkArray[networkNum].convStructs[k].convLayer4Kernel4[i] = new hiddenTensorKernel();

                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer4Kernel4[i].depth1[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer4Kernel4[i].depth1[j];
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer4Kernel4[i].depth2[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer4Kernel4[i].depth2[j];
                        }
                    }
                }
            }
            else if (layerNum == 5)
            {
                //number of input nodes from previous layer
                int n = 14 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(1.0 / n);

                for (int i = 0; i < 14; i++)
                {
                    predictorGui.networkArray[networkNum].convStructs[0].convLayer5Kernel5[i] = new hiddenTensorKernel();
                    for (int j = 0; j < 14; j++)
                    {
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer5Kernel5[i].depth1[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                        predictorGui.networkArray[networkNum].convStructs[0].convLayer5Kernel5[i].depth2[j] = SampleGaussian(predictorGui.rand, 0.0, std);
                    }
                }

                for (int k = 1; k < 32; k++)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        predictorGui.networkArray[networkNum].convStructs[k].convLayer5Kernel5[i] = new hiddenTensorKernel();

                        for (int j = 0; j < 14; j++)
                        {
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer5Kernel5[i].depth1[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer5Kernel5[i].depth1[j];
                            predictorGui.networkArray[networkNum].convStructs[k].convLayer5Kernel5[i].depth2[j] = predictorGui.networkArray[networkNum].convStructs[0].convLayer5Kernel5[i].depth2[j];
                        }
                    }
                }
            }
        }

        public void convLayerBiases_init_GA(int networkNum)
        {
            for (int k = 0; k < 32; k++)
            {
                for (int i = 0; i < 1400; i++)
                {
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer1Bias[i] = 0;
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer2Bias[i] = 0;
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer3Bias[i] = 0;
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer4Bias[i] = 0;
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer5Bias[i] = 0;
                }
            }
        }

        public void convLayerPReLUParams_init_GA(int networkNum)
        {
            for(int k = 0; k < 32; k++)
            {
                for (int i = 0; i < 1400; i++)
                {
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer1PReLUParam[i] = 0.02F;
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer2PReLUParam[i] = 0.02F;
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer3PReLUParam[i] = 0.02F;
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer4PReLUParam[i] = 0.02F;
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer5PReLUParam[i] = 0.02F;
                }
            }
        }

        public void convLayerNormGammaBetaInit_GA(int networkNum)
        {
            for (int k = 0; k < 32; k++)
            {
                for (int i = 0; i < 1400; i++)
                {
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer5OutputNormGamma[i] = 1;
                    predictorGui.networkArray[networkNum].convStructs[k].convLayer5OutputNormBeta[i] = 0;
                }
            }
        }

        public double SampleGaussian(Random random, double mean, double stddev)
        {
            // The method requires sampling from a uniform random of (0,1]
            // but Random.NextDouble() returns a sample of [0,1).
            double x1 = 1 - random.NextDouble();
            double x2 = 1 - random.NextDouble();

            double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
            return y1 * stddev + mean;
        }
    }
}
