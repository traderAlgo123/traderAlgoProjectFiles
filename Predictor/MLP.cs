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

namespace Predictor
{
    public class MLP
    {
        matrixOps matOps = new matrixOps();
        public void firstLayer(int networkNum, int exampleNum)
        {
            double[] temp;
            int M = 64;
            int K = 1500;
            int N = 1;

            if (predictorGui.trainingBackProp == true)
            {
            //    Array.Copy(predictorGui.networkArray[networkNum].transStructs[exampleNum].transformerBlock2Output, 0, backProp.firstLayerInCpy, 0, 1500);
            }

            //CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            //CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

            //first MLP layer
            //CudaDeviceVariable<double> d_in1 = predictorGui.networkArray[networkNum].transStructs[exampleNum].transformerBlock2Output;
            //CudaDeviceVariable<double> d_firstLayerWeights = predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerWeights;
            //CudaDeviceVariable<double> d_firstLayerOut = new CudaDeviceVariable<double>(64);

            //kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
            //kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

            //kernel.Run(d_firstLayerWeights.DevicePointer, d_in1.DevicePointer, d_firstLayerOut.DevicePointer, M, K, N);

            //predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut = d_firstLayerOut;

            //StreamWriter output = File.AppendText(@"X:\debugOutput\mlpFirstLayerWeights" + networkNum + "-" + exampleNum + ".txt");
            //for (int m = 0; m < 96000; m++)
            //{
            //    output.WriteLine(predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerWeights[m].ToString());
            //}
            //output.Close();

            temp = matOps.matrixMulCpu(predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerWeights,
                                       predictorGui.networkArray[networkNum].transStructs[exampleNum].transformerBlockFinalOutput, M, K, N);
            Array.Copy(temp, 0, predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut, 0, M * N);

            if (predictorGui.predictorGui1.preluSelectFinalMLP.Checked == true)
            {
                mlpLayer1_PReLU_and_add_bias1(networkNum, exampleNum);
            }
            else if (predictorGui.predictorGui1.mishSelectFinalMLP.Checked == true)
            {
                mlpLayer1_Mish_and_add_bias(networkNum, exampleNum);
            }

            /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\mlpLayer1.txt");
                for (int i = 0; i < 64; i++)
                {
                    output.WriteLine("First MLP Layer Out[" + i.ToString() + "] = " + predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut[i].ToString());
                }
                output.Close();
            }*/

            //d_in1.Dispose();
            //d_firstLayerWeights.Dispose();
            //d_firstLayerOut.Dispose();
            //ctx.UnloadKernel(kernel);
            //ctx.Dispose();
        }

        public void secondLayer(int networkNum, int exampleNum)
        {
            double[] temp;
            int M = 3;
            int K = 64;
            int N = 1;

            if (predictorGui.predictorGui1.activateTraining.Checked == true)
            {
                //dropOut(networkNum, exampleNum, 1);
            }
            else
            {
                if (predictorGui.trainingBackProp == true)
                {
                    //Array.Copy(predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut, 0, backProp.firstLayerOutCpy, 0, 64);
                }
            }

            //CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            //CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

            //first MLP layer
            //CudaDeviceVariable<double> d_in1 = predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut;
            //CudaDeviceVariable<double> d_secondLayerWeights = predictorGui.networkArray[networkNum].mlpStructs[exampleNum].secondLayerWeights;
            //CudaDeviceVariable<double> d_secondLayerOut = new CudaDeviceVariable<double>(3);

            //kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
            //kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

            //kernel.Run(d_secondLayerWeights.DevicePointer, d_in1.DevicePointer, d_secondLayerOut.DevicePointer, M, K, N);

            //predictorGui.networkArray[networkNum].mlpStructs[exampleNum].secondLayerOut = d_secondLayerOut;
            temp = matOps.matrixMulCpu(predictorGui.networkArray[networkNum].mlpStructs[exampleNum].secondLayerWeights,
                                       predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut, M, K, N);
            Array.Copy(temp, 0, predictorGui.networkArray[networkNum].mlpStructs[exampleNum].secondLayerOut, 0, M * N);

            /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\mlpLayer2.txt");
                for (int i = 0; i < 3; i++)
                {
                    output.WriteLine("Second MLP Layer Out[" + i.ToString() + "] = " + predictorGui.mlpStructs[0].secondLayerOut[i].ToString());
                }
                output.Close();
            }*/

            Array.Copy(predictorGui.networkArray[networkNum].mlpStructs[exampleNum].secondLayerOut, 0, predictorGui.networkArray[networkNum].mlpStructs[exampleNum].secondLayerOutRaw, 0, 3);
            softmax(networkNum, exampleNum);

            //StreamWriter output = File.AppendText(@"X:\debugOutput\mlpOut" + networkNum  + "-" + exampleNum + ".txt");
            //for (int m = 0; m < 3; m++)
            //{
            //    output.WriteLine(predictorGui.networkArray[networkNum].mlpStructs[exampleNum].secondLayerOut[m].ToString());
            //}
            //output.Close();

            //d_in1.Dispose();
            //d_secondLayerWeights.Dispose();
            //d_secondLayerOut.Dispose();
            //ctx.UnloadKernel(kernel);
            //ctx.Dispose();
        }

        public void softmax(int networkNum, int exampleNum)
        {
            double exp_summation1 = 0;

            //find exponential summation for use with softmax function
            for (int i = 0; i < 3; i++)
            {
                exp_summation1 += Math.Exp(predictorGui.networkArray[networkNum].mlpStructs[exampleNum].secondLayerOut[i]);
            }

            //apply softmax function
            for (int i = 0; i < 3; i++)
            {
                predictorGui.networkArray[networkNum].mlpStructs[exampleNum].secondLayerOut[i] = Math.Exp(predictorGui.networkArray[networkNum].mlpStructs[exampleNum].secondLayerOut[i]) / exp_summation1;
            }

            if (predictorGui.trainingBackProp == true)
            {
                Array.Copy(predictorGui.networkArray[networkNum].mlpStructs[exampleNum].secondLayerOut, 0, backProp.secondLayerOutCpy, 0, 3);
            }
        }

        public void mlpLayer1_PReLU_and_add_bias1(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 64; i++)
            {
                if (predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut[i] + predictorGui.networkArray[networkNum].mlpStructs[exampleNum].mlpLayer1Bias[i] > 0.0F)
                {
                    predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut[i] += predictorGui.networkArray[networkNum].mlpStructs[exampleNum].mlpLayer1Bias[i];
                }
                else
                {
                    predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut[i] = /*(firstLayerOut[i] + mlpLayer1Bias[i]) * mlpLayer1PReLUParam[i]*/0;
                }
            }
        }

        public void mlpLayer1_Mish_and_add_bias(int networkNum, int exampleNum)
        {
            for (int i = 0; i < 64; i++)
            {
                predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut[i] = (predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut[i] + predictorGui.networkArray[networkNum].mlpStructs[exampleNum].mlpLayer1Bias[i]) * Math.Tanh(softplus(predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut[i] + predictorGui.networkArray[networkNum].mlpStructs[exampleNum].mlpLayer1Bias[i]));
            }
        }
        public double softplus(double x)
        {
            double temp;
            temp = Math.Log(1 + Math.Exp(x));
            return temp;
        }

        public void mlpLayerBiases_init1()
        {
            if (!File.Exists(@"X:\mlpFirstLayerBiasFlatFile.txt"))
            {
                StreamWriter output = File.AppendText(@"X:\mlpFirstLayerBiasFlatFile.txt");
                for (int i = 0; i < 64; i++)
                {
                    predictorGui.networkArray[0].mlpStructs[0].mlpLayer1Bias[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output.WriteLine(predictorGui.networkArray[0].mlpStructs[0].mlpLayer1Bias[i].ToString());
                }
                output.Close();
            }
            else
            {
                string[] arr;
                arr = File.ReadAllLines(@"X:\mlpFirstLayerBiasFlatFile.txt");
                for (int i = 0; i < 64; i++)
                {
                    predictorGui.networkArray[0].mlpStructs[0].mlpLayer1Bias[i] = Convert.ToDouble(arr[i]);
                    predictorGui.numOfLearnableParams++;
                }
            }
        }

        public void mlpLayerPReLUParams_init1()
        {
            if (!File.Exists(@"X:\mlpFirstLayerPReLUParamsFlatFile.txt"))
            {
                StreamWriter output = File.AppendText(@"X:\mlpFirstLayerPReLUParamsFlatFile.txt");
                for (int i = 0; i < 64; i++)
                {
                    predictorGui.networkArray[0].mlpStructs[0].mlpLayer1PReLUParam[i] = 0.02F;
                    predictorGui.numOfLearnableParams++;
                    output.WriteLine(predictorGui.networkArray[0].mlpStructs[0].mlpLayer1PReLUParam[i].ToString());
                }
                output.Close();
            }
            else
            {
                string[] arr;
                arr = File.ReadAllLines(@"X:\mlpFirstLayerPReLUParamsFlatFile.txt");
                for (int i = 0; i < 64; i++)
                {
                    predictorGui.networkArray[0].mlpStructs[0].mlpLayer1PReLUParam[i] = Convert.ToDouble(arr[i]);
                    predictorGui.numOfLearnableParams++;
                }
            }
        }

        public void kaiming_he_init_weights(int layerNum)
        {
            if (layerNum == 1)
            {
                //number of input nodes from previous layer
                int n = 15 * 100;//1400

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(2.0 / n);
                if (!File.Exists(@"X:\mlpFirstLayerWeightsFlatFile.txt"))
                {
                    StreamWriter output = File.AppendText(@"X:\mlpFirstLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 96000; i++)
                    {
                        predictorGui.networkArray[0].mlpStructs[0].firstLayerWeights[i] = SampleGaussian(predictorGui.rand, 0.0, std);
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(predictorGui.networkArray[0].mlpStructs[0].firstLayerWeights[i]);
                    }
                    output.Close();
                }
                else
                {
                    string[] arr = File.ReadAllLines(@"X:\mlpFirstLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 96000; i++)
                    {
                        predictorGui.networkArray[0].mlpStructs[0].firstLayerWeights[i] = Convert.ToDouble(arr[i]);
                        predictorGui.numOfLearnableParams++;
                    }
                }
            }
            else if(layerNum == 2)
            {
                //number of input nodes from previous layer
                int n = 64;//64

                //calculate standard deviation (range) for the weights
                double std = Math.Sqrt(2.0 / n);
                if (!File.Exists(@"X:\mlpSecondLayerWeightsFlatFile.txt"))
                {
                    StreamWriter output = File.AppendText(@"X:\mlpSecondLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 192; i++)
                    {
                        predictorGui.networkArray[0].mlpStructs[0].secondLayerWeights[i] = SampleGaussian(predictorGui.rand, 0.0, std);
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(predictorGui.networkArray[0].mlpStructs[0].secondLayerWeights[i]);
                    }
                    output.Close();
                }
                else
                {
                    string[] arr = File.ReadAllLines(@"X:\mlpSecondLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 192; i++)
                    {
                        predictorGui.networkArray[0].mlpStructs[0].secondLayerWeights[i] = Convert.ToDouble(arr[i]);
                        predictorGui.numOfLearnableParams++;
                    }
                }
            }
        }
        public void xavier_init_weights(int layerNum)
        {
            if (layerNum == 1)
            {
                if (!File.Exists(@"X:\mlpFirstLayerWeightsFlatFile.txt"))
                {
                    StreamWriter output = File.AppendText(@"X:\mlpFirstLayerWeightsFlatFile.txt");
                    double fan_in = 1500;
                    double fan_out = 64;
                    double upper = 1 / Math.Sqrt(fan_in);
                    double lower = -(1 / Math.Sqrt(fan_in));

                    for (int i = 0; i < 96000; i++)
                    {
                        predictorGui.networkArray[0].mlpStructs[0].firstLayerWeights[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(predictorGui.networkArray[0].mlpStructs[0].firstLayerWeights[i]);
                    }
                    output.Close();
                }
                else
                {
                    string[] arr;
                    arr = File.ReadAllLines(@"X:\mlpFirstLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 96000; i++)
                    {
                        predictorGui.networkArray[0].mlpStructs[0].firstLayerWeights[i] = Convert.ToDouble(arr[i]);
                        predictorGui.numOfLearnableParams++;
                    }
                }
            }
            else if (layerNum == 2)
            {
                if (!File.Exists(@"X:\mlpSecondLayerWeightsFlatFile.txt"))
                {
                    StreamWriter output = File.AppendText(@"X:\mlpSecondLayerWeightsFlatFile.txt");

                    double fan_in = 64;
                    double fan_out = 3;
                    double upper = 1 / Math.Sqrt(fan_in);
                    double lower = -(1 / Math.Sqrt(fan_in));

                    for (int i = 0; i < 192; i++)
                    {
                        predictorGui.networkArray[0].mlpStructs[0].secondLayerWeights[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(predictorGui.networkArray[0].mlpStructs[0].secondLayerWeights[i].ToString());
                    }
                    output.Close();
                }
                else
                {
                    string[] arr;
                    arr = File.ReadAllLines(@"X:\mlpSecondLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 192; i++)
                    {
                        predictorGui.networkArray[0].mlpStructs[0].secondLayerWeights[i] = Convert.ToDouble(arr[i]);
                        predictorGui.numOfLearnableParams++;
                    }
                }
            }
        }

        public void mlpLayerBiases_init1_GA(int networkNum)
        {
            for(int k = 0; k < 32; k++)
            {
                for (int i = 0; i < 64; i++)
                {
                    predictorGui.networkArray[networkNum].mlpStructs[k].mlpLayer1Bias[i] = 0;
                }
            }
        }

        public void mlpLayerPReLUParams_init1_GA(int networkNum)
        {
            for (int k = 0; k < 32; k++)
            {
                for (int i = 0; i < 64; i++)
                {
                    predictorGui.networkArray[networkNum].mlpStructs[k].mlpLayer1PReLUParam[i] = 0.02F;
                }
            }
        }
        public void xavier_init_weights_GA(int networkNum, int layerNum)
        {
            if (layerNum == 1)
            {
                double fan_in = 1500;
                double fan_out = 64;
                double upper = 1 / Math.Sqrt(fan_in);
                double lower = -(1 / Math.Sqrt(fan_in));

                for (int i = 0; i < 96000; i++)
                {
                    predictorGui.networkArray[networkNum].mlpStructs[0].firstLayerWeights[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                }

                for (int k = 1; k < 32; k++)
                {
                    for (int i = 0; i < 96000; i++)
                    {
                        predictorGui.networkArray[networkNum].mlpStructs[k].firstLayerWeights[i] = predictorGui.networkArray[networkNum].mlpStructs[0].firstLayerWeights[i];
                    }
                }
            }
            else if (layerNum == 2)
            {
                double fan_in = 64;
                double fan_out = 3;
                double upper = 1 / Math.Sqrt(fan_in);
                double lower = -(1 / Math.Sqrt(fan_in));

                for (int i = 0; i < 192; i++)
                {
                    predictorGui.networkArray[networkNum].mlpStructs[0].secondLayerWeights[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                }

                for (int k = 1; k < 32; k++)
                {
                    for (int i = 0; i < 192; i++)
                    {
                        predictorGui.networkArray[networkNum].mlpStructs[k].secondLayerWeights[i] = predictorGui.networkArray[networkNum].mlpStructs[0].secondLayerWeights[i];
                    }
                }
            }
        }

        public double SampleGaussian(Random random, double mean, double stddev)
        {
            // The method requires sampling from a uniform random of (0,1]
            // but Random.NextDouble() returns a sample of [0,1).
            double x1 = 1 - random.NextDouble();
            double x2 = 1 - random.NextDouble();

            double y1 = System.Math.Sqrt(-2.0 * System.Math.Log(x1)) * System.Math.Cos(2.0 * System.Math.PI * x2);
            return y1 * stddev + mean;
        }

        public void dropOut(int networkNum, int exampleNum, int layerNum)
        {
            int hiddenPercentageDropOut = Convert.ToInt32(backProp.hiddenDropOutRate * 100F);

            //implement dropout with a certain specified percentage frequency
            if (layerNum == 1)
            {
                //Array.Copy(predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut, 0, backProp.firstLayerOutCpy, 0, 64);
                for (int i = 0; i < 64; i++)
                {
                    int randomValue0To99 = predictorGui.rand.Next(100);
                    if (randomValue0To99 < hiddenPercentageDropOut)
                    {
                        predictorGui.networkArray[networkNum].mlpStructs[exampleNum].dropout_mask[i] = 0; //changed 5/8/2022 potential dropout bug with earlier implementation
                    }
                    else
                    {
                        predictorGui.networkArray[networkNum].mlpStructs[exampleNum].dropout_mask[i] = 1.0F / (1.0F - backProp.hiddenDropOutRate);
                    }
                }
                for (int i = 0; i < 64; i++)
                {
                    //backProp.firstLayerOutCpy[i] *= predictorGui.networkArray[0].mlpStructs[0].dropout_mask[i];
                    predictorGui.networkArray[networkNum].mlpStructs[exampleNum].firstLayerOut[i] *= predictorGui.networkArray[0].mlpStructs[0].dropout_mask[i];
                }
                /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\firstLayerOutCpy.txt");
                    for(int i = 0; i < 64; i++)
                    {
                        output.WriteLine(backProp.firstLayerOutCpy[i].ToString());
                    }
                    output.Close();
                }*/
            }
        }
    }
}
