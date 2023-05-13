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
        public static double[] firstLayerWeights = new double[96000];
        public static double[] mlpLayer1Bias = new double[64];
        public static double[] mlpLayer1PReLUParam = new double[64];

        public static double[] secondLayerWeights = new double[192];
        public static double[] mlpLayer2Bias = new double[3];
        public static double[] mlpLayer2PReLUParam = new double[3];

        public static double[] thirdLayerWeights = new double[9];

        public static double[] firstLayerOut = new double[64];
        public static double[] secondLayerOut = new double[3];
        public static double[] secondLayerOutRaw = new double[3];

        public static double[] dropout_mask = new double[64];

        public void firstLayer()
        {
            int M = 64;
            int K = 1500;
            int N = 1;

            Array.Copy(predictorGui.transformerBlock2Output, 0, backProp.firstLayerInCpy, 0, 1500);

            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

            //first MLP layer
            CudaDeviceVariable<double> d_in1 = backProp.firstLayerInCpy;
            CudaDeviceVariable<double> d_firstLayerWeights = firstLayerWeights;
            CudaDeviceVariable<double> d_firstLayerOut = new CudaDeviceVariable<double>(64);

            kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
            kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

            kernel.Run(d_firstLayerWeights.DevicePointer, d_in1.DevicePointer, d_firstLayerOut.DevicePointer, M, K, N);

            firstLayerOut = d_firstLayerOut;

            if (predictorGui.predictorGui1.preluSelectFinalMLP.Checked == true)
            {
                mlpLayer1_PReLU_and_add_bias1();
            }
            else if (predictorGui.predictorGui1.mishSelectFinalMLP.Checked == true)
            {
                mlpLayer1_Mish_and_add_bias();
            }

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\mlpLayer1.txt");
                for (int i = 0; i < 64; i++)
                {
                    output.WriteLine("First MLP Layer Out[" + i.ToString() + "] = " + firstLayerOut[i].ToString());
                }
                output.Close();
            }

            d_in1.Dispose();
            d_firstLayerWeights.Dispose();
            d_firstLayerOut.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();
        }

        public void secondLayer()
        {
            int M = 3;
            int K = 64;
            int N = 1;

            if (predictorGui.predictorGui1.activateTraining.Checked == true)
            {
                dropOut(1);
            }
            else
            {
                Array.Copy(firstLayerOut, 0, backProp.firstLayerOutCpy, 0, 64);
            }

            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

            //first MLP layer
            CudaDeviceVariable<double> d_in1 = backProp.firstLayerOutCpy;
            CudaDeviceVariable<double> d_secondLayerWeights = secondLayerWeights;
            CudaDeviceVariable<double> d_secondLayerOut = new CudaDeviceVariable<double>(3);

            kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
            kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

            kernel.Run(d_secondLayerWeights.DevicePointer, d_in1.DevicePointer, d_secondLayerOut.DevicePointer, M, K, N);

            secondLayerOut = d_secondLayerOut;

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\mlpLayer2.txt");
                for (int i = 0; i < 3; i++)
                {
                    output.WriteLine("Second MLP Layer Out[" + i.ToString() + "] = " + secondLayerOut[i].ToString());
                }
                output.Close();
            }

            Array.Copy(secondLayerOut, 0, secondLayerOutRaw, 0, 3);
            softmax();

            d_in1.Dispose();
            d_secondLayerWeights.Dispose();
            d_secondLayerOut.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();
        }

        public void softmax()
        {
            double exp_summation1 = 0;

            //find exponential summation for use with softmax function
            for (int i = 0; i < 3; i++)
            {
                exp_summation1 += Math.Exp(secondLayerOut[i]);
            }

            //apply softmax function
            for (int i = 0; i < 3; i++)
            {
                secondLayerOut[i] = Math.Exp(secondLayerOut[i]) / exp_summation1;
            }

            Array.Copy(secondLayerOut, 0, backProp.secondLayerOutCpy, 0, 3);
        }

        public void mlpLayer1_PReLU_and_add_bias1()
        {
            for (int i = 0; i < 64; i++)
            {
                if (firstLayerOut[i] + mlpLayer1Bias[i] > 0.0F)
                {
                    firstLayerOut[i] += mlpLayer1Bias[i];
                }
                else
                {
                    firstLayerOut[i] = /*(firstLayerOut[i] + mlpLayer1Bias[i]) * mlpLayer1PReLUParam[i]*/0;
                }
            }
        }

        public void mlpLayer1_Mish_and_add_bias()
        {
            for (int i = 0; i < 64; i++)
            {
                firstLayerOut[i] = (firstLayerOut[i] + mlpLayer1Bias[i]) * Math.Tanh(softplus(firstLayerOut[i] + mlpLayer1Bias[i]));
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
                    mlpLayer1Bias[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output.WriteLine(mlpLayer1Bias[i].ToString());
                }
                output.Close();
            }
            else
            {
                string[] arr;
                arr = File.ReadAllLines(@"X:\mlpFirstLayerBiasFlatFile.txt");
                for (int i = 0; i < 64; i++)
                {
                    mlpLayer1Bias[i] = Convert.ToDouble(arr[i]);
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
                    mlpLayer1PReLUParam[i] = 0.02F;
                    predictorGui.numOfLearnableParams++;
                    output.WriteLine(mlpLayer1PReLUParam[i].ToString());
                }
                output.Close();
            }
            else
            {
                string[] arr;
                arr = File.ReadAllLines(@"X:\mlpFirstLayerPReLUParamsFlatFile.txt");
                for (int i = 0; i < 64; i++)
                {
                    mlpLayer1PReLUParam[i] = Convert.ToDouble(arr[i]);
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
                        firstLayerWeights[i] = SampleGaussian(predictorGui.rand, 0.0, std);
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(firstLayerWeights[i]);
                    }
                    output.Close();
                }
                else
                {
                    string[] arr = File.ReadAllLines(@"X:\mlpFirstLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 96000; i++)
                    {
                        firstLayerWeights[i] = Convert.ToDouble(arr[i]);
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
                        secondLayerWeights[i] = SampleGaussian(predictorGui.rand, 0.0, std);
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(secondLayerWeights[i]);
                    }
                    output.Close();
                }
                else
                {
                    string[] arr = File.ReadAllLines(@"X:\mlpSecondLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 192; i++)
                    {
                        secondLayerWeights[i] = Convert.ToDouble(arr[i]);
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
                        firstLayerWeights[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(firstLayerWeights[i]);
                    }
                    output.Close();
                }
                else
                {
                    string[] arr;
                    arr = File.ReadAllLines(@"X:\mlpFirstLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 96000; i++)
                    {
                        firstLayerWeights[i] = Convert.ToDouble(arr[i]);
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
                        secondLayerWeights[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(secondLayerWeights[i].ToString());
                    }
                    output.Close();
                }
                else
                {
                    string[] arr;
                    arr = File.ReadAllLines(@"X:\mlpSecondLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 192; i++)
                    {
                        secondLayerWeights[i] = Convert.ToDouble(arr[i]);
                        predictorGui.numOfLearnableParams++;
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

        public void dropOut(int layerNum)
        {
            int hiddenPercentageDropOut = Convert.ToInt32(backProp.hiddenDropOutRate * 100F);

            //implement dropout with a certain specified percentage frequency
            if (layerNum == 1)
            {
                Array.Copy(firstLayerOut, 0, backProp.firstLayerOutCpy, 0, 64);
                for (int i = 0; i < 64; i++)
                {
                    int randomValue0To99 = predictorGui.rand.Next(100);
                    if (randomValue0To99 < hiddenPercentageDropOut)
                    {
                        dropout_mask[i] = 0; //changed 5/8/2022 potential dropout bug with earlier implementation
                    }
                    else
                    {
                        dropout_mask[i] = 1.0F / (1.0F - backProp.hiddenDropOutRate);
                    }
                }
                for (int i = 0; i < 64; i++)
                {
                    backProp.firstLayerOutCpy[i] *= dropout_mask[i];
                }
                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\firstLayerOutCpy.txt");
                    for(int i = 0; i < 64; i++)
                    {
                        output.WriteLine(backProp.firstLayerOutCpy[i].ToString());
                    }
                    output.Close();
                }
            }
        }
    }
}
