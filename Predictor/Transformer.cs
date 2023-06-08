/*
* Copyright 2021-2023 - Tim Prishtina, and Luke Koch
*
* All rights reserved. No part of this software may be re-produced, re-engineered, 
* re-compiled, modified, used to create derivatives, stored in a retrieval system, 
* or transmitted in any form or by any means, whether electronic, mechanical, 
* photocopying, recording, or otherwise, without the prior written permission of 
* Tim Prishtina, and Luke Koch.
*/

using ManagedCuda;
using System;
using System.IO;
using System.Windows.Forms;

namespace Predictor
{
    public class Transformer
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            predictorGui.predictorGui1 = new predictorGui();
            predictorGui.guiThread.Start();
        }
    }

    public class Transformer_Implementation
    {
        public static double[] mean1 = new double[15];
        public static double[] mean1_block2 = new double[15];
        public static double[] mean2 = new double[15];
        public static double[] mean2_block2 = new double[15];
        public static double[] variance1 = new double[15];
        public static double[] variance1_block2 = new double[15];
        public static double[] variance2 = new double[15];
        public static double[] variance2_block2 = new double[15];
        public static double epsilon;
        public static double[] positionalEncodingArray = new double[1500];
        public static double[] positionalEncodingArrayCpy = new double[1500];
        public static double[] inputFromConvModule = new double[1500];
        public static double[] inputFromConvModuleCpy = new double[1500];

        public static double[] queryLinearLayerWeights_head1 = new double[75];
        public static double[] keyLinearLayerWeights_head1 = new double[75];
        public static double[] valueLinearLayerWeights_head1 = new double[75];

        public static double[] queryLinearLayerWeights_head2 = new double[75];
        public static double[] keyLinearLayerWeights_head2 = new double[75];
        public static double[] valueLinearLayerWeights_head2 = new double[75];

        public static double[] queryLinearLayerWeights_head3 = new double[75];
        public static double[] keyLinearLayerWeights_head3 = new double[75];
        public static double[] valueLinearLayerWeights_head3 = new double[75];

        public static double[] query_head1 = new double[500];
        public static double[] key_head1 = new double[500];
        public static double[] value_head1 = new double[500];
        public static double[] filtered_value_head1 = new double[500];
        public static double[] query_head1Cpy = new double[500];
        public static double[] key_head1Cpy = new double[500];
        public static double[] value_head1Cpy = new double[500];
        public static double[] filtered_value_head1Cpy = new double[500];

        public static double[] query_head2 = new double[500];
        public static double[] key_head2 = new double[500];
        public static double[] value_head2 = new double[500];
        public static double[] filtered_value_head2 = new double[500];
        public static double[] query_head2Cpy = new double[500];
        public static double[] key_head2Cpy = new double[500];
        public static double[] value_head2Cpy = new double[500];
        public static double[] filtered_value_head2Cpy = new double[500];

        public static double[] query_head3 = new double[500];
        public static double[] key_head3 = new double[500];
        public static double[] value_head3 = new double[500];
        public static double[] filtered_value_head3 = new double[500];
        public static double[] query_head3Cpy = new double[500];
        public static double[] key_head3Cpy = new double[500];
        public static double[] value_head3Cpy = new double[500];
        public static double[] filtered_value_head3Cpy = new double[500];

        public static double[] attention_filter_head1 = new double[10000];
        public static double[] attention_filter_head2 = new double[10000];
        public static double[] attention_filter_head3 = new double[10000];
        public static double[] attention_filter_head1Cpy = new double[10000];
        public static double[] attention_filter_head2Cpy = new double[10000];
        public static double[] attention_filter_head3Cpy = new double[10000];

        public static double[] concatenatedFilteredValueMatrix = new double[1500];
        public static double[] concatenatedFilteredValueMatrixCpy = new double[1500];

        public static double[] finalLinearLayerWeights = new double[225];
        public static double[] finalAttentionBlockOutput = new double[1500];
        public static double[] finalAttentionBlockOutputCpy = new double[1500];

        public static double[] residualConnectionOutputNorm = new double[1500];
        public static double[] residualConnectionOutputNormCpy = new double[1500];

        public static double[] residualConnectionOutputNormIntermediate1 = new double[1500];
        public static double[] residualConnectionOutputNormIntermediate2 = new double[1500];

        public static double[] addAndNorm1Gamma = new double[1500];
        public static double[] addAndNorm1Beta = new double[1500];
        public static double[] addAndNorm2Gamma = new double[1500];
        public static double[] addAndNorm2Beta = new double[1500];

        public static double[] affineTransWeights1 = new double[900];
        public static double[] affineTransWeights2 = new double[900];
        public static double[] affineTransWeights3 = new double[900];
        public static double[] affineTransWeights4 = new double[900];

        public static double[] transPReLUParam = new double[6000];
        public static double[] transPReLUBias = new double[6000];

        public static double[] transMLPSecondLayerBias = new double[1500];

        public static double[] affineIntermediateRes = new double[6000];
        public static double[] affineIntermediateRes2 = new double[6000];
        public static double[] affineIntermediateRes3 = new double[6000];
        public static double[] affineIntermediateRes4 = new double[6000];

        public static double[] transformerBlockFinalOutput = new double[1500];
        public static double[] transformerBlockFinalOutput2 = new double[1500];
        public static double[] transformerBlockFinalOutputIntermediate1 = new double[1500];
        public static double[] transformerBlockFinalOutputIntermediate2 = new double[1500];

        matrixOps matOps = new matrixOps();

        //this function requires you to input the matrix dimensions you WANT the transposed matrix to have
        //i.e. if you are transposing a 100x5 matrix you input M = 5 and K = 100 as the parameters.
        public void transposeConvKeyMat(int M, int K, int head)
        {
            if (head == 1)
            {
                CudaContext ctx = new CudaContext(predictorGui.selectGpu);
                CudaKernel kernel = ctx.LoadKernel("transpose.ptx", "transpose");

                CudaDeviceVariable<double> d_in1 = key_head1;
                CudaDeviceVariable<double> d_in1_T = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                //kernel.Run(d_in1.DevicePointer, d_in1_T.DevicePointer, M, K);
                kernel.Run(d_in1.DevicePointer, d_in1_T.DevicePointer, M, K);

                key_head1 = d_in1_T;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\keyMatrix1_transposed.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Key Matrix 1 Transposed[" + i.ToString() + "] = " + key_head1[i].ToString());
                    }
                    output.Close();
                }

                d_in1.Dispose();
                d_in1_T.Dispose();
                ctx.UnloadKernel(kernel);
                ctx.Dispose();
            }
            else if (head == 2)
            {
                CudaContext ctx = new CudaContext(predictorGui.selectGpu);
                CudaKernel kernel = ctx.LoadKernel("transpose.ptx", "transpose");

                CudaDeviceVariable<double> d_in2 = key_head2;
                CudaDeviceVariable<double> d_in2_T = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_in2.DevicePointer, d_in2_T.DevicePointer, M, K);

                key_head2 = d_in2_T;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\keyMatrix2_transposed.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Key Matrix 2 Transposed[" + i.ToString() + "] = " + key_head2[i].ToString());
                    }
                    output.Close();
                }

                d_in2.Dispose();
                d_in2_T.Dispose();
                ctx.UnloadKernel(kernel);
                ctx.Dispose();
            }
            else if (head == 3)
            {
                CudaContext ctx = new CudaContext(predictorGui.selectGpu);
                CudaKernel kernel = ctx.LoadKernel("transpose.ptx", "transpose");

                CudaDeviceVariable<double> d_in3 = key_head3;
                CudaDeviceVariable<double> d_in3_T = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_in3.DevicePointer, d_in3_T.DevicePointer, M, K);

                key_head3 = d_in3_T;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\keyMatrix3_transposed.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Key Matrix 3 Transposed[" + i.ToString() + "] = " + key_head3[i].ToString());
                    }
                    output.Close();
                }

                d_in3.Dispose();
                d_in3_T.Dispose();
                ctx.UnloadKernel(kernel);
                ctx.Dispose();
            }
        }

        public void attentionHeads(int headNum)
        {
            if (headNum == 1)
            {
                int M = 100;
                int K = 15;
                int N = 5;
                double[] temp;
                temp = matOps.matrixMulCpu(inputFromConvModule, queryLinearLayerWeights_head1, M, K, N);
                Array.Copy(temp, 0, query_head1, 0, M * N);
                CudaContext ctx = new CudaContext(predictorGui.selectGpu);
                CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

                //query matrix linear layer head 1
                //CudaDeviceVariable<double> d_in1 = inputFromConvModule;
                //CudaDeviceVariable<double> d_queryWeights = queryLinearLayerWeights_head1;
                //CudaDeviceVariable<double> d_queryMat = new CudaDeviceVariable<double>(500);

                //kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                //kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                //kernel.Run(d_in1.DevicePointer, d_queryWeights.DevicePointer, d_queryMat.DevicePointer, M, K, N);

                //query_head1 = d_queryMat;

                //if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                //{
                //    StreamWriter output = File.AppendText(@"X:\queryMatrix1.txt");
                //    for (int i = 0; i < 500; i++)
                //    {
                //        output.WriteLine("Query Matrix[" + i.ToString() + "] = " + query_head1[i].ToString());
                //    }
                //    output.Close();

                //    StreamWriter linearLayerWeights = File.AppendText(@"X:\querylinearLayerWeights_head1.txt");
                //    for (int i = 0; i < 75; i++)
                //    {
                //        linearLayerWeights.WriteLine(queryLinearLayerWeights_head1[i].ToString());
                //    }
                //    linearLayerWeights.Close();

                //    if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                //    {
                //        verify_result(M, K, N, query_head1, "query", 1);
                //    }
                //}

                //d_in1.Dispose();
                //d_queryWeights.Dispose();
                //d_queryMat.Dispose();
                //query matrix linear layer head 1 END

                //key matrix linear layer head 1
                CudaDeviceVariable<double> d_in2 = inputFromConvModule;
                CudaDeviceVariable<double> d_keyWeights = keyLinearLayerWeights_head1;
                CudaDeviceVariable<double> d_keyMat = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_in2.DevicePointer, d_keyWeights.DevicePointer, d_keyMat.DevicePointer, M, K, N);

                key_head1 = d_keyMat;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\keyMatrix1.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Key Matrix[" + i.ToString() + "] = " + key_head1[i].ToString());
                    }
                    output.Close();

                    StreamWriter linearLayerWeights = File.AppendText(@"X:\keylinearLayerWeights_head1.txt");
                    for (int i = 0; i < 75; i++)
                    {
                        linearLayerWeights.WriteLine(keyLinearLayerWeights_head1[i].ToString());
                    }
                    linearLayerWeights.Close();

                    if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                    {
                        verify_result(M, K, N, key_head1, "key", 1);
                    }
                }

                d_in2.Dispose();
                d_keyWeights.Dispose();
                d_keyMat.Dispose();
                //key matrix linear layer head 1 END

                //value matrix linear layer head 1
                CudaDeviceVariable<double> d_in3 = inputFromConvModule;
                CudaDeviceVariable<double> d_valueWeights = valueLinearLayerWeights_head1;
                CudaDeviceVariable<double> d_valueMat = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_in3.DevicePointer, d_valueWeights.DevicePointer, d_valueMat.DevicePointer, M, K, N);

                value_head1 = d_valueMat;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\valueMatrix1.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Value Matrix[" + i.ToString() + "] = " + value_head1[i].ToString());
                    }
                    output.Close();

                    StreamWriter linearLayerWeights = File.AppendText(@"X:\valuelinearLayerWeights_head1.txt");
                    for (int i = 0; i < 75; i++)
                    {
                        linearLayerWeights.WriteLine(valueLinearLayerWeights_head1[i].ToString());
                    }
                    linearLayerWeights.Close();

                    if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                    {
                        verify_result(M, K, N, value_head1, "value", 1);
                    }
                }
                //value matrix linear layer head 1 END

                d_in3.Dispose();
                d_valueWeights.Dispose();
                d_valueMat.Dispose();

                //multiply query and key matrices together
                M = 100;
                K = 5;
                N = 100;

                transposeConvKeyMat(5, 100, 1);

                CudaDeviceVariable<double> d_query = query_head1;
                CudaDeviceVariable<double> d_key = key_head1;
                CudaDeviceVariable<double> d_preliminary_attention_filter_head1 = new CudaDeviceVariable<double>(10000);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_query.DevicePointer, d_key.DevicePointer, d_preliminary_attention_filter_head1.DevicePointer, M, K, N);

                attention_filter_head1 = d_preliminary_attention_filter_head1;
                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\attentionFilter1.txt");
                    for (int i = 0; i < 10000; i++)
                    {
                        output.WriteLine("Attention Matrix[" + i.ToString() + "] = " + attention_filter_head1[i].ToString());
                    }
                    output.Close();
                }

                if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                {
                    verify_result(M, K, N, attention_filter_head1, "attention", 1);
                }

                d_query.Dispose();
                d_key.Dispose();
                d_preliminary_attention_filter_head1.Dispose();
                ctx.UnloadKernel(kernel);
                ctx.Dispose();
            }
            else if (headNum == 2)
            {
                int M = 100;
                int K = 15;
                int N = 5;

                CudaContext ctx = new CudaContext(predictorGui.selectGpu);
                CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

                //query matrix linear layer head 2
                CudaDeviceVariable<double> d_in1 = inputFromConvModule;
                CudaDeviceVariable<double> d_queryWeights2 = queryLinearLayerWeights_head2;
                CudaDeviceVariable<double> d_queryMat2 = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_in1.DevicePointer, d_queryWeights2.DevicePointer, d_queryMat2.DevicePointer, M, K, N);

                query_head2 = d_queryMat2;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\queryMatrix2.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Query Matrix[" + i.ToString() + "] = " + query_head2[i].ToString());
                    }
                    output.Close();

                    StreamWriter linearLayerWeights = File.AppendText(@"X:\querylinearLayerWeights_head2.txt");
                    for (int i = 0; i < 75; i++)
                    {
                        linearLayerWeights.WriteLine(queryLinearLayerWeights_head2[i].ToString());
                    }
                    linearLayerWeights.Close();

                    if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                    {
                        verify_result(M, K, N, query_head2, "query", 2);
                    }
                }

                d_in1.Dispose();
                d_queryWeights2.Dispose();
                d_queryMat2.Dispose();
                //query matrix linear layer head 2 END

                //key matrix linear layer head 2
                CudaDeviceVariable<double> d_in2 = inputFromConvModule;
                CudaDeviceVariable<double> d_keyWeights2 = keyLinearLayerWeights_head2;
                CudaDeviceVariable<double> d_keyMat2 = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_in2.DevicePointer, d_keyWeights2.DevicePointer, d_keyMat2.DevicePointer, M, K, N);

                key_head2 = d_keyMat2;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\keyMatrix2.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Key Matrix[" + i.ToString() + "] = " + key_head2[i].ToString());
                    }
                    output.Close();

                    StreamWriter linearLayerWeights = File.AppendText(@"X:\keylinearLayerWeights_head2.txt");
                    for (int i = 0; i < 75; i++)
                    {
                        linearLayerWeights.WriteLine(keyLinearLayerWeights_head2[i].ToString());
                    }
                    linearLayerWeights.Close();

                    if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                    {
                        verify_result(M, K, N, key_head2, "key", 2);
                    }
                }

                d_in2.Dispose();
                d_keyWeights2.Dispose();
                d_keyMat2.Dispose();
                //key matrix linear layer head 2 END

                //value matrix linear layer head 2
                CudaDeviceVariable<double> d_in3 = inputFromConvModule;
                CudaDeviceVariable<double> d_valueWeights2 = valueLinearLayerWeights_head2;
                CudaDeviceVariable<double> d_valueMat2 = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_in3.DevicePointer, d_valueWeights2.DevicePointer, d_valueMat2.DevicePointer, M, K, N);

                value_head2 = d_valueMat2;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\valueMatrix2.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Value Matrix[" + i.ToString() + "] = " + value_head2[i].ToString());
                    }
                    output.Close();

                    StreamWriter linearLayerWeights = File.AppendText(@"X:\valuelinearLayerWeights_head2.txt");
                    for (int i = 0; i < 75; i++)
                    {
                        linearLayerWeights.WriteLine(valueLinearLayerWeights_head2[i].ToString());
                    }
                    linearLayerWeights.Close();

                    if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                    {
                        verify_result(M, K, N, value_head2, "value", 2);
                    }
                }
                //value matrix linear layer head 2 END

                d_in3.Dispose();
                d_valueWeights2.Dispose();
                d_valueMat2.Dispose();

                //multiply query and key matrices together
                M = 100;
                K = 5;
                N = 100;

                transposeConvKeyMat(5, 100, 2);

                CudaDeviceVariable<double> d_query2 = query_head2;
                CudaDeviceVariable<double> d_key2 = key_head2;
                CudaDeviceVariable<double> d_preliminary_attention_filter_head2 = new CudaDeviceVariable<double>(10000);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_query2.DevicePointer, d_key2.DevicePointer, d_preliminary_attention_filter_head2.DevicePointer, M, K, N);

                attention_filter_head2 = d_preliminary_attention_filter_head2;
                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\attentionFilter2.txt");
                    for (int i = 0; i < 10000; i++)
                    {
                        output.WriteLine("Attention Matrix[" + i.ToString() + "] = " + attention_filter_head2[i].ToString());
                    }
                    output.Close();
                }

                if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                {
                    verify_result(M, K, N, attention_filter_head2, "attention", 2);
                }

                d_query2.Dispose();
                d_key2.Dispose();
                d_preliminary_attention_filter_head2.Dispose();
                ctx.UnloadKernel(kernel);
                ctx.Dispose();
            }
            else if (headNum == 3)
            {
                int M = 100;
                int K = 15;
                int N = 5;

                CudaContext ctx = new CudaContext(predictorGui.selectGpu);
                CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

                //query matrix linear layer head 3
                CudaDeviceVariable<double> d_in1 = inputFromConvModule;
                CudaDeviceVariable<double> d_queryWeights3 = queryLinearLayerWeights_head3;
                CudaDeviceVariable<double> d_queryMat3 = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_in1.DevicePointer, d_queryWeights3.DevicePointer, d_queryMat3.DevicePointer, M, K, N);

                query_head3 = d_queryMat3;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\queryMatrix3.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Query Matrix[" + i.ToString() + "] = " + query_head3[i].ToString());
                    }
                    output.Close();

                    StreamWriter linearLayerWeights = File.AppendText(@"X:\querylinearLayerWeights_head3.txt");
                    for (int i = 0; i < 75; i++)
                    {
                        linearLayerWeights.WriteLine(queryLinearLayerWeights_head3[i].ToString());
                    }
                    linearLayerWeights.Close();

                    if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                    {
                        verify_result(M, K, N, query_head3, "query", 3);
                    }
                }

                d_in1.Dispose();
                d_queryWeights3.Dispose();
                d_queryMat3.Dispose();
                //query matrix linear layer head 3 END

                //key matrix linear layer head 3
                CudaDeviceVariable<double> d_in2 = inputFromConvModule;
                CudaDeviceVariable<double> d_keyWeights3 = keyLinearLayerWeights_head3;
                CudaDeviceVariable<double> d_keyMat3 = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_in2.DevicePointer, d_keyWeights3.DevicePointer, d_keyMat3.DevicePointer, M, K, N);

                key_head3 = d_keyMat3;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\keyMatrix3.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Key Matrix[" + i.ToString() + "] = " + key_head3[i].ToString());
                    }
                    output.Close();

                    StreamWriter linearLayerWeights = File.AppendText(@"X:\keylinearLayerWeights_head3.txt");
                    for (int i = 0; i < 75; i++)
                    {
                        linearLayerWeights.WriteLine(keyLinearLayerWeights_head3[i].ToString());
                    }
                    linearLayerWeights.Close();

                    if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                    {
                        verify_result(M, K, N, key_head3, "key", 3);
                    }
                }

                d_in2.Dispose();
                d_keyWeights3.Dispose();
                d_keyMat3.Dispose();
                //key matrix linear layer head 3 END

                //value matrix linear layer head 3
                CudaDeviceVariable<double> d_in3 = inputFromConvModule;
                CudaDeviceVariable<double> d_valueWeights3 = valueLinearLayerWeights_head3;
                CudaDeviceVariable<double> d_valueMat3 = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_in3.DevicePointer, d_valueWeights3.DevicePointer, d_valueMat3.DevicePointer, M, K, N);

                value_head3 = d_valueMat3;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\valueMatrix3.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Value Matrix[" + i.ToString() + "] = " + value_head3[i].ToString());
                    }
                    output.Close();

                    StreamWriter linearLayerWeights = File.AppendText(@"X:\valuelinearLayerWeights_head3.txt");
                    for (int i = 0; i < 75; i++)
                    {
                        linearLayerWeights.WriteLine(valueLinearLayerWeights_head3[i].ToString());
                    }
                    linearLayerWeights.Close();

                    if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                    {
                        verify_result(M, K, N, value_head3, "value", 3);
                    }
                }
                //value matrix linear layer head 3 END

                d_in3.Dispose();
                d_valueWeights3.Dispose();
                d_valueMat3.Dispose();

                //multiply query and key matrices together
                M = 100;
                K = 5;
                N = 100;

                transposeConvKeyMat(5, 100, 3);

                CudaDeviceVariable<double> d_query3 = query_head3;
                CudaDeviceVariable<double> d_key3 = key_head3;
                CudaDeviceVariable<double> d_preliminary_attention_filter_head3 = new CudaDeviceVariable<double>(10000);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_query3.DevicePointer, d_key3.DevicePointer, d_preliminary_attention_filter_head3.DevicePointer, M, K, N);

                attention_filter_head3 = d_preliminary_attention_filter_head3;
                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\attentionFilter3.txt");
                    for (int i = 0; i < 10000; i++)
                    {
                        output.WriteLine("Attention Matrix[" + i.ToString() + "] = " + attention_filter_head3[i].ToString());
                    }
                    output.Close();
                }

                if (predictorGui.predictorGui1.confirmCPU.Checked == true)
                {
                    verify_result(M, K, N, attention_filter_head3, "attention", 3);
                }

                d_query3.Dispose();
                d_key3.Dispose();
                d_preliminary_attention_filter_head3.Dispose();
                ctx.UnloadKernel(kernel);
                ctx.Dispose();
            }
        }

        public void verify_result(int M, int K, int N, double[] matrix, string matType, int head)
        {
            double[] temp_array = new double[500];
            double[] temp_array2 = new double[10000];

            if (matType.Equals("query") && head == 1)
            {
                StreamWriter output = File.AppendText(@"X:\multiplied_coefficients_inputConv_to_queryWeights.txt");
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += inputFromConvModule[row * K + i] * queryLinearLayerWeights_head1[i * N + col];
                            output.WriteLine(inputFromConvModule[row * K + i] + " * " + queryLinearLayerWeights_head1[i * N + col]);
                        }
                        temp_array[row * N + col] = tmp;
                    }
                }
                output.Close();
            }
            else if (matType.Equals("key") && head == 1)
            {
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += inputFromConvModule[row * K + i] * keyLinearLayerWeights_head1[i * N + col];
                        }
                        temp_array[row * N + col] = tmp;
                    }
                }
            }
            else if (matType.Equals("value") && head == 1)
            {
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += inputFromConvModule[row * K + i] * valueLinearLayerWeights_head1[i * N + col];
                        }
                        temp_array[row * N + col] = tmp;
                    }
                }
            }
            else if (matType.Equals("query") && head == 2)
            {
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += inputFromConvModule[row * K + i] * queryLinearLayerWeights_head2[i * N + col];
                        }
                        temp_array[row * N + col] = tmp;
                    }
                }
            }
            else if (matType.Equals("key") && head == 2)
            {
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += inputFromConvModule[row * K + i] * keyLinearLayerWeights_head2[i * N + col];
                        }
                        temp_array[row * N + col] = tmp;
                    }
                }
            }
            else if (matType.Equals("value") && head == 2)
            {
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += inputFromConvModule[row * K + i] * valueLinearLayerWeights_head2[i * N + col];
                        }
                        temp_array[row * N + col] = tmp;
                    }
                }
            }
            else if (matType.Equals("query") && head == 3)
            {
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += inputFromConvModule[row * K + i] * queryLinearLayerWeights_head3[i * N + col];
                        }
                        temp_array[row * N + col] = tmp;
                    }
                }
            }
            else if (matType.Equals("key") && head == 3)
            {
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += inputFromConvModule[row * K + i] * keyLinearLayerWeights_head3[i * N + col];
                        }
                        temp_array[row * N + col] = tmp;
                    }
                }
            }
            else if (matType.Equals("value") && head == 3)
            {
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += inputFromConvModule[row * K + i] * valueLinearLayerWeights_head3[i * N + col];
                        }
                        temp_array[row * N + col] = tmp;
                    }
                }
            }
            else if (matType.Equals("attention") && head == 1)
            {
                StreamWriter output = File.AppendText(@"X:\multiplied_coefficients_query_to_keyTrans.txt");
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += query_head1[row * K + i] * key_head1[i * N + col];
                            output.WriteLine(query_head1[row * K + i] + " * " + key_head1[i * N + col]);
                        }
                        temp_array2[row * N + col] = tmp;
                    }
                }
                output.Close();
            }
            else if (matType.Equals("attention") && head == 2)
            {
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += query_head2[row * K + i] * key_head2[i * N + col];
                        }
                        temp_array2[row * N + col] = tmp;
                    }
                }
            }
            else if (matType.Equals("attention") && head == 3)
            {
                for (int row = 0; row < M; row++)
                {
                    for (int col = 0; col < N; col++)
                    {
                        double tmp = 0;
                        for (int i = 0; i < K; i++)
                        {
                            tmp += query_head3[row * K + i] * key_head3[i * N + col];
                        }
                        temp_array2[row * N + col] = tmp;
                    }
                }
            }

            if (!matType.Equals("attention"))
            {
                StreamWriter output = File.AppendText(@"X:\" + matType + "Matrix_CPU" + head.ToString() + ".txt");
                for (int i = 0; i < 500; i++)
                {
                    if (Math.Round(temp_array[i], 5) != Math.Round(matrix[i], 5))
                    {
                        predictorGui.predictorGui1.transOut.Text = "Not equal at index " + i.ToString();
                    }
                    output.WriteLine(matType + " Matrix[" + i.ToString() + "] = " + temp_array[i].ToString());
                }
                output.Close();
            }
            else
            {
                StreamWriter output = File.AppendText(@"X:\" + matType + "Matrix_CPU" + head.ToString() + ".txt");
                for (int i = 0; i < 10000; i++)
                {
                    if (Math.Round(temp_array2[i], 5) != Math.Round(matrix[i], 5))
                    {
                        predictorGui.predictorGui1.transOut.Text = "Not equal at index " + i.ToString();
                    }
                    output.WriteLine(matType + " Matrix[" + i.ToString() + "] = " + temp_array2[i].ToString());
                }
                output.Close();
            }
        }

        public void positionalEncoding(int pass)
        {
            if (pass == 1)
            {

                //formula to be used is sin(pos/10000^((2 * i) / d)
                double pos = 0;
                double i = 0;
                double d = 15; //full length of a single feature vector

                Array.Copy(predictorGui.transformerInput, 0, inputFromConvModule, 0, 1500);

                /* unknown if we even need sinusoidal positional encoding at all, there is a distinct possibility this is
                 * drowning out the output of the convolutional module, which would cause learning to stall. We already have
                 * a linear version of positional encoding in the temporal encoding concatenation, this may be enough so we will
                 * test using this 
                for (int j = 0; j < 1500; j++)
                {
                    if (i % 2 == 0)
                    {
                        positionalEncodingArray[j] = Math.Sin(pos / Math.Pow(10000, 2 * i / d)) / d;
                        inputFromConvModule[j] += positionalEncodingArray[j];
                    }
                    else
                    {
                        positionalEncodingArray[j] = Math.Cos(pos / Math.Pow(10000, 2 * i / d)) / d;
                        inputFromConvModule[j] += positionalEncodingArray[j];
                    }
                    i++;
                    if ((j % 14 == 0) && j != 0)
                    {
                        pos++;
                        i = 0;
                    }
                }

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\positionalEncodingOut.txt");
                    StreamWriter output2 = File.AppendText(@"X:\posEncodedInput.txt");
                    for (int j = 0; j < 1500; j++)
                    {
                        output.WriteLine("Positional Encoding[" + j.ToString() + "] = " + positionalEncodingArray[j].ToString());
                        output2.WriteLine("Positionally Encoded Input[" + j.ToString() + "] = " + inputFromConvModule[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                }
                */
            }
            else if (pass == 2)
            {
                /* unknown if we need to add positional encoding to the output of the transformer block 1 before processing in block 2
                 * it is possible but for now we will leave it out until testing reveals that we need it or do not need it.
                //formula to be used is sin(pos/10000^((2 * i) / d)
                double pos = 0;
                double i = 0;
                double d = 15; //full length of a single feature vector

                for (int j = 0; j < 1500; j++)
                {
                    if (i % 2 == 0)
                    {
                        positionalEncodingArray[j] = Math.Sin(pos / (Math.Pow(10000, ((2 * i) / d))));
                        transformerBlockFinalOutput[j] += positionalEncodingArray[j];
                    }
                    else
                    {
                        positionalEncodingArray[j] = Math.Cos(pos / (Math.Pow(10000, ((2 * i) / d))));
                        transformerBlockFinalOutput[j] += positionalEncodingArray[j];
                    }
                    i++;
                    if ((j % 14 == 0) && j != 0)
                    {
                        pos++;
                        i = 0;
                    }
                }*/
                //trick attention head functions into using the output of the first transformer block
                Array.Copy(transformerBlockFinalOutput, 0, inputFromConvModule, 0, 1500);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\positionalEncodingOut.txt");
                    StreamWriter output2 = File.AppendText(@"X:\posEncodedInput.txt");
                    for (int j = 0; j < 1500; j++)
                    {
                        output.WriteLine("Positional Encoding[" + j.ToString() + "] = " + positionalEncodingArray[j].ToString());
                        output2.WriteLine("Positionally Encoded Input[" + j.ToString() + "] = " + transformerBlockFinalOutput[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                }
            }
        }

        public void scaleAndSoftmax_with_masking()
        {
            double scaleVal = Math.Sqrt(5); //value used to scale attention filters
            double[] exp_summation1 = new double[100];
            double[] exp_summation2 = new double[100];
            double[] exp_summation3 = new double[100];
            double[] mask = new double[99];
            int sumIdx = 0;

            for (int i = 0; i < 99; i++)
            {
                mask[i] = Double.NegativeInfinity;
            }

            //apply scaling
            for (int i = 0; i < 10000; i++)
            {
                attention_filter_head1[i] = attention_filter_head1[i] / scaleVal;
                attention_filter_head2[i] = attention_filter_head2[i] / scaleVal;
                attention_filter_head3[i] = attention_filter_head3[i] / scaleVal;
            }

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\scaledSoftmaxOutput1.txt");
                for (int i = 0; i < 10000; i++)
                {
                    output.WriteLine("Attention Matrix Unmasked[" + i.ToString() + "] = " + attention_filter_head1[i].ToString());
                }
                output.Close();

                StreamWriter output2 = File.AppendText(@"X:\scaledSoftmaxOutput2.txt");
                for (int i = 0; i < 10000; i++)
                {
                    output2.WriteLine("Attention Matrix Unmasked[" + i.ToString() + "] = " + attention_filter_head2[i].ToString());
                }
                output2.Close();

                StreamWriter output3 = File.AppendText(@"X:\scaledSoftmaxOutput3.txt");
                for (int i = 0; i < 10000; i++)
                {
                    output3.WriteLine("Attention Matrix Unmasked[" + i.ToString() + "] = " + attention_filter_head3[i].ToString());
                }
                output3.Close();
            }

            int destIdx = 1;
            int len = 99;
            //apply masking
            for (int i = 0; i < 100; i++)
            {
                Array.Copy(mask, 0, attention_filter_head1, destIdx, len);
                Array.Copy(mask, 0, attention_filter_head2, destIdx, len);
                Array.Copy(mask, 0, attention_filter_head3, destIdx, len);
                destIdx += 101;
                len--;
            }

            //find exponential summation for each row for use with softmax function
            for (int i = 0; i < 10000; i++)
            {
                if (i % 100 == 0 && i != 0)
                {
                    sumIdx++;
                }
                exp_summation1[sumIdx] += Math.Exp(attention_filter_head1[i]);
                exp_summation2[sumIdx] += Math.Exp(attention_filter_head2[i]);
                exp_summation3[sumIdx] += Math.Exp(attention_filter_head3[i]);
            }
            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\exp_summation1.txt");
                for (int i = 0; i < 100; i++)
                {
                    output.WriteLine(exp_summation1[i].ToString());
                }
                output.Close();
            }
            sumIdx = 0;
            for (int i = 0; i < 10000; i++)
            {
                if (i % 100 == 0 && i != 0)
                {
                    sumIdx++;
                }
                attention_filter_head1[i] = Math.Exp(attention_filter_head1[i]) / exp_summation1[sumIdx];
                attention_filter_head2[i] = Math.Exp(attention_filter_head2[i]) / exp_summation2[sumIdx];
                attention_filter_head3[i] = Math.Exp(attention_filter_head3[i]) / exp_summation3[sumIdx];
            }
            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\scaledSoftmaxMaskedOutput1.txt");
                for (int i = 0; i < 10000; i++)
                {
                    output.WriteLine("Attention Matrix Masked[" + i.ToString() + "] = " + attention_filter_head1[i].ToString());
                }
                output.Close();

                StreamWriter output2 = File.AppendText(@"X:\scaledSoftmaxMaskedOutput2.txt");
                for (int i = 0; i < 10000; i++)
                {
                    output2.WriteLine("Attention Matrix Masked[" + i.ToString() + "] = " + attention_filter_head2[i].ToString());
                }
                output2.Close();

                StreamWriter output3 = File.AppendText(@"X:\scaledSoftmaxMaskedOutput3.txt");
                for (int i = 0; i < 10000; i++)
                {
                    output3.WriteLine("Attention Matrix Masked[" + i.ToString() + "] = " + attention_filter_head3[i].ToString());
                }
                output3.Close();
            }
        }

        public void matMulFilteredValueMat(int headNum)
        {
            if (headNum == 1)
            {
                CudaContext ctx = new CudaContext(predictorGui.selectGpu);
                CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

                int M = 100;
                int K = 100;
                int N = 5;
                CudaDeviceVariable<double> d_maskedSelfAttentionFilter1 = attention_filter_head1;
                CudaDeviceVariable<double> d_value1 = value_head1;
                CudaDeviceVariable<double> d_filteredValMat = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_maskedSelfAttentionFilter1.DevicePointer, d_value1.DevicePointer, d_filteredValMat.DevicePointer, M, K, N);

                filtered_value_head1 = d_filteredValMat;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\filteredValMat1.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Filtered Value Matrix[" + i.ToString() + "] = " + filtered_value_head1[i].ToString());
                    }
                    output.Close();
                }

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    verify_filteredVal_result(M, K, N, 1);
                }

                d_maskedSelfAttentionFilter1.Dispose();
                d_value1.Dispose();
                d_filteredValMat.Dispose();
                ctx.UnloadKernel(kernel);
                ctx.Dispose();
            }
            if (headNum == 2)
            {
                CudaContext ctx = new CudaContext(predictorGui.selectGpu);
                CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

                int M = 100;
                int K = 100;
                int N = 5;
                CudaDeviceVariable<double> d_maskedSelfAttentionFilter2 = attention_filter_head2;
                CudaDeviceVariable<double> d_value2 = value_head2;
                CudaDeviceVariable<double> d_filteredValMat = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_maskedSelfAttentionFilter2.DevicePointer, d_value2.DevicePointer, d_filteredValMat.DevicePointer, M, K, N);

                filtered_value_head2 = d_filteredValMat;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\filteredValMat2.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Filtered Value Matrix[" + i.ToString() + "] = " + filtered_value_head2[i].ToString());
                    }
                    output.Close();
                }

                d_maskedSelfAttentionFilter2.Dispose();
                d_value2.Dispose();
                d_filteredValMat.Dispose();
                ctx.UnloadKernel(kernel);
                ctx.Dispose();
            }
            if (headNum == 3)
            {
                CudaContext ctx = new CudaContext(predictorGui.selectGpu);
                CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

                int M = 100;
                int K = 100;
                int N = 5;
                CudaDeviceVariable<double> d_maskedSelfAttentionFilter3 = attention_filter_head3;
                CudaDeviceVariable<double> d_value3 = value_head3;
                CudaDeviceVariable<double> d_filteredValMat = new CudaDeviceVariable<double>(500);

                kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
                kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

                kernel.Run(d_maskedSelfAttentionFilter3.DevicePointer, d_value3.DevicePointer, d_filteredValMat.DevicePointer, M, K, N);

                filtered_value_head3 = d_filteredValMat;

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\filteredValMat3.txt");
                    for (int i = 0; i < 500; i++)
                    {
                        output.WriteLine("Filtered Value Matrix[" + i.ToString() + "] = " + filtered_value_head3[i].ToString());
                    }
                    output.Close();
                }

                d_maskedSelfAttentionFilter3.Dispose();
                d_value3.Dispose();
                d_filteredValMat.Dispose();
                ctx.UnloadKernel(kernel);
                ctx.Dispose();
            }
        }

        public void verify_filteredVal_result(int M, int K, int N, int head)
        {
            double[] temp_array = new double[500];

            for (int row = 0; row < M; row++)
            {
                for (int col = 0; col < N; col++)
                {
                    double tmp = 0;
                    for (int i = 0; i < K; i++)
                    {
                        tmp += attention_filter_head1[row * K + i] * value_head1[i * N + col];
                    }
                    temp_array[row * N + col] = tmp;
                }
            }

            StreamWriter output = File.AppendText(@"X:\filteredValueMatrix_CPU" + head.ToString() + ".txt");
            for (int i = 0; i < 500; i++)
            {
                if (Math.Round(temp_array[i], 5) != Math.Round(filtered_value_head1[i], 5))
                {
                    predictorGui.predictorGui1.transOut.Text = "Not equal at index " + i.ToString();
                }
                output.WriteLine("Filtered Value Matrix_CPU[" + i.ToString() + "] = " + temp_array[i].ToString());
            }
            output.Close();
        }

        public void concatFilteredValMats()
        {
            int concatIdx = 0;
            int j = 0;
            for (int i = 0; i < 1500; i++)
            {
                if (concatIdx < 5)
                {
                    concatenatedFilteredValueMatrix[i] = filtered_value_head1[j];
                    if (j % 4 == 0 && j != 0)
                    {
                        j = j / 4 - 1;
                    }
                    else
                    {
                        j++;
                    }
                }
                else if (concatIdx >= 5 && concatIdx < 10)
                {
                    concatenatedFilteredValueMatrix[i] = filtered_value_head2[j];
                    if (j % 4 == 0 && j != 0)
                    {
                        j = j / 4 - 1;
                    }
                    else
                    {
                        j++;
                    }
                }
                else if (concatIdx >= 10 && concatIdx < 15)
                {
                    concatenatedFilteredValueMatrix[i] = filtered_value_head3[j];
                    if (concatIdx == 14)
                    {
                        concatIdx = 0;
                    }

                    if (j % 4 == 0 && j != 0)
                    {
                        j = j / 4 - 1;
                    }
                    else
                    {
                        j++;
                    }
                }
                concatIdx++;
            }

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\concatenatedfilteredValueMatrix.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output.WriteLine("Concatenated Filtered Value Matrix[" + i.ToString() + "] = " + concatenatedFilteredValueMatrix[i].ToString());
                }
                output.Close();
            }
        }

        public void finalAttentionBlockLinearLayer()
        {
            CudaContext ctx = new CudaContext(predictorGui.selectGpu);
            CudaKernel kernel = ctx.LoadKernel("matMul.ptx", "matrixMul");

            int M = 100;
            int K = 15;
            int N = 15;
            CudaDeviceVariable<double> d_concatFiltValMat = concatenatedFilteredValueMatrix;
            CudaDeviceVariable<double> d_finalLinearLayerWeights = finalLinearLayerWeights;
            CudaDeviceVariable<double> d_finalAttentionBlockOut = new CudaDeviceVariable<double>(1500);

            kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3((K + 2048 - 1) / 32, (M + 2048 - 1) / 32);
            kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(32, 32);

            kernel.Run(d_concatFiltValMat.DevicePointer, d_finalLinearLayerWeights.DevicePointer, d_finalAttentionBlockOut.DevicePointer, M, K, N);

            finalAttentionBlockOutput = d_finalAttentionBlockOut;

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\finalAttentionBlockOutput.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output.WriteLine("Final Output Matrix[" + i.ToString() + "] = " + finalAttentionBlockOutput[i].ToString());
                }
                output.Close();
            }

            if (predictorGui.predictorGui1.confirmCPU.Checked == true)
            {
                verify_finalOutput_result(M, K, N);
            }

            d_concatFiltValMat.Dispose();
            d_finalLinearLayerWeights.Dispose();
            d_finalAttentionBlockOut.Dispose();
            ctx.UnloadKernel(kernel);
            ctx.Dispose();
        }

        public void verify_finalOutput_result(int M, int K, int N)
        {
            double[] temp_array = new double[1500];

            for (int row = 0; row < M; row++)
            {
                for (int col = 0; col < N; col++)
                {
                    double tmp = 0;
                    for (int i = 0; i < K; i++)
                    {
                        tmp += concatenatedFilteredValueMatrix[row * K + i] * finalLinearLayerWeights[i * N + col];
                    }
                    temp_array[row * N + col] = tmp;
                }
            }

            StreamWriter output = File.AppendText(@"X:\finalAttentionBlockOutput_CPU.txt");
            for (int i = 0; i < 1500; i++)
            {
                if (Math.Round(temp_array[i], 5) != Math.Round(finalAttentionBlockOutput[i], 5))
                {
                    predictorGui.predictorGui1.transOut.Text = "Not equal at index " + i.ToString();
                }
                output.WriteLine("Final Output Matrix_CPU[" + i.ToString() + "] = " + temp_array[i].ToString());
            }
            output.Close();
        }

        public void addAndNormLayer(int pass, int blockNum)
        {
            //needs to be redesigned to execute normalization across each feature row, instead of how it is currently applying
            //normalization across the entire layer mean and variance (NOTE: Done on 5/25/2022 for transformer module forward pass)
            if (pass == 1)
            {
                //implement out residual connection from input to attention heads to output of attention block
                for (int i = 0; i < 1500; i++)
                {
                    residualConnectionOutputNorm[i] = inputFromConvModule[i] + finalAttentionBlockOutput[i];
                }
                /*
                double feature_summation = 0;
                double var_summation = 0;
                int meanVarianceIdx = 0;
                epsilon = backProp.layerNormEpsilon; // 1 * (10 ^ -5) APPARENTLY C# doesn't know how to do scientific notation
                                          //one times ten to the -5 dumbass, apparently it should be 1e-05

                if (blockNum == 2)
                {
                    Array.Copy(residualConnectionOutputNorm, 0, residualConnectionOutputNormIntermediate2, 0, 1500);
                    for (int i = 0; i < 1500; i++)
                    {
                        mean1_block2[0] += residualConnectionOutputNorm[i];
                    }
                    mean1_block2[0] /= 1500;

                    meanVarianceIdx = 0;
                    for (int i = 0; i < 1500; i++)
                    {
                        variance1_block2[0] += Math.Pow((residualConnectionOutputNorm[i] - mean1_block2[0]), 2);
                    }

                    meanVarianceIdx = 0;
                    for (int i = 0; i < 1500; i++)
                    {
                        residualConnectionOutputNorm[i] = (residualConnectionOutputNorm[i] - mean1_block2[0]) / (Math.Sqrt(variance1_block2[0] + epsilon));
                    }
                }
                else
                {
                    Array.Copy(residualConnectionOutputNorm, 0, residualConnectionOutputNormIntermediate1, 0, 1500);
                    for (int i = 0; i < 1500; i++)
                    {
                        mean1[0] += residualConnectionOutputNorm[i];
                    }
                    mean1[0] /= 1500;

                    meanVarianceIdx = 0;
                    for (int i = 0; i < 1500; i++)
                    {
                        variance1[0] += Math.Pow((residualConnectionOutputNorm[i] - mean1[0]), 2);
                    }

                    meanVarianceIdx = 0;
                    for (int i = 0; i < 1500; i++)
                    {
                        residualConnectionOutputNorm[i] = (residualConnectionOutputNorm[i] - mean1[meanVarianceIdx]) / (Math.Sqrt(variance1[meanVarianceIdx] + epsilon));
                    }
                }

                scaleAndShift(1);
                */
            }
            if (pass == 2)
            {
                //implement our residual connection from input to affineMLP to output of affineMLP
                for (int i = 0; i < 1500; i++)
                {
                    transformerBlockFinalOutput[i] += residualConnectionOutputNorm[i];
                }
                /*
                matrixOps matOps = new matrixOps();

                double feature_summation = 0;
                double var_summation = 0;
                int meanVarianceIdx = 0;
                epsilon = backProp.layerNormEpsilon; // 1 * (10 ^ -5) APPARENTLY C# doesn't know how to do scientific notation
                                                     //one times ten to the -5 dumbass, apparently it should be 1e-05

                if (blockNum == 2)
                {
                    Array.Copy(transformerBlockFinalOutput, 0, transformerBlockFinalOutputIntermediate2, 0, 1500);
                    for (int i = 0; i < 1500; i++)
                    {
                        mean2_block2[meanVarianceIdx] += transformerBlockFinalOutput[i];
                    }
                    mean2_block2[0] /= 1500;

                    meanVarianceIdx = 0;
                    for (int i = 0; i < 1500; i++)
                    {
                        variance2_block2[meanVarianceIdx] += Math.Pow((transformerBlockFinalOutput[i] - mean2_block2[meanVarianceIdx]), 2);
                    }

                    meanVarianceIdx = 0;
                    for (int i = 0; i < 1500; i++)
                    {
                        transformerBlockFinalOutput[i] = (transformerBlockFinalOutput[i] - mean2_block2[meanVarianceIdx]) / (Math.Sqrt(variance2_block2[meanVarianceIdx] + epsilon));
                    }
                }
                else
                {
                    Array.Copy(transformerBlockFinalOutput, 0, transformerBlockFinalOutputIntermediate1, 0, 1500);
                    for (int i = 0; i < 1500; i++)
                    {
                        mean2[meanVarianceIdx]  += transformerBlockFinalOutput[i];
                    }
                    mean2[0] /= 1500;

                    meanVarianceIdx = 0;
                    for (int i = 0; i < 1500; i++)
                    {
                        variance2[meanVarianceIdx] += Math.Pow((transformerBlockFinalOutput[i] - mean2[meanVarianceIdx]), 2);
                    }

                    meanVarianceIdx = 0;
                    for (int i = 0; i < 1500; i++)
                    {
                        transformerBlockFinalOutput[i] = (transformerBlockFinalOutput[i] - mean2[meanVarianceIdx]) / (Math.Sqrt(variance2[meanVarianceIdx] + epsilon));
                    }
                }

                scaleAndShift(2);
                */
            }
        }

        public void scaleAndShift(int pass)
        {
            if (pass == 1)
            {
                for (int i = 0; i < 1500; i++)
                {
                    residualConnectionOutputNorm[i] = (addAndNorm1Gamma[i] * residualConnectionOutputNorm[i]) + addAndNorm1Beta[i];
                }
            }
            if (pass == 2)
            {
                for (int i = 0; i < 1500; i++)
                {
                    transformerBlockFinalOutput[i] = (addAndNorm2Gamma[i] * transformerBlockFinalOutput[i]) + addAndNorm2Beta[i];
                }
            }
        }

        public void affineTransformMLP(int block)
        {
            if (block == 1)
            {
                matrixOps matOps = new matrixOps();
                double[] temp;
                int M = 100;
                int K = 15;
                int N = 60;

                temp = matOps.matrixMul(residualConnectionOutputNorm, affineTransWeights1, M, K, N);
                Array.Copy(temp, 0, affineIntermediateRes, 0, 6000);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\affineIntermediateRes.txt");
                    for (int i = 0; i < 6000; i++)
                    {
                        output.WriteLine("Intermediate Affine Transformed Matrix[" + i.ToString() + "] = " + affineIntermediateRes[i].ToString());
                    }
                    output.Close();
                }

                //apply PReLU
                //transformerMLP_PReLU_and_add_bias1();
                //apply Mish
                transformerMLP_Mish_and_add_bias1();

                M = 100;
                K = 60;
                N = 15;

                temp = matOps.matrixMul(affineIntermediateRes, affineTransWeights2, M, K, N);
                Array.Copy(temp, 0, transformerBlockFinalOutput, 0, 1500);

                transformerMLP_add_bias_to_output();

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\transformerBlockFinalOutput1.txt");
                    for (int i = 0; i < 1500; i++)
                    {
                        output.WriteLine("Transformer Block 1 Output[" + i.ToString() + "] = " + transformerBlockFinalOutput[i].ToString());
                    }
                    output.Close();
                }
            }
            if (block == 2)
            {
                matrixOps matOps = new matrixOps();
                double[] temp;
                int M = 100;
                int K = 15;
                int N = 60;

                temp = matOps.matrixMul(residualConnectionOutputNorm, affineTransWeights1, M, K, N);
                Array.Copy(temp, 0, affineIntermediateRes2, 0, 6000);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\affineIntermediateRes2.txt");
                    for (int i = 0; i < 6000; i++)
                    {
                        output.WriteLine("Intermediate Affine Transformed Matrix 2[" + i.ToString() + "] = " + affineIntermediateRes2[i].ToString());
                    }
                    output.Close();
                }

                //apply PReLU
                //transformerMLP_PReLU_and_add_bias2();
                //apply Mish
                transformerMLP_Mish_and_add_bias2();

                M = 100;
                K = 60;
                N = 15;

                temp = matOps.matrixMul(affineIntermediateRes2, affineTransWeights2, M, K, N);
                Array.Copy(temp, 0, transformerBlockFinalOutput, 0, 1500);

                transformerMLP_add_bias_to_output();

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\transformerBlockFinalOutput2.txt");
                    for (int i = 0; i < 1500; i++)
                    {
                        output.WriteLine("Transformer Block 2 Output[" + i.ToString() + "] = " + transformerBlockFinalOutput[i].ToString());
                    }
                    output.Close();
                }
            }
        }

        public void transformerMLP_PReLU_and_add_bias1()
        {
            for (int i = 0; i < 6000; i++)
            {
                if (affineIntermediateRes[i] + transPReLUBias[i] > 0.0F)
                {
                    affineIntermediateRes[i] += transPReLUBias[i];
                }
                else
                {
                    affineIntermediateRes[i] = 0;
                }
            }
        }

        public void transformerMLP_PReLU_and_add_bias2()
        {
            for (int i = 0; i < 6000; i++)
            {
                if (affineIntermediateRes2[i] + transPReLUBias[i] > 0.0F)
                {
                    affineIntermediateRes2[i] += transPReLUBias[i];
                }
                else
                {
                    affineIntermediateRes2[i] = 0;
                }
            }
        }

        public void transformerMLP_Mish_and_add_bias1()
        {
            for (int i = 0; i < 6000; i++)
            {
                affineIntermediateRes[i] = (affineIntermediateRes[i] + transPReLUBias[i]) * Math.Tanh(softplus(affineIntermediateRes[i] + transPReLUBias[i]));
            }
        }

        public void transformerMLP_Mish_and_add_bias2()
        {
            for (int i = 0; i < 6000; i++)
            {
                affineIntermediateRes2[i] = (affineIntermediateRes2[i] + transPReLUBias[i]) * Math.Tanh(softplus(affineIntermediateRes2[i] + transPReLUBias[i]));
            }
        }

        public double softplus(double x)
        {
            double temp;
            temp = Math.Log(1 + Math.Exp(x));
            return temp;
        }

        public void transformerMLP_add_bias_to_output()
        {
            for (int i = 0; i < 1500; i++)
            {
                transformerBlockFinalOutput[i] += transMLPSecondLayerBias[i];
            }
        }

        public void transMLPBiases_init()
        {
            if (!File.Exists(@"X:\affineMLPBiasesFlatFile.txt"))
            {
                StreamWriter output = File.AppendText(@"X:\affineMLPBiasesFlatFile.txt");
                for (int i = 0; i < 6000; i++)
                {
                    transPReLUBias[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output.WriteLine(transPReLUBias[i].ToString());
                }
                output.Close();
            }
            else
            {
                string[] arr;
                arr = File.ReadAllLines(@"X:\affineMLPBiasesFlatFile.txt");
                for (int i = 0; i < 6000; i++)
                {
                    transPReLUBias[i] = Convert.ToDouble(arr[i]);
                    predictorGui.numOfLearnableParams++;
                }
            }
            if (!File.Exists(@"X:\affineMLPSecondLayerBiasesFlatFile.txt"))
            {
                StreamWriter output = File.AppendText(@"X:\affineMLPSecondLayerBiasesFlatFile.txt");
                for (int i = 0; i < 1500; i++)
                {
                    transMLPSecondLayerBias[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output.WriteLine(transMLPSecondLayerBias[i].ToString());
                }
                output.Close();
            }
            else
            {
                string[] arr;
                arr = File.ReadAllLines(@"X:\affineMLPSecondLayerBiasesFlatFile.txt");
                for (int i = 0; i < 1500; i++)
                {
                    transMLPSecondLayerBias[i] = Convert.ToDouble(arr[i]);
                    predictorGui.numOfLearnableParams++;
                }
            }
        }

        public void transMLPPReLUParams_init()
        {
            if (!File.Exists(@"X:\affineMLPPreluParamFlatFile.txt"))
            {
                StreamWriter output = File.AppendText(@"X:\affineMLPPreluParamFlatFile.txt");
                for (int i = 0; i < 6000; i++)
                {
                    transPReLUParam[i] = 0.02F;
                    predictorGui.numOfLearnableParams++;
                    output.WriteLine(transPReLUParam[i].ToString());
                }
                output.Close();
            }
            else
            {
                string[] arr;
                arr = File.ReadAllLines(@"X:\affineMLPPreluParamFlatFile.txt");
                for (int i = 0; i < 6000; i++)
                {
                    transPReLUParam[i] = Convert.ToDouble(arr[i]);
                    predictorGui.numOfLearnableParams++;
                }
            }
        }

        public void addAndNormGammaBetaInit()
        {
            if (!File.Exists(@"X:\addAndNorm1GammaFlatFile.txt"))
            {
                StreamWriter output = File.AppendText(@"X:\addAndNorm1GammaFlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\addAndNorm1BetaFlatFile.txt");
                StreamWriter output3 = File.AppendText(@"X:\addAndNorm2GammaFlatFile.txt");
                StreamWriter output4 = File.AppendText(@"X:\addAndNorm2BetaFlatFile.txt");
                for (int i = 0; i < 1500; i++)
                {
                    addAndNorm1Gamma[i] = 1;
                    predictorGui.numOfLearnableParams++;
                    output.WriteLine(addAndNorm1Gamma[i]);
                    addAndNorm1Beta[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output2.WriteLine(addAndNorm1Beta[i]);
                    addAndNorm2Gamma[i] = 1;
                    predictorGui.numOfLearnableParams++;
                    output3.WriteLine(addAndNorm2Gamma[i]);
                    addAndNorm2Beta[i] = 0;
                    predictorGui.numOfLearnableParams++;
                    output4.WriteLine(addAndNorm2Beta[i]);
                }
                output.Close();
                output2.Close();
                output3.Close();
                output4.Close();
            }
            else
            {
                string[] arr;
                arr = File.ReadAllLines(@"X:\addAndNorm1GammaFlatFile.txt");
                string[] arr2;
                arr2 = File.ReadAllLines(@"X:\addAndNorm1BetaFlatFile.txt");
                string[] arr3;
                arr3 = File.ReadAllLines(@"X:\addAndNorm2GammaFlatFile.txt");
                string[] arr4;
                arr4 = File.ReadAllLines(@"X:\addAndNorm2BetaFlatFile.txt");
                for (int i = 0; i < 1500; i++)
                {
                    addAndNorm1Gamma[i] = Convert.ToDouble(arr[i]);
                    predictorGui.numOfLearnableParams++;
                    addAndNorm1Beta[i] = Convert.ToDouble(arr2[i]);
                    predictorGui.numOfLearnableParams++;
                    addAndNorm2Gamma[i] = Convert.ToDouble(arr3[i]);
                    predictorGui.numOfLearnableParams++;
                    addAndNorm2Beta[i] = Convert.ToDouble(arr4[i]);
                    predictorGui.numOfLearnableParams++;
                }
            }
        }

        public void tfixupInit_attention_linearLayer(int layerNum)
        {
            if (layerNum == 1)
            {
                //calculate standard deviation (range) for the weights
                double fan_in = 1500;
                double fan_out = 500;
                double upper = 1 / Math.Sqrt(fan_in);
                double lower = -(1 / Math.Sqrt(fan_in));

                if (!File.Exists(@"X:\queryLinearLayerWeightsHead1FlatFile.txt"))
                {
                    StreamWriter output = File.AppendText(@"X:\queryLinearLayerWeightsHead1FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\keyLinearLayerWeightsHead1FlatFile.txt");
                    StreamWriter output3 = File.AppendText(@"X:\valueLinearLayerWeightsHead1FlatFile.txt");
                    StreamWriter output4 = File.AppendText(@"X:\queryLinearLayerWeightsHead2FlatFile.txt");
                    StreamWriter output5 = File.AppendText(@"X:\keyLinearLayerWeightsHead2FlatFile.txt");
                    StreamWriter output6 = File.AppendText(@"X:\valueLinearLayerWeightsHead2FlatFile.txt");
                    StreamWriter output7 = File.AppendText(@"X:\queryLinearLayerWeightsHead3FlatFile.txt");
                    StreamWriter output8 = File.AppendText(@"X:\keyLinearLayerWeightsHead3FlatFile.txt");
                    StreamWriter output9 = File.AppendText(@"X:\valueLinearLayerWeightsHead3FlatFile.txt");

                    for (int i = 0; i < 75; i++)
                    {
                        queryLinearLayerWeights_head1[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(queryLinearLayerWeights_head1[i].ToString());
                        keyLinearLayerWeights_head1[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output2.WriteLine(keyLinearLayerWeights_head1[i].ToString());
                        valueLinearLayerWeights_head1[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output3.WriteLine(valueLinearLayerWeights_head1[i].ToString());

                        queryLinearLayerWeights_head2[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output4.WriteLine(queryLinearLayerWeights_head2[i].ToString());
                        keyLinearLayerWeights_head2[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output5.WriteLine(keyLinearLayerWeights_head2[i].ToString());
                        valueLinearLayerWeights_head2[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output6.WriteLine(valueLinearLayerWeights_head2[i].ToString());

                        queryLinearLayerWeights_head3[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output7.WriteLine(queryLinearLayerWeights_head3[i].ToString());
                        keyLinearLayerWeights_head3[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output8.WriteLine(keyLinearLayerWeights_head3[i].ToString());
                        valueLinearLayerWeights_head3[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output9.WriteLine(valueLinearLayerWeights_head3[i].ToString());
                    }
                    output.Close();
                    output2.Close();
                    output3.Close();
                    output4.Close();
                    output5.Close();
                    output6.Close();
                    output7.Close();
                    output8.Close();
                    output9.Close();
                }
                else
                {
                    string[] arr;
                    arr = File.ReadAllLines(@"X:\queryLinearLayerWeightsHead1FlatFile.txt");
                    string[] arr2;
                    arr2 = File.ReadAllLines(@"X:\keyLinearLayerWeightsHead1FlatFile.txt");
                    string[] arr3;
                    arr3 = File.ReadAllLines(@"X:\valueLinearLayerWeightsHead1FlatFile.txt");
                    string[] arr4;
                    arr4 = File.ReadAllLines(@"X:\queryLinearLayerWeightsHead2FlatFile.txt");
                    string[] arr5;
                    arr5 = File.ReadAllLines(@"X:\keyLinearLayerWeightsHead2FlatFile.txt");
                    string[] arr6;
                    arr6 = File.ReadAllLines(@"X:\valueLinearLayerWeightsHead2FlatFile.txt");
                    string[] arr7;
                    arr7 = File.ReadAllLines(@"X:\queryLinearLayerWeightsHead3FlatFile.txt");
                    string[] arr8;
                    arr8 = File.ReadAllLines(@"X:\keyLinearLayerWeightsHead3FlatFile.txt");
                    string[] arr9;
                    arr9 = File.ReadAllLines(@"X:\valueLinearLayerWeightsHead3FlatFile.txt");

                    for (int i = 0; i < 75; i++)
                    {
                        queryLinearLayerWeights_head1[i] = Convert.ToDouble(arr[i]);
                        predictorGui.numOfLearnableParams++;
                        keyLinearLayerWeights_head1[i] = Convert.ToDouble(arr2[i]);
                        predictorGui.numOfLearnableParams++;
                        valueLinearLayerWeights_head1[i] = Convert.ToDouble(arr3[i]);
                        predictorGui.numOfLearnableParams++;

                        queryLinearLayerWeights_head2[i] = Convert.ToDouble(arr4[i]);
                        predictorGui.numOfLearnableParams++;
                        keyLinearLayerWeights_head2[i] = Convert.ToDouble(arr5[i]);
                        predictorGui.numOfLearnableParams++;
                        valueLinearLayerWeights_head2[i] = Convert.ToDouble(arr6[i]);
                        predictorGui.numOfLearnableParams++;

                        queryLinearLayerWeights_head3[i] = Convert.ToDouble(arr7[i]);
                        predictorGui.numOfLearnableParams++;
                        keyLinearLayerWeights_head3[i] = Convert.ToDouble(arr8[i]);
                        predictorGui.numOfLearnableParams++;
                        valueLinearLayerWeights_head3[i] = Convert.ToDouble(arr9[i]);
                        predictorGui.numOfLearnableParams++;
                    }
                }

                fan_in = 1500;
                fan_out = 1500;
                upper = 1 / Math.Sqrt(fan_in);
                lower = -(1 / Math.Sqrt(fan_in));

                if (!File.Exists(@"X:\finalLinearLayerWeightsFlatFile.txt"))
                {
                    StreamWriter output = File.AppendText(@"X:\finalLinearLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 225; i++)
                    {
                        finalLinearLayerWeights[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(finalLinearLayerWeights[i].ToString());
                    }
                    output.Close();
                }
                else
                {
                    string[] arr = File.ReadAllLines(@"X:\finalLinearLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 225; i++)
                    {
                        finalLinearLayerWeights[i] = Convert.ToDouble(arr[i]);
                        predictorGui.numOfLearnableParams++;
                    }
                }
            }
        }

        public void xavierInit_attention_linearLayer(int layerNum)
        {
            if (layerNum == 1)
            {
                //calculate standard deviation (range) for the weights
                double fan_in = 1500;
                double fan_out = 500;
                double upper = 1 / Math.Sqrt(fan_in);
                double lower = -(1 / Math.Sqrt(fan_in));

                if (!File.Exists(@"X:\queryLinearLayerWeightsHead1FlatFile.txt"))
                {
                    StreamWriter output = File.AppendText(@"X:\queryLinearLayerWeightsHead1FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\keyLinearLayerWeightsHead1FlatFile.txt");
                    StreamWriter output3 = File.AppendText(@"X:\valueLinearLayerWeightsHead1FlatFile.txt");
                    StreamWriter output4 = File.AppendText(@"X:\queryLinearLayerWeightsHead2FlatFile.txt");
                    StreamWriter output5 = File.AppendText(@"X:\keyLinearLayerWeightsHead2FlatFile.txt");
                    StreamWriter output6 = File.AppendText(@"X:\valueLinearLayerWeightsHead2FlatFile.txt");
                    StreamWriter output7 = File.AppendText(@"X:\queryLinearLayerWeightsHead3FlatFile.txt");
                    StreamWriter output8 = File.AppendText(@"X:\keyLinearLayerWeightsHead3FlatFile.txt");
                    StreamWriter output9 = File.AppendText(@"X:\valueLinearLayerWeightsHead3FlatFile.txt");

                    for (int i = 0; i < 75; i++)
                    {
                        queryLinearLayerWeights_head1[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(queryLinearLayerWeights_head1[i].ToString());
                        keyLinearLayerWeights_head1[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output2.WriteLine(keyLinearLayerWeights_head1[i].ToString());
                        valueLinearLayerWeights_head1[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output3.WriteLine(valueLinearLayerWeights_head1[i].ToString());

                        queryLinearLayerWeights_head2[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output4.WriteLine(queryLinearLayerWeights_head2[i].ToString());
                        keyLinearLayerWeights_head2[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output5.WriteLine(keyLinearLayerWeights_head2[i].ToString());
                        valueLinearLayerWeights_head2[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output6.WriteLine(valueLinearLayerWeights_head2[i].ToString());

                        queryLinearLayerWeights_head3[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output7.WriteLine(queryLinearLayerWeights_head3[i].ToString());
                        keyLinearLayerWeights_head3[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output8.WriteLine(keyLinearLayerWeights_head3[i].ToString());
                        valueLinearLayerWeights_head3[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output9.WriteLine(valueLinearLayerWeights_head3[i].ToString());
                    }
                    output.Close();
                    output2.Close();
                    output3.Close();
                    output4.Close();
                    output5.Close();
                    output6.Close();
                    output7.Close();
                    output8.Close();
                    output9.Close();
                }
                else
                {
                    string[] arr;
                    arr = File.ReadAllLines(@"X:\queryLinearLayerWeightsHead1FlatFile.txt");
                    string[] arr2;
                    arr2 = File.ReadAllLines(@"X:\keyLinearLayerWeightsHead1FlatFile.txt");
                    string[] arr3;
                    arr3 = File.ReadAllLines(@"X:\valueLinearLayerWeightsHead1FlatFile.txt");
                    string[] arr4;
                    arr4 = File.ReadAllLines(@"X:\queryLinearLayerWeightsHead2FlatFile.txt");
                    string[] arr5;
                    arr5 = File.ReadAllLines(@"X:\keyLinearLayerWeightsHead2FlatFile.txt");
                    string[] arr6;
                    arr6 = File.ReadAllLines(@"X:\valueLinearLayerWeightsHead2FlatFile.txt");
                    string[] arr7;
                    arr7 = File.ReadAllLines(@"X:\queryLinearLayerWeightsHead3FlatFile.txt");
                    string[] arr8;
                    arr8 = File.ReadAllLines(@"X:\keyLinearLayerWeightsHead3FlatFile.txt");
                    string[] arr9;
                    arr9 = File.ReadAllLines(@"X:\valueLinearLayerWeightsHead3FlatFile.txt");

                    for (int i = 0; i < 75; i++)
                    {
                        queryLinearLayerWeights_head1[i] = Convert.ToDouble(arr[i]);
                        predictorGui.numOfLearnableParams++;
                        keyLinearLayerWeights_head1[i] = Convert.ToDouble(arr2[i]);
                        predictorGui.numOfLearnableParams++;
                        valueLinearLayerWeights_head1[i] = Convert.ToDouble(arr3[i]);
                        predictorGui.numOfLearnableParams++;

                        queryLinearLayerWeights_head2[i] = Convert.ToDouble(arr4[i]);
                        predictorGui.numOfLearnableParams++;
                        keyLinearLayerWeights_head2[i] = Convert.ToDouble(arr5[i]);
                        predictorGui.numOfLearnableParams++;
                        valueLinearLayerWeights_head2[i] = Convert.ToDouble(arr6[i]);
                        predictorGui.numOfLearnableParams++;

                        queryLinearLayerWeights_head3[i] = Convert.ToDouble(arr7[i]);
                        predictorGui.numOfLearnableParams++;
                        keyLinearLayerWeights_head3[i] = Convert.ToDouble(arr8[i]);
                        predictorGui.numOfLearnableParams++;
                        valueLinearLayerWeights_head3[i] = Convert.ToDouble(arr9[i]);
                        predictorGui.numOfLearnableParams++;
                    }
                }

                fan_in = 1500;
                fan_out = 1500;
                upper = 1 / Math.Sqrt(fan_in);
                lower = -(1 / Math.Sqrt(fan_in));

                if (!File.Exists(@"X:\finalLinearLayerWeightsFlatFile.txt"))
                {
                    StreamWriter output = File.AppendText(@"X:\finalLinearLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 225; i++)
                    {
                        finalLinearLayerWeights[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(finalLinearLayerWeights[i].ToString());
                    }
                    output.Close();
                }
                else
                {
                    string[] arr = File.ReadAllLines(@"X:\finalLinearLayerWeightsFlatFile.txt");
                    for (int i = 0; i < 225; i++)
                    {
                        finalLinearLayerWeights[i] = Convert.ToDouble(arr[i]);
                        predictorGui.numOfLearnableParams++;
                    }
                }
            }
        }

        public void tfixup_init_affineMLPLayers(int layerNum)
        {
            if (layerNum == 1)
            {
                double fan_in = 1500;
                double fan_out = 6000;
                double upper = 1 / Math.Sqrt(fan_in);
                double lower = -(1 / Math.Sqrt(fan_in));

                if (!File.Exists(@"X:\affineMLPTransformerWeightsFlatFile1.txt"))
                {
                    StreamWriter output = File.AppendText(@"X:\affineMLPTransformerWeightsFlatFile1.txt");
                    StreamWriter output2 = File.AppendText(@"X:\affineMLPTransformerWeightsFlatFile2.txt");

                    for (int i = 0; i < 900; i++)
                    {
                        affineTransWeights1[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(affineTransWeights1[i].ToString());
                    }
                    output.Close();

                    fan_in = 6000;
                    fan_out = 1500;
                    upper = 1 / Math.Sqrt(fan_in);
                    lower = -(1 / Math.Sqrt(fan_in));

                    for (int i = 0; i < 900; i++)
                    {
                        affineTransWeights2[i] = (lower + (predictorGui.rand.NextDouble() * (upper - lower))) / (0.67 * Math.Pow(2, -0.25));
                        predictorGui.numOfLearnableParams++;
                        output2.WriteLine(affineTransWeights2[i]);
                    }
                    output2.Close();
                }
                else
                {
                    string[] arr;
                    arr = File.ReadAllLines(@"X:\affineMLPTransformerWeightsFlatFile1.txt");
                    string[] arr2;
                    arr2 = File.ReadAllLines(@"X:\affineMLPTransformerWeightsFlatFile2.txt");

                    for (int i = 0; i < 900; i++)
                    {
                        affineTransWeights1[i] = Convert.ToDouble(arr[i]);
                        predictorGui.numOfLearnableParams++;
                        affineTransWeights2[i] = Convert.ToDouble(arr2[i]);
                        predictorGui.numOfLearnableParams++;
                    }
                }
            }
        }

        public void xavier_init_affineMLPLayers(int layerNum)
        {
            if (layerNum == 1)
            {
                double fan_in = 1500;
                double fan_out = 6000;
                double upper = 1 / Math.Sqrt(fan_in);
                double lower = -(1 / Math.Sqrt(fan_in));

                if (!File.Exists(@"X:\affineMLPTransformerWeightsFlatFile1.txt"))
                {
                    StreamWriter output = File.AppendText(@"X:\affineMLPTransformerWeightsFlatFile1.txt");
                    StreamWriter output2 = File.AppendText(@"X:\affineMLPTransformerWeightsFlatFile2.txt");

                    for (int i = 0; i < 900; i++)
                    {
                        affineTransWeights1[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output.WriteLine(affineTransWeights1[i].ToString());
                    }

                    output.Close();

                    fan_in = 6000;
                    fan_out = 1500;
                    upper = Math.Sqrt(6.0) / Math.Sqrt(fan_in + fan_out);
                    lower = -(Math.Sqrt(6.0) / Math.Sqrt(fan_in + fan_out));
                    for (int i = 0; i < 900; i++)
                    {
                        affineTransWeights2[i] = lower + (predictorGui.rand.NextDouble() * (upper - lower));
                        predictorGui.numOfLearnableParams++;
                        output2.WriteLine(affineTransWeights2[i]);
                    }
                    output2.Close();
                }
                else
                {
                    string[] arr;
                    arr = File.ReadAllLines(@"X:\affineMLPTransformerWeightsFlatFile1.txt");
                    string[] arr2;
                    arr2 = File.ReadAllLines(@"X:\affineMLPTransformerWeightsFlatFile2.txt");

                    for (int i = 0; i < 900; i++)
                    {
                        affineTransWeights1[i] = Convert.ToDouble(arr[i]);
                        predictorGui.numOfLearnableParams++;
                        affineTransWeights2[i] = Convert.ToDouble(arr2[i]);
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
    }
}
