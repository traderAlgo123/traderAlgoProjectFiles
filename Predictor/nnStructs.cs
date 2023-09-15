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

namespace Predictor
{
    public class nnStructsArray
    {
        public nnConvStructs[] convStructs = new nnConvStructs[predictorGui.miniBatchSize];
        public nnTransStructs[] transStructs = new nnTransStructs[predictorGui.miniBatchSize];
        public nnMLPStructs[] mlpStructs = new nnMLPStructs[predictorGui.miniBatchSize];
    }
    public class nnConvStructs
    {
        public double[] mean = new double[14];
        public double[] variance = new double[14];
        public double epsilon;

        public inputTensorKernel[] convLayer1Kernel1 = new inputTensorKernel[14];
        public hiddenTensorKernel[] convLayer2Kernel2 = new hiddenTensorKernel[14];
        public hiddenTensorKernel[] convLayer3Kernel3 = new hiddenTensorKernel[14];
        public hiddenTensorKernel[] convLayer4Kernel4 = new hiddenTensorKernel[14];
        public hiddenTensorKernel[] convLayer5Kernel5 = new hiddenTensorKernel[14];

        public double[] convLayer1Bias = new double[1400];
        public double[] convLayer2Bias = new double[1400];
        public double[] convLayer3Bias = new double[1400];
        public double[] convLayer4Bias = new double[1400];
        public double[] convLayer5Bias = new double[1400];

        //public nnChromosomeWeights[] convLayer1Bias_chromo = new nnChromosomeWeights[1400];
        //public nnChromosomeWeights[] convLayer2Bias_chromo = new nnChromosomeWeights[1400];
        //public nnChromosomeWeights[] convLayer3Bias_chromo = new nnChromosomeWeights[1400];
        //public nnChromosomeWeights[] convLayer4Bias_chromo = new nnChromosomeWeights[1400];
        //public nnChromosomeWeights[] convLayer5Bias_chromo = new nnChromosomeWeights[1400];

        public double[] convLayer1PReLUParam = new double[1400];
        public double[] convLayer2PReLUParam = new double[1400];
        public double[] convLayer3PReLUParam = new double[1400];
        public double[] convLayer4PReLUParam = new double[1400];
        public double[] convLayer5PReLUParam = new double[1400];

        //public nnChromosomeWeights[] convLayer1PReLUParam_chromo = new nnChromosomeWeights[1400];
        //public nnChromosomeWeights[] convLayer2PReLUParam_chromo = new nnChromosomeWeights[1400];
        //public nnChromosomeWeights[] convLayer3PReLUParam_chromo = new nnChromosomeWeights[1400];
        //public nnChromosomeWeights[] convLayer4PReLUParam_chromo = new nnChromosomeWeights[1400];
        //public nnChromosomeWeights[] convLayer5PReLUParam_chromo = new nnChromosomeWeights[1400];

        public double[] convLayer1Output = new double[1400];
        public double[] convLayer2Output = new double[1400];
        public double[] convLayer3Output = new double[1400];
        public double[] convLayer4Output = new double[1400];
        public double[] convLayer5Output = new double[1400];

        public double[] convLayer4OutPadded = new double[1624];
        public double[] convLayer3OutPadded = new double[1512];
        public double[] convLayer2OutPadded = new double[1456];
        public double[] convLayer1OutPadded = new double[1428];

        public double[] convLayer5OutputNorm = new double[1400];
        public double[] convLayer5OutputNormGamma = new double[1400];
        public double[] convLayer5OutputNormBeta = new double[1400];

        public double[] temporalEncodedNormOutput = new double[1500];
    }

    public class nnTransStructs
    {
        public double[] mean1 = new double[15];
        public double[] mean1_block2 = new double[15];
        public double[] mean2 = new double[15];
        public double[] mean2_block2 = new double[15];
        public double[] variance1 = new double[15];
        public double[] variance1_block2 = new double[15];
        public double[] variance2 = new double[15];
        public double[] variance2_block2 = new double[15];
        public double epsilon;
        public double[] positionalEncodingArray = new double[1500];
        public double[] positionalEncodingArrayCpy = new double[1500];
        public double[] inputFromConvModule = new double[1500];
        public double[] inputFromConvModuleCpy = new double[1500];

        public double[] queryLinearLayerWeights_head1 = new double[75];
        public double[] keyLinearLayerWeights_head1 = new double[75];
        public double[] valueLinearLayerWeights_head1 = new double[75];

        //public nnChromosomeWeights[] queryLinearLayerWeights_head1_chromo = new nnChromosomeWeights[75];
        //public nnChromosomeWeights[] keyLinearLayerWeights_head1_chromo = new nnChromosomeWeights[75];
        //public nnChromosomeWeights[] valueLinearLayerWeights_head1_chromo = new nnChromosomeWeights[75];

        public double[] queryLinearLayerWeights_head2 = new double[75];
        public double[] keyLinearLayerWeights_head2 = new double[75];
        public double[] valueLinearLayerWeights_head2 = new double[75];

        //public nnChromosomeWeights[] queryLinearLayerWeights_head2_chromo = new nnChromosomeWeights[75];
        //public nnChromosomeWeights[] keyLinearLayerWeights_head2_chromo = new nnChromosomeWeights[75];
        //public nnChromosomeWeights[] valueLinearLayerWeights_head2_chromo = new nnChromosomeWeights[75];

        public double[] queryLinearLayerWeights_head3 = new double[75];
        public double[] keyLinearLayerWeights_head3 = new double[75];
        public double[] valueLinearLayerWeights_head3 = new double[75];

        //public nnChromosomeWeights[] queryLinearLayerWeights_head3_chromo = new nnChromosomeWeights[75];
        //public nnChromosomeWeights[] keyLinearLayerWeights_head3_chromo = new nnChromosomeWeights[75];
        //public nnChromosomeWeights[] valueLinearLayerWeights_head3_chromo = new nnChromosomeWeights[75];

        public double[] query_head1 = new double[500];
        public double[] key_head1 = new double[500];
        public double[] value_head1 = new double[500];
        public double[] filtered_value_head1 = new double[500];
        public double[] query_head1Cpy = new double[500];
        public double[] key_head1Cpy = new double[500];
        public double[] value_head1Cpy = new double[500];
        public double[] filtered_value_head1Cpy = new double[500];

        public double[] query_head2 = new double[500];
        public double[] key_head2 = new double[500];
        public double[] value_head2 = new double[500];
        public double[] filtered_value_head2 = new double[500];
        public double[] query_head2Cpy = new double[500];
        public double[] key_head2Cpy = new double[500];
        public double[] value_head2Cpy = new double[500];
        public double[] filtered_value_head2Cpy = new double[500];

        public double[] query_head3 = new double[500];
        public double[] key_head3 = new double[500];
        public double[] value_head3 = new double[500];
        public double[] filtered_value_head3 = new double[500];
        public double[] query_head3Cpy = new double[500];
        public double[] key_head3Cpy = new double[500];
        public double[] value_head3Cpy = new double[500];
        public double[] filtered_value_head3Cpy = new double[500];

        public double[] attention_filter_head1 = new double[10000];
        public double[] attention_filter_head2 = new double[10000];
        public double[] attention_filter_head3 = new double[10000];
        public double[] attention_filter_head1Cpy = new double[10000];
        public double[] attention_filter_head2Cpy = new double[10000];
        public double[] attention_filter_head3Cpy = new double[10000];

        public double[] concatenatedFilteredValueMatrix = new double[1500];
        public double[] concatenatedFilteredValueMatrixCpy = new double[1500];

        public double[] finalLinearLayerWeights = new double[225];
        //public nnChromosomeWeights[] finalLinearLayerWeights_chromo = new nnChromosomeWeights[225];

        public double[] finalAttentionBlockOutput = new double[1500];
        public double[] finalAttentionBlockOutputCpy = new double[1500];

        public double[] residualConnectionOutputNorm = new double[1500];
        public double[] residualConnectionOutputNormCpy = new double[1500];

        public double[] residualConnectionOutputNormIntermediate1 = new double[1500];
        public double[] residualConnectionOutputNormIntermediate2 = new double[1500];

        public double[] addAndNorm1Gamma = new double[1500];
        public double[] addAndNorm1Beta = new double[1500];
        public double[] addAndNorm2Gamma = new double[1500];
        public double[] addAndNorm2Beta = new double[1500];

        public double[] affineTransWeights1 = new double[900];
        public double[] affineTransWeights2 = new double[900];

        //public nnChromosomeWeights[] affineTransWeights1_chromo = new nnChromosomeWeights[900];
        //public nnChromosomeWeights[] affineTransWeights2_chromo = new nnChromosomeWeights[900];

        public double[] transPReLUParam = new double[6000];
        public double[] transPReLUBias = new double[6000];

        //public nnChromosomeWeights[] transPReLUParam_chromo = new nnChromosomeWeights[6000];
        //public nnChromosomeWeights[] transPReLUBias_chromo = new nnChromosomeWeights[6000];

        public double[] transMLPSecondLayerBias = new double[1500];

        //public nnChromosomeWeights[] transMLPSecondLayerBias_chromo = new nnChromosomeWeights[1500];

        public double[] affineIntermediateRes = new double[6000];
        public double[] affineIntermediateRes2 = new double[6000];
        public double[] affineIntermediateRes3 = new double[6000];
        public double[] affineIntermediateRes4 = new double[6000];

        public double[] transformerInput = new double[1500];
        public double[] transformerBlock1Output = new double[1500];
        public double[] transformerBlock2Output = new double[1500];

        public double[] transformerBlockFinalOutput = new double[1500];
        public double[] transformerBlockFinalOutput2 = new double[1500];
        public double[] transformerBlockFinalOutputIntermediate1 = new double[1500];
        public double[] transformerBlockFinalOutputIntermediate2 = new double[1500];
    }

    public class nnMLPStructs
    {
        public double[] firstLayerWeights = new double[96000];
        public double[] mlpLayer1Bias = new double[64];
        public double[] mlpLayer1PReLUParam = new double[64];

        //public nnChromosomeWeights[] firstLayerWeights_chromo = new nnChromosomeWeights[96000];
        //public nnChromosomeWeights[] mlpLayer1Bias_chromo = new nnChromosomeWeights[64];
        //public nnChromosomeWeights[] mlpLayer1PReLUParam_chromo = new nnChromosomeWeights[64];

        public double[] secondLayerWeights = new double[192];
        public double[] mlpLayer2Bias = new double[3];
        public double[] mlpLayer2PReLUParam = new double[3];

        //public nnChromosomeWeights[] secondLayerWeights_chromo = new nnChromosomeWeights[192];
        //public nnChromosomeWeights[] mlpLayer2Bias_chromo = new nnChromosomeWeights[3];
        //public nnChromosomeWeights[] mlpLayer2PReLUParam_chromo = new nnChromosomeWeights[3];

        public double[] thirdLayerWeights = new double[9];

        public double[] firstLayerOut = new double[64];
        public double[] secondLayerOut = new double[3];
        public double[] secondLayerOutRaw = new double[3];

        public double[] dropout_mask = new double[64];

        public double[] actualOutcomes = new double[3];
        public double cross_entropy_loss_per_example = 0;
    }

    public class nnChromosomeWeights
    {
        //commented out currently due to causing the system ram to run out, need to either optimize or add more space
        //public bool[] weightAsBinaryArray = new bool[64];
        //public bool[] weightAsGrayCodedArray = new bool[64];

        /*public static void initializeChromosomes()
        {
            for (int i = 0; i < 32; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    for (int k = 0; k < 75; k++)
                    {
                        predictorGui.networkArray[i].transStructs[j].queryLinearLayerWeights_head1_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].transStructs[j].keyLinearLayerWeights_head1_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].transStructs[j].valueLinearLayerWeights_head1_chromo[k] = new nnChromosomeWeights();

                        predictorGui.networkArray[i].transStructs[j].queryLinearLayerWeights_head2_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].transStructs[j].keyLinearLayerWeights_head2_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].transStructs[j].valueLinearLayerWeights_head2_chromo[k] = new nnChromosomeWeights();

                        predictorGui.networkArray[i].transStructs[j].queryLinearLayerWeights_head3_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].transStructs[j].keyLinearLayerWeights_head3_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].transStructs[j].valueLinearLayerWeights_head3_chromo[k] = new nnChromosomeWeights();
                    }

                    for(int k = 0; k < 225; k++)
                    {
                        predictorGui.networkArray[i].transStructs[j].finalLinearLayerWeights_chromo[k] = new nnChromosomeWeights();
                    }

                    for (int k = 0; k < 900; k++)
                    {
                        predictorGui.networkArray[i].transStructs[j].affineTransWeights1_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].transStructs[j].affineTransWeights2_chromo[k] = new nnChromosomeWeights();
                    }

                    for(int k = 0; k < 96000; k++)
                    {
                        predictorGui.networkArray[i].mlpStructs[j].firstLayerWeights_chromo[k] = new nnChromosomeWeights();
                    }

                    for(int k = 0; k < 192; k++)
                    {
                        predictorGui.networkArray[i].mlpStructs[j].secondLayerWeights_chromo[k] = new nnChromosomeWeights();
                    }

                    for(int k = 0; k < 64; k++)
                    {
                        predictorGui.networkArray[i].mlpStructs[j].mlpLayer1Bias_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].mlpStructs[j].mlpLayer1PReLUParam_chromo[k] = new nnChromosomeWeights();
                    }

                    for(int k = 0; k < 3; k++)
                    {
                        predictorGui.networkArray[i].mlpStructs[j].mlpLayer2Bias_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].mlpStructs[j].mlpLayer2PReLUParam_chromo[k] = new nnChromosomeWeights();
                    }

                    for(int k = 0; k < 1400; k++)
                    {
                        predictorGui.networkArray[i].convStructs[j].convLayer1Bias_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].convStructs[j].convLayer2Bias_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].convStructs[j].convLayer3Bias_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].convStructs[j].convLayer4Bias_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].convStructs[j].convLayer5Bias_chromo[k] = new nnChromosomeWeights();
                    }

                    for(int k = 0; k < 6000; k++)
                    {
                        predictorGui.networkArray[i].transStructs[j].transPReLUBias_chromo[k] = new nnChromosomeWeights();
                        predictorGui.networkArray[i].transStructs[j].transPReLUParam_chromo[k] = new nnChromosomeWeights();
                    }

                    for(int k = 0; k < 1500; k++)
                    {
                        predictorGui.networkArray[i].transStructs[j].transMLPSecondLayerBias_chromo[k] = new nnChromosomeWeights();
                    }

                    for(int k = 0; k < 14; k++)
                    {
                        for(int m = 0; m < 32; m++)
                        {
                            predictorGui.networkArray[i].convStructs[j].convLayer1Kernel1[k].depth1_chromo[m] = new nnChromosomeWeights();
                            predictorGui.networkArray[i].convStructs[j].convLayer1Kernel1[k].depth2_chromo[m] = new nnChromosomeWeights();
                            predictorGui.networkArray[i].convStructs[j].convLayer1Kernel1[k].depth3_chromo[m] = new nnChromosomeWeights();
                            predictorGui.networkArray[i].convStructs[j].convLayer1Kernel1[k].depth4_chromo[m] = new nnChromosomeWeights();
                        }

                        for(int m = 0; m < 14; m++)
                        {
                            predictorGui.networkArray[i].convStructs[j].convLayer2Kernel2[k].depth1_chromo[m] = new nnChromosomeWeights();
                            predictorGui.networkArray[i].convStructs[j].convLayer2Kernel2[k].depth2_chromo[m] = new nnChromosomeWeights();

                            predictorGui.networkArray[i].convStructs[j].convLayer3Kernel3[k].depth1_chromo[m] = new nnChromosomeWeights();
                            predictorGui.networkArray[i].convStructs[j].convLayer3Kernel3[k].depth2_chromo[m] = new nnChromosomeWeights();

                            predictorGui.networkArray[i].convStructs[j].convLayer4Kernel4[k].depth1_chromo[m] = new nnChromosomeWeights();
                            predictorGui.networkArray[i].convStructs[j].convLayer4Kernel4[k].depth2_chromo[m] = new nnChromosomeWeights();

                            predictorGui.networkArray[i].convStructs[j].convLayer5Kernel5[k].depth1_chromo[m] = new nnChromosomeWeights();
                            predictorGui.networkArray[i].convStructs[j].convLayer5Kernel5[k].depth2_chromo[m] = new nnChromosomeWeights();
                        }
                    }
                }
            }
        }*/

        //function to convert double to binary array
        public static bool[] DoubleToBinaryArray(double doubleNumber)
        {
            // Convert the double to bytes
            byte[] bytes = BitConverter.GetBytes(doubleNumber);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(bytes);
            }

            // Convert each byte to a binary string
            string binaryString = string.Join("", bytes.Select(byteValue => Convert.ToString(byteValue, 2).PadLeft(8, '0')));

            // Convert the binary string to an array of booleans
            bool[] binaryArray = binaryString.Select(bit => bit == '1').ToArray();

            return binaryArray;
        }

        //function to convert binary array to gray code
        public static bool[] BinaryArrayToGrayCode(bool[] binaryArray)
        {

            int length = binaryArray.Length;
            bool[] grayCode = new bool[length];

            // Copy the MSB
            grayCode[0] = binaryArray[0];

            // Perform XOR operation on subsequent bits
            for (int i = 1; i < length; i++)
            {
                grayCode[i] = binaryArray[i - 1] ^ binaryArray[i];
            }

            return grayCode;
        }

        //Convert gray coded array to binary array
        public static bool[] GrayCodeToBinaryArray(bool[] grayCode)
        {
            int length = grayCode.Length;
            bool[] binaryArray = new bool[length];

            // Copy the MSB
            binaryArray[0] = grayCode[0];

            // Perform XOR operation on subsequent bits
            for (int i = 1; i < length; i++)
            {
                binaryArray[i] = binaryArray[i - 1] ^ grayCode[i];
            }

            return binaryArray;
        }

        //Convert binary array to double
        public static double BinaryArrayToDouble(bool[] binaryArray)
        {
            // Convert the binary array back to a binary string
            string binaryString = new string(binaryArray.Select(b => b ? '1' : '0').ToArray());

            // Convert the binary string to a byte array
            byte[] bytes = Enumerable.Range(0, binaryString.Length / 8)
                                 .Select(i => Convert.ToByte(binaryString.Substring(i * 8, 8), 2))
                                 .ToArray();

            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(bytes);
            }

            // Convert the byte array to a float
            double doubleNumber = BitConverter.ToDouble(bytes, 0);

            return doubleNumber;
        }
    }

    public class nnConvStructsBackProp
    {

    }

    public class nnTransStructsBackProp
    {

    }

    public class nnMLPStructsBackProp
    {

    }
}
