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

        public double[] convLayer1PReLUParam = new double[1400];
        public double[] convLayer2PReLUParam = new double[1400];
        public double[] convLayer3PReLUParam = new double[1400];
        public double[] convLayer4PReLUParam = new double[1400];
        public double[] convLayer5PReLUParam = new double[1400];

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

        public double[] queryLinearLayerWeights_head2 = new double[75];
        public double[] keyLinearLayerWeights_head2 = new double[75];
        public double[] valueLinearLayerWeights_head2 = new double[75];

        public double[] queryLinearLayerWeights_head3 = new double[75];
        public double[] keyLinearLayerWeights_head3 = new double[75];
        public double[] valueLinearLayerWeights_head3 = new double[75];

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
        public double[] affineTransWeights3 = new double[900];
        public double[] affineTransWeights4 = new double[900];

        public double[] transPReLUParam = new double[6000];
        public double[] transPReLUBias = new double[6000];

        public double[] transMLPSecondLayerBias = new double[1500];

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

        public double[] secondLayerWeights = new double[192];
        public double[] mlpLayer2Bias = new double[3];
        public double[] mlpLayer2PReLUParam = new double[3];

        public double[] thirdLayerWeights = new double[9];

        public double[] firstLayerOut = new double[64];
        public double[] secondLayerOut = new double[3];
        public double[] secondLayerOutRaw = new double[3];

        public double[] dropout_mask = new double[64];

        public double[] actualOutcomes = new double[3];
        public double cross_entropy_loss_per_example = 0;
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
