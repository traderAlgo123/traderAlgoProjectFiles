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
    public class weightsAdjList
    {
        //MLP block parameter adjustments
        public double[] mlpThirdLayerWeightsAdj = new double[9];
        public double[] mlpSecondLayerWeightsAdj = new double[192];
        public double[] mlpFirstLayerWeightsAdj = new double[96000];

        public double[] mlpLayer1BiasAdj = new double[64];
        public double[] mlpLayer2BiasAdj = new double[3];
        public double[] mlpLayer1PReLUAdj = new double[64];
        public double[] mlpLayer2PReLUAdj = new double[3];
        //end MLP block parameter adjustments

        //affine transformation MLP calculated adjustments
        //this block is for the adjustments calculated by using transformer block 2 gradient
        public double[] affineMLPSecondLayerWeightsPass2Block2Adj = new double[900];
        public double[] affineMLPSecondLayerWeightsPass1Block2Adj = new double[900];
        public double[] affineMLPFirstLayerWeightsPass2Block2Adj = new double[900];
        public double[] affineMLPFirstLayerWeightsPass1Block2Adj = new double[900];

        //this block is for the adjustments calculated by using transformer block 2 gradient
        public double[] affineMLPSecondLayerWeightsPass2Block1Adj = new double[900];
        public double[] affineMLPSecondLayerWeightsPass1Block1Adj = new double[900];
        public double[] affineMLPFirstLayerWeightsPass2Block1Adj = new double[900];
        public double[] affineMLPFirstLayerWeightsPass1Block1Adj = new double[900];

        public double[] affineMLPFirstLayerBiasPass2Block2Adj = new double[6000];
        public double[] affineMLPFirstLayerPreluPass2Block2Adj = new double[6000];

        public double[] affineMLPFirstLayerBiasPass2Block1Adj = new double[6000];
        public double[] affineMLPFirstLayerPreluPass2Block1Adj = new double[6000];

        public double[] affineMLPSecondLayerBiasBlock2Adj = new double[1500];
        public double[] affineMLPSecondLayerBiasBlock1Adj = new double[1500];
        //end affine transformation MLP calculated adjustments

        public double[] AddAndNormGamma2Adj = new double[1500];
        public double[] AddAndNormBeta2Adj = new double[1500];
        public double[] AddAndNormGamma1Adj = new double[1500];
        public double[] AddAndNormBeta1Adj = new double[1500];

        public double[] AddAndNormGamma2Adj1 = new double[1500];
        public double[] AddAndNormBeta2Adj1 = new double[1500];
        public double[] AddAndNormGamma1Adj1 = new double[1500];
        public double[] AddAndNormBeta1Adj1 = new double[1500];

        public double[] finalLinearLayerWeightsAdj = new double[225];

        public double[] queryHead1LinearLayerWeightsAdj = new double[75];
        public double[] keyHead1LinearLayerWeightsAdj = new double[75];
        public double[] valueHead1LinearLayerWeightsAdj = new double[75];

        public double[] queryHead2LinearLayerWeightsAdj = new double[75];
        public double[] keyHead2LinearLayerWeightsAdj = new double[75];
        public double[] valueHead2LinearLayerWeightsAdj = new double[75];

        public double[] queryHead3LinearLayerWeightsAdj = new double[75];
        public double[] keyHead3LinearLayerWeightsAdj = new double[75];
        public double[] valueHead3LinearLayerWeightsAdj = new double[75];

        public double[] finalLinearLayerWeightsAdj1 = new double[225];

        public double[] queryHead1LinearLayerWeightsAdj1 = new double[75];
        public double[] keyHead1LinearLayerWeightsAdj1 = new double[75];
        public double[] valueHead1LinearLayerWeightsAdj1 = new double[75];

        public double[] queryHead2LinearLayerWeightsAdj1 = new double[75];
        public double[] keyHead2LinearLayerWeightsAdj1 = new double[75];
        public double[] valueHead2LinearLayerWeightsAdj1 = new double[75];

        public double[] queryHead3LinearLayerWeightsAdj1 = new double[75];
        public double[] keyHead3LinearLayerWeightsAdj1 = new double[75];
        public double[] valueHead3LinearLayerWeightsAdj1 = new double[75];

        public double[] convLayer5BiasesAdj = new double[1400];
        public double[] convLayer5PReLUParamAdj = new double[1400];
        public double[] convLayer5NormGammaAdj = new double[1400];
        public double[] convLayer5NormBetaAdj = new double[1400];
        public double[] convLayer5Kernel1Depth1Adj = new double[14];
        public double[] convLayer5Kernel1Depth2Adj = new double[14];
        public double[] convLayer5Kernel2Depth1Adj = new double[14];
        public double[] convLayer5Kernel2Depth2Adj = new double[14];
        public double[] convLayer5Kernel3Depth1Adj = new double[14];
        public double[] convLayer5Kernel3Depth2Adj = new double[14];
        public double[] convLayer5Kernel4Depth1Adj = new double[14];
        public double[] convLayer5Kernel4Depth2Adj = new double[14];
        public double[] convLayer5Kernel5Depth1Adj = new double[14];
        public double[] convLayer5Kernel5Depth2Adj = new double[14];
        public double[] convLayer5Kernel6Depth1Adj = new double[14];
        public double[] convLayer5Kernel6Depth2Adj = new double[14];
        public double[] convLayer5Kernel7Depth1Adj = new double[14];
        public double[] convLayer5Kernel7Depth2Adj = new double[14];
        public double[] convLayer5Kernel8Depth1Adj = new double[14];
        public double[] convLayer5Kernel8Depth2Adj = new double[14];
        public double[] convLayer5Kernel9Depth1Adj = new double[14];
        public double[] convLayer5Kernel9Depth2Adj = new double[14];
        public double[] convLayer5Kernel10Depth1Adj = new double[14];
        public double[] convLayer5Kernel10Depth2Adj = new double[14];
        public double[] convLayer5Kernel11Depth1Adj = new double[14];
        public double[] convLayer5Kernel11Depth2Adj = new double[14];
        public double[] convLayer5Kernel12Depth1Adj = new double[14];
        public double[] convLayer5Kernel12Depth2Adj = new double[14];
        public double[] convLayer5Kernel13Depth1Adj = new double[14];
        public double[] convLayer5Kernel13Depth2Adj = new double[14];
        public double[] convLayer5Kernel14Depth1Adj = new double[14];
        public double[] convLayer5Kernel14Depth2Adj = new double[14];

        public double[] convLayer4BiasesAdj = new double[1400];
        public double[] convLayer4PReLUParamAdj = new double[1400];
        public double[] convLayer4Kernel1Depth1Adj = new double[14];
        public double[] convLayer4Kernel1Depth2Adj = new double[14];
        public double[] convLayer4Kernel2Depth1Adj = new double[14];
        public double[] convLayer4Kernel2Depth2Adj = new double[14];
        public double[] convLayer4Kernel3Depth1Adj = new double[14];
        public double[] convLayer4Kernel3Depth2Adj = new double[14];
        public double[] convLayer4Kernel4Depth1Adj = new double[14];
        public double[] convLayer4Kernel4Depth2Adj = new double[14];
        public double[] convLayer4Kernel5Depth1Adj = new double[14];
        public double[] convLayer4Kernel5Depth2Adj = new double[14];
        public double[] convLayer4Kernel6Depth1Adj = new double[14];
        public double[] convLayer4Kernel6Depth2Adj = new double[14];
        public double[] convLayer4Kernel7Depth1Adj = new double[14];
        public double[] convLayer4Kernel7Depth2Adj = new double[14];
        public double[] convLayer4Kernel8Depth1Adj = new double[14];
        public double[] convLayer4Kernel8Depth2Adj = new double[14];
        public double[] convLayer4Kernel9Depth1Adj = new double[14];
        public double[] convLayer4Kernel9Depth2Adj = new double[14];
        public double[] convLayer4Kernel10Depth1Adj = new double[14];
        public double[] convLayer4Kernel10Depth2Adj = new double[14];
        public double[] convLayer4Kernel11Depth1Adj = new double[14];
        public double[] convLayer4Kernel11Depth2Adj = new double[14];
        public double[] convLayer4Kernel12Depth1Adj = new double[14];
        public double[] convLayer4Kernel12Depth2Adj = new double[14];
        public double[] convLayer4Kernel13Depth1Adj = new double[14];
        public double[] convLayer4Kernel13Depth2Adj = new double[14];
        public double[] convLayer4Kernel14Depth1Adj = new double[14];
        public double[] convLayer4Kernel14Depth2Adj = new double[14];

        public double[] convLayer3BiasesAdj = new double[1400];
        public double[] convLayer3PReLUParamAdj = new double[1400];
        public double[] convLayer3Kernel1Depth1Adj = new double[14];
        public double[] convLayer3Kernel1Depth2Adj = new double[14];
        public double[] convLayer3Kernel2Depth1Adj = new double[14];
        public double[] convLayer3Kernel2Depth2Adj = new double[14];
        public double[] convLayer3Kernel3Depth1Adj = new double[14];
        public double[] convLayer3Kernel3Depth2Adj = new double[14];
        public double[] convLayer3Kernel4Depth1Adj = new double[14];
        public double[] convLayer3Kernel4Depth2Adj = new double[14];
        public double[] convLayer3Kernel5Depth1Adj = new double[14];
        public double[] convLayer3Kernel5Depth2Adj = new double[14];
        public double[] convLayer3Kernel6Depth1Adj = new double[14];
        public double[] convLayer3Kernel6Depth2Adj = new double[14];
        public double[] convLayer3Kernel7Depth1Adj = new double[14];
        public double[] convLayer3Kernel7Depth2Adj = new double[14];
        public double[] convLayer3Kernel8Depth1Adj = new double[14];
        public double[] convLayer3Kernel8Depth2Adj = new double[14];
        public double[] convLayer3Kernel9Depth1Adj = new double[14];
        public double[] convLayer3Kernel9Depth2Adj = new double[14];
        public double[] convLayer3Kernel10Depth1Adj = new double[14];
        public double[] convLayer3Kernel10Depth2Adj = new double[14];
        public double[] convLayer3Kernel11Depth1Adj = new double[14];
        public double[] convLayer3Kernel11Depth2Adj = new double[14];
        public double[] convLayer3Kernel12Depth1Adj = new double[14];
        public double[] convLayer3Kernel12Depth2Adj = new double[14];
        public double[] convLayer3Kernel13Depth1Adj = new double[14];
        public double[] convLayer3Kernel13Depth2Adj = new double[14];
        public double[] convLayer3Kernel14Depth1Adj = new double[14];
        public double[] convLayer3Kernel14Depth2Adj = new double[14];

        public double[] convLayer2BiasesAdj = new double[1400];
        public double[] convLayer2PReLUParamAdj = new double[1400];
        public double[] convLayer2Kernel1Depth1Adj = new double[14];
        public double[] convLayer2Kernel1Depth2Adj = new double[14];
        public double[] convLayer2Kernel2Depth1Adj = new double[14];
        public double[] convLayer2Kernel2Depth2Adj = new double[14];
        public double[] convLayer2Kernel3Depth1Adj = new double[14];
        public double[] convLayer2Kernel3Depth2Adj = new double[14];
        public double[] convLayer2Kernel4Depth1Adj = new double[14];
        public double[] convLayer2Kernel4Depth2Adj = new double[14];
        public double[] convLayer2Kernel5Depth1Adj = new double[14];
        public double[] convLayer2Kernel5Depth2Adj = new double[14];
        public double[] convLayer2Kernel6Depth1Adj = new double[14];
        public double[] convLayer2Kernel6Depth2Adj = new double[14];
        public double[] convLayer2Kernel7Depth1Adj = new double[14];
        public double[] convLayer2Kernel7Depth2Adj = new double[14];
        public double[] convLayer2Kernel8Depth1Adj = new double[14];
        public double[] convLayer2Kernel8Depth2Adj = new double[14];
        public double[] convLayer2Kernel9Depth1Adj = new double[14];
        public double[] convLayer2Kernel9Depth2Adj = new double[14];
        public double[] convLayer2Kernel10Depth1Adj = new double[14];
        public double[] convLayer2Kernel10Depth2Adj = new double[14];
        public double[] convLayer2Kernel11Depth1Adj = new double[14];
        public double[] convLayer2Kernel11Depth2Adj = new double[14];
        public double[] convLayer2Kernel12Depth1Adj = new double[14];
        public double[] convLayer2Kernel12Depth2Adj = new double[14];
        public double[] convLayer2Kernel13Depth1Adj = new double[14];
        public double[] convLayer2Kernel13Depth2Adj = new double[14];
        public double[] convLayer2Kernel14Depth1Adj = new double[14];
        public double[] convLayer2Kernel14Depth2Adj = new double[14];

        public double[] convLayer1BiasesAdj = new double[1400];
        public double[] convLayer1PReLUParamAdj = new double[1400];
        public double[] convLayer1Kernel1Depth1Adj = new double[32];
        public double[] convLayer1Kernel1Depth2Adj = new double[32];
        public double[] convLayer1Kernel1Depth3Adj = new double[32];
        public double[] convLayer1Kernel1Depth4Adj = new double[32];
        public double[] convLayer1Kernel2Depth1Adj = new double[32];
        public double[] convLayer1Kernel2Depth2Adj = new double[32];
        public double[] convLayer1Kernel2Depth3Adj = new double[32];
        public double[] convLayer1Kernel2Depth4Adj = new double[32];
        public double[] convLayer1Kernel3Depth1Adj = new double[32];
        public double[] convLayer1Kernel3Depth2Adj = new double[32];
        public double[] convLayer1Kernel3Depth3Adj = new double[32];
        public double[] convLayer1Kernel3Depth4Adj = new double[32];
        public double[] convLayer1Kernel4Depth1Adj = new double[32];
        public double[] convLayer1Kernel4Depth2Adj = new double[32];
        public double[] convLayer1Kernel4Depth3Adj = new double[32];
        public double[] convLayer1Kernel4Depth4Adj = new double[32];
        public double[] convLayer1Kernel5Depth1Adj = new double[32];
        public double[] convLayer1Kernel5Depth2Adj = new double[32];
        public double[] convLayer1Kernel5Depth3Adj = new double[32];
        public double[] convLayer1Kernel5Depth4Adj = new double[32];
        public double[] convLayer1Kernel6Depth1Adj = new double[32];
        public double[] convLayer1Kernel6Depth2Adj = new double[32];
        public double[] convLayer1Kernel6Depth3Adj = new double[32];
        public double[] convLayer1Kernel6Depth4Adj = new double[32];
        public double[] convLayer1Kernel7Depth1Adj = new double[32];
        public double[] convLayer1Kernel7Depth2Adj = new double[32];
        public double[] convLayer1Kernel7Depth3Adj = new double[32];
        public double[] convLayer1Kernel7Depth4Adj = new double[32];
        public double[] convLayer1Kernel8Depth1Adj = new double[32];
        public double[] convLayer1Kernel8Depth2Adj = new double[32];
        public double[] convLayer1Kernel8Depth3Adj = new double[32];
        public double[] convLayer1Kernel8Depth4Adj = new double[32];
        public double[] convLayer1Kernel9Depth1Adj = new double[32];
        public double[] convLayer1Kernel9Depth2Adj = new double[32];
        public double[] convLayer1Kernel9Depth3Adj = new double[32];
        public double[] convLayer1Kernel9Depth4Adj = new double[32];
        public double[] convLayer1Kernel10Depth1Adj = new double[32];
        public double[] convLayer1Kernel10Depth2Adj = new double[32];
        public double[] convLayer1Kernel10Depth3Adj = new double[32];
        public double[] convLayer1Kernel10Depth4Adj = new double[32];
        public double[] convLayer1Kernel11Depth1Adj = new double[32];
        public double[] convLayer1Kernel11Depth2Adj = new double[32];
        public double[] convLayer1Kernel11Depth3Adj = new double[32];
        public double[] convLayer1Kernel11Depth4Adj = new double[32];
        public double[] convLayer1Kernel12Depth1Adj = new double[32];
        public double[] convLayer1Kernel12Depth2Adj = new double[32];
        public double[] convLayer1Kernel12Depth3Adj = new double[32];
        public double[] convLayer1Kernel12Depth4Adj = new double[32];
        public double[] convLayer1Kernel13Depth1Adj = new double[32];
        public double[] convLayer1Kernel13Depth2Adj = new double[32];
        public double[] convLayer1Kernel13Depth3Adj = new double[32];
        public double[] convLayer1Kernel13Depth4Adj = new double[32];
        public double[] convLayer1Kernel14Depth1Adj = new double[32];
        public double[] convLayer1Kernel14Depth2Adj = new double[32];
        public double[] convLayer1Kernel14Depth3Adj = new double[32];
        public double[] convLayer1Kernel14Depth4Adj = new double[32];
    }
}
