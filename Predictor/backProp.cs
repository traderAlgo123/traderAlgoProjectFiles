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
    public class backProp
    {
        public matrixOps matOps = new matrixOps();
        public backPropFunctions funcs = new backPropFunctions();

        public static int miniBatchSize = 32;
        public static double midPointCompareSmoothed1 = 0.0;
        public static double midPointCompareSmoothed2 = 0.0;
        public static double midPointCompare1 = 0.0;
        public static double midPointCompare2 = 0.0;

        public static double previousMidPoint = 0.0;
        public static double currentMidPoint = 0.0;

        public static double pricePercentChange = 0.0;

        public static double cross_entropy_loss = new double();
        public static double[] cross_entropy_per_neuron = new double[3];
        public static double[] actualOutcomes = new double[3];
        public static double[] cross_entropy_losses_mini_batch = new double[32];

        public static double[] dE_dZ_WRTThirdLayerInputs = new double[3];
        public static double[] error_of_second_layer = new double[3];
        public static double[] error_of_first_layer = new double[64];

        public static double[] ones = new double[1400];
        public static double[] onesTrans = new double[1500];

        public static double singleDivar = new double();
        public static double[] dxhatTrans = new double[1500];
        public static double singleDsqrtvar = new double();
        public static double singleDvar = new double();
        public static double[] dx1Trans = new double[1500];
        public static double[] dx2Trans = new double[1500];
        public static double[] dsqTrans = new double[1500];
        public static double dmuTrans = new double();
        public static double[] dxmuTrans = new double[1500];
        public static double[] dxmu2Trans = new double[1500];

        public static double[] divar = new double[14];
        public static double[] dxhat = new double[1400];
        public static double[] dxmu = new double[1400];
        public static double[] dxmu2 = new double[1400];
        public static double[] dsqrtvar = new double[14];
        public static double[] dvar = new double[14];
        public static double[] dsq = new double[1400];
        public static double[] dx1 = new double[1400];
        public static double[] dx2 = new double[1400];
        public static double[] dmu = new double[14];
        public static double[] error_of_affine_outBlock2 = new double[1500];
        public static double[] error_of_affine_intermediate_Block2 = new double[6000];
        public static double[] error_of_affine_out2Pass2Block1 = new double[1500];
        public static double[] error_of_affine_out2Pass1Block1 = new double[1500];

        public static double[] error_of_affine_out1Pass2Block2 = new double[1500];
        public static double[] error_of_affine_out1Pass1Block2 = new double[6000];
        public static double[] error_of_affine_out1Pass2Block1 = new double[1500];
        public static double[] error_of_output_attention_block = new double[1500];
        public static double[] error_of_concat_filteredVal_output = new double[1500];

        public static double[] error_of_input_from_conv_module1 = new double[1500];
        public static double[] error_of_input_from_conv_module2 = new double[1500];
        public static double[] error_of_input_from_conv_module3 = new double[1500];
        public static double[] error_of_input_from_conv_module4 = new double[1500];
        public static double[] error_of_input_from_conv_module5 = new double[1500];
        public static double[] error_of_input_from_conv_module6 = new double[1500];
        public static double[] error_of_input_from_conv_module7 = new double[1500];
        public static double[] error_of_input_from_conv_module8 = new double[1500];
        public static double[] error_of_input_from_conv_module9 = new double[1500];
        public static double[] avg_error_of_input_from_conv_module = new double[1500];
        public static double[] detempencoded_avg_error_mat = new double[1400];

        public static double[] derivativeOfSecondLayerOut = new double[3];
        public static double[] derivativeOfFirstLayerOut = new double[64];

        public static double[] derivativeOfAffine2Output = new double[1500];
        public static double[] derivativeOfAffine1Output = new double[6000];

        public static double[] derivativeOfConvLayerOut = new double[1400];
        public static double[] error_of_convLayer4 = new double[1400];
        public static double[] error_of_convLayer3 = new double[1400];
        public static double[] error_of_convLayer2 = new double[1400];
        public static double[] error_of_convLayer1 = new double[1400];

        public static double[] derivativeOfAttentionBlockOutput = new double[1500];

        public static double[] thirdLayerOutCpy = new double[3];
        public static double[] secondLayerOutCpy = new double[3];
        public static double[] firstLayerOutCpy = new double[64];
        public static double[] firstLayerInCpy = new double[1500];

        public static ArrayList listOfTrainingExamples = new ArrayList();

        //hyperparameters
        public static double pricePercentChangeCmp = 0.001; //0.002 is the original value used
        public static double initial_learning_rate = 0.0001; //0.0001 is the learning rate chosen by the paper's version 
        public static double hiddenDropOutRate = 0.0; //0.1 is the paper's chosen value
        public static double adam_beta_1 = 0.9;
        public static double adam_beta_2 = 0.999;
        public static double layerNormEpsilon = 0.00001;
        public static double epsilon = 0.00000001;
        public static double L2RegLambda = 0.00; //0.01 is the paper's chosen value
        //end hyperparameters

        public static double[] mlpThirdLayer_m_vec = new double[9];
        public static double[] mlpThirdLayer_v_vec = new double[9];
        public static double[] mlpThirdLayer_m_hat_vec = new double[9];
        public static double[] prev_mlpThirdLayer_m_vec = new double[9];
        public static double[] prev_mlpThirdLayer_v_vec = new double[9];
        public static double[] mlpThirdLayer_adapted_rate = new double[9];

        public static double[] mlpSecondLayer_m_vec = new double[192];
        public static double[] mlpSecondLayer_v_vec = new double[192];
        public static double[] mlpSecondLayer_m_hat_vec = new double[192];
        public static double[] prev_mlpSecondLayer_m_vec = new double[192];
        public static double[] prev_mlpSecondLayer_v_vec = new double[192];
        public static double[] mlpSecondLayer_adapted_rate = new double[192];

        public static double[] mlpSecondLayerBias_m_vec = new double[3];
        public static double[] mlpSecondLayerBias_v_vec = new double[3];
        public static double[] mlpSecondLayerBias_m_hat_vec = new double[3];
        public static double[] prev_mlpSecondLayerBias_m_vec = new double[3];
        public static double[] prev_mlpSecondLayerBias_v_vec = new double[3];
        public static double[] mlpSecondLayerBias_adapted_rate = new double[3];

        public static double[] mlpFirstLayer_m_vec = new double[96000];
        public static double[] mlpFirstLayer_v_vec = new double[96000];
        public static double[] mlpFirstLayer_m_hat_vec = new double[96000];
        public static double[] prev_mlpFirstLayer_m_vec = new double[96000];
        public static double[] prev_mlpFirstLayer_v_vec = new double[96000];
        public static double[] mlpFirstLayer_adapted_rate = new double[96000];

        public static double[] mlpFirstLayerBias_m_vec = new double[64];
        public static double[] mlpFirstLayerBias_v_vec = new double[64];
        public static double[] mlpFirstLayerBias_m_hat_vec = new double[64];
        public static double[] prev_mlpFirstLayerBias_m_vec = new double[64];
        public static double[] prev_mlpFirstLayerBias_v_vec = new double[64];
        public static double[] mlpFirstLayerBias_adapted_rate = new double[64];

        public static double[] thirdLayerWeightsTransposed = new double[9];
        public static double[] secondLayerWeightsTransposed = new double[192];
        public static double[] firstLayerWeightsTransposed = new double[96000];

        public static double[] deltaThirdWeights = new double[9];
        public static double[] deltaSecondWeights = new double[192];
        public static double[] deltaFirstWeights = new double[96000];

        public static double[] secondLayerOutTransposed = new double[3];
        public static double[] firstLayerOutTransposed = new double[64];
        public static double[] firstLayerInputTransposed = new double[1500];

        public static double[] affineIntermediateRes2Transposed = new double[6000];
        public static double[] affineIntermediateRes1Transposed = new double[6000];
        public static double[] affineTrans2WeightsTransposed = new double[900];
        public static double[] affineTrans1WeightsTransposed = new double[900];
        public static double[] residualConnectionOutputNorm1Transposed = new double[1500];
        public static double[] residualConnectionOutputNorm2Transposed = new double[1500];
        public static double[] inputFromConvModuleTransposed = new double[1500];

        public static double[] finalLinearLayerWeightsTransposed = new double[225];
        public static double[] deconcatMat1 = new double[500];
        public static double[] deconcatMat2 = new double[500];
        public static double[] deconcatMat3 = new double[500];

        public static double[] queryLinearLayerWeightsHead1Trans = new double[75];
        public static double[] keyLinearLayerWeightsHead1Trans = new double[75];
        public static double[] valueLinearLayerWeightsHead1Trans = new double[75];

        public static double[] queryLinearLayerWeightsHead2Trans = new double[75];
        public static double[] keyLinearLayerWeightsHead2Trans = new double[75];
        public static double[] valueLinearLayerWeightsHead2Trans = new double[75];

        public static double[] queryLinearLayerWeightsHead3Trans = new double[75];
        public static double[] keyLinearLayerWeightsHead3Trans = new double[75];
        public static double[] valueLinearLayerWeightsHead3Trans = new double[75];

        public static double[] attention_filter_head1_trans = new double[10000];
        public static double[] attention_filter_head2_trans = new double[10000];
        public static double[] attention_filter_head3_trans = new double[10000];

        public static double[] attention_filter_head1_der = new double[10000];
        public static double[] attention_filter_head2_der = new double[10000];
        public static double[] attention_filter_head3_der = new double[10000];

        public static double[] scores_query_gradient_head1 = new double[10000];
        public static double[] scores_key_gradient_head1 = new double[10000];

        public static double[] scores_query_gradient_head2 = new double[10000];
        public static double[] scores_key_gradient_head2 = new double[10000];

        public static double[] scores_query_gradient_head3 = new double[10000];
        public static double[] scores_key_gradient_head3 = new double[10000];

        public static double[] queryHead1Err = new double[500];
        public static double[] queryHead2Err = new double[500];
        public static double[] queryHead3Err = new double[500];

        public static double[] keyHead1Err = new double[500];
        public static double[] keyHead2Err = new double[500];
        public static double[] keyHead3Err = new double[500];

        public static double[] valueHead1Err = new double[500];
        public static double[] valueHead2Err = new double[500];
        public static double[] valueHead3Err = new double[500];

        public static double[] deltaFirstBias = new double[64];
        public static double[] deltaSecondBias = new double[3];
        public static double[] deltaFirstPReLU = new double[64];
        public static double[] deltaSecondPReLU = new double[3];

        public static double[] deltaAffineWeights4 = new double[900];
        public static double[] deltaAffineWeights3 = new double[900];
        public static double[] deltaAffineWeights2 = new double[900];
        public static double[] deltaAffineWeights1 = new double[900];

        public static double[] deltaAffineMLPBias2 = new double[1500];

        public static double[] affineMLPWeights4_m_vec = new double[900];
        public static double[] affineMLPWeights4_v_vec = new double[900];
        public static double[] affineMLPWeights4_m_hat_vec = new double[900];
        public static double[] prev_affineMLPWeights4_m_vec = new double[900];
        public static double[] prev_affineMLPWeights4_v_vec = new double[900];
        public static double[] affineMLPWeights4_adapted_rate = new double[900];

        public static double[] affineMLPWeights4Block2_m_vec = new double[900];
        public static double[] affineMLPWeights4Block2_v_vec = new double[900];
        public static double[] affineMLPWeights4Block2_m_hat_vec = new double[900];
        public static double[] prev_affineMLPWeights4Block2_m_vec = new double[900];
        public static double[] prev_affineMLPWeights4Block2_v_vec = new double[900];
        public static double[] affineMLPWeights4Block2_adapted_rate = new double[900];

        public static double[] affineMLPWeights3_m_vec = new double[900];
        public static double[] affineMLPWeights3_v_vec = new double[900];
        public static double[] affineMLPWeights3_m_hat_vec = new double[900];
        public static double[] prev_affineMLPWeights3_m_vec = new double[900];
        public static double[] prev_affineMLPWeights3_v_vec = new double[900];
        public static double[] affineMLPWeights3_adapted_rate = new double[900];

        public static double[] affineMLPWeights3Block2_m_vec = new double[900];
        public static double[] affineMLPWeights3Block2_v_vec = new double[900];
        public static double[] affineMLPWeights3Block2_m_hat_vec = new double[900];
        public static double[] prev_affineMLPWeights3Block2_m_vec = new double[900];
        public static double[] prev_affineMLPWeights3Block2_v_vec = new double[900];
        public static double[] affineMLPWeights3Block2_adapted_rate = new double[900];

        public static double[] affineMLPWeights2_m_vec = new double[900];
        public static double[] affineMLPWeights2_v_vec = new double[900];
        public static double[] affineMLPWeights2_m_hat_vec = new double[900];
        public static double[] prev_affineMLPWeights2_m_vec = new double[900];
        public static double[] prev_affineMLPWeights2_v_vec = new double[900];
        public static double[] affineMLPWeights2_adapted_rate = new double[900];

        public static double[] affineMLPWeights2Block2_m_vec = new double[900];
        public static double[] affineMLPWeights2Block2_v_vec = new double[900];
        public static double[] affineMLPWeights2Block2_m_hat_vec = new double[900];
        public static double[] prev_affineMLPWeights2Block2_m_vec = new double[900];
        public static double[] prev_affineMLPWeights2Block2_v_vec = new double[900];
        public static double[] affineMLPWeights2Block2_adapted_rate = new double[900];

        public static double[] affineMLPWeights1_m_vec = new double[900];
        public static double[] affineMLPWeights1_v_vec = new double[900];
        public static double[] affineMLPWeights1_m_hat_vec = new double[900];
        public static double[] prev_affineMLPWeights1_m_vec = new double[900];
        public static double[] prev_affineMLPWeights1_v_vec = new double[900];
        public static double[] affineMLPWeights1_adapted_rate = new double[900];

        public static double[] affineMLPWeights1Block2_m_vec = new double[900];
        public static double[] affineMLPWeights1Block2_v_vec = new double[900];
        public static double[] affineMLPWeights1Block2_m_hat_vec = new double[900];
        public static double[] prev_affineMLPWeights1Block2_m_vec = new double[900];
        public static double[] prev_affineMLPWeights1Block2_v_vec = new double[900];
        public static double[] affineMLPWeights1Block2_adapted_rate = new double[900];

        public static double[] deltaAffineMLPBias1 = new double[6000];
        public static double[] deltaAffineMLPPreluParam1 = new double[6000];

        public static double[] affineMLPBias2_m_vec = new double[1500];
        public static double[] affineMLPBias2_v_vec = new double[1500];
        public static double[] affineMLPBias2_m_hat_vec = new double[1500];
        public static double[] prev_affineMLPBias2_m_vec = new double[1500];
        public static double[] prev_affineMLPBias2_v_vec = new double[1500];
        public static double[] affineMLPBias2_adapted_rate = new double[1500];

        public static double[] affineMLPBias2Block2_m_vec = new double[6000];
        public static double[] affineMLPBias2Block2_v_vec = new double[6000];
        public static double[] affineMLPBias2Block2_m_hat_vec = new double[6000];
        public static double[] prev_affineMLPBias2Block2_m_vec = new double[6000];
        public static double[] prev_affineMLPBias2Block2_v_vec = new double[6000];
        public static double[] affineMLPBias2Block2_adapted_rate = new double[6000];

        public static double[] affineMLPBias1_m_vec = new double[1500];
        public static double[] affineMLPBias1_v_vec = new double[1500];
        public static double[] affineMLPBias1_m_hat_vec = new double[1500];
        public static double[] prev_affineMLPBias1_m_vec = new double[1500];
        public static double[] prev_affineMLPBias1_v_vec = new double[1500];
        public static double[] affineMLPBias1_adapted_rate = new double[1500];

        public static double[] affineMLPBeta2_m_vec = new double[1500];
        public static double[] affineMLPBeta2_v_vec = new double[1500];
        public static double[] affineMLPBeta2_m_hat_vec = new double[1500];
        public static double[] prev_affineMLPBeta2_m_vec = new double[1500];
        public static double[] prev_affineMLPBeta2_v_vec = new double[1500];
        public static double[] affineMLPBeta2_adapted_rate = new double[1500];

        public static double[] affineMLPGamma2_m_vec = new double[1500];
        public static double[] affineMLPGamma2_v_vec = new double[1500];
        public static double[] affineMLPGamma2_m_hat_vec = new double[1500];
        public static double[] prev_affineMLPGamma2_m_vec = new double[1500];
        public static double[] prev_affineMLPGamma2_v_vec = new double[1500];
        public static double[] affineMLPGamma2_adapted_rate = new double[1500];

        public static double[] affineMLPGamma1_m_vec = new double[1500];
        public static double[] affineMLPGamma1_v_vec = new double[1500];
        public static double[] affineMLPGamma1_m_hat_vec = new double[1500];
        public static double[] prev_affineMLPGamma1_m_vec = new double[1500];
        public static double[] prev_affineMLPGamma1_v_vec = new double[1500];
        public static double[] affineMLPGamma1_adapted_rate = new double[1500];

        public static double[] affineMLPBeta1_m_vec = new double[1500];
        public static double[] affineMLPBeta1_v_vec = new double[1500];
        public static double[] affineMLPBeta1_m_hat_vec = new double[1500];
        public static double[] prev_affineMLPBeta1_m_vec = new double[1500];
        public static double[] prev_affineMLPBeta1_v_vec = new double[1500];
        public static double[] affineMLPBeta1_adapted_rate = new double[1500];

        public static double[] deltaAddAndNormGamma2 = new double[1500];
        public static double[] deltaAddAndNormBeta2 = new double[1500];
        public static double[] deltaAddAndNormGamma1 = new double[1500];
        public static double[] deltaAddAndNormBeta1 = new double[1500];

        public static double[] deltaFinalLinearLayerWeights = new double[225];

        public static double[] finalLinearLayerBlock2_m_vec = new double[225];
        public static double[] finalLinearLayerBlock2_v_vec = new double[225];
        public static double[] finalLinearLayerBlock2_m_hat_vec = new double[225];
        public static double[] prev_finalLinearLayerBlock2_m_vec = new double[225];
        public static double[] prev_finalLinearLayerBlock2_v_vec = new double[225];
        public static double[] finalLinearLayerBlock2_adapted_rate = new double[225];

        public static double[] finalLinearLayer_m_vec = new double[225];
        public static double[] finalLinearLayer_v_vec = new double[225];
        public static double[] finalLinearLayer_m_hat_vec = new double[225];
        public static double[] prev_finalLinearLayer_m_vec = new double[225];
        public static double[] prev_finalLinearLayer_v_vec = new double[225];
        public static double[] finalLinearLayer_adapted_rate = new double[225];

        public static double[] deltaQueryWeightsHead1 = new double[75];
        public static double[] deltaKeyWeightsHead1 = new double[75];
        public static double[] deltaValueWeightsHead1 = new double[75];

        public static double[] queryHead1Block2_m_vec = new double[75];
        public static double[] queryHead1Block2_v_vec = new double[75];
        public static double[] queryHead1Block2_m_hat_vec = new double[75];
        public static double[] prev_queryHead1Block2_m_vec = new double[75];
        public static double[] prev_queryHead1Block2_v_vec = new double[75];
        public static double[] queryHead1Block2_adapted_rate = new double[75];

        public static double[] queryHead1_m_vec = new double[75];
        public static double[] queryHead1_v_vec = new double[75];
        public static double[] queryHead1_m_hat_vec = new double[75];
        public static double[] prev_queryHead1_m_vec = new double[75];
        public static double[] prev_queryHead1_v_vec = new double[75];
        public static double[] queryHead1_adapted_rate = new double[75];

        public static double[] queryHead2Block2_m_vec = new double[75];
        public static double[] queryHead2Block2_v_vec = new double[75];
        public static double[] queryHead2Block2_m_hat_vec = new double[75];
        public static double[] prev_queryHead2Block2_m_vec = new double[75];
        public static double[] prev_queryHead2Block2_v_vec = new double[75];
        public static double[] queryHead2Block2_adapted_rate = new double[75];

        public static double[] queryHead2_m_vec = new double[75];
        public static double[] queryHead2_v_vec = new double[75];
        public static double[] queryHead2_m_hat_vec = new double[75];
        public static double[] prev_queryHead2_m_vec = new double[75];
        public static double[] prev_queryHead2_v_vec = new double[75];
        public static double[] queryHead2_adapted_rate = new double[75];

        public static double[] queryHead3Block2_m_vec = new double[75];
        public static double[] queryHead3Block2_v_vec = new double[75];
        public static double[] queryHead3Block2_m_hat_vec = new double[75];
        public static double[] prev_queryHead3Block2_m_vec = new double[75];
        public static double[] prev_queryHead3Block2_v_vec = new double[75];
        public static double[] queryHead3Block2_adapted_rate = new double[75];

        public static double[] queryHead3_m_vec = new double[75];
        public static double[] queryHead3_v_vec = new double[75];
        public static double[] queryHead3_m_hat_vec = new double[75];
        public static double[] prev_queryHead3_m_vec = new double[75];
        public static double[] prev_queryHead3_v_vec = new double[75];
        public static double[] queryHead3_adapted_rate = new double[75];

        public static double[] keyHead1Block2_m_vec = new double[75];
        public static double[] keyHead1Block2_v_vec = new double[75];
        public static double[] keyHead1Block2_m_hat_vec = new double[75];
        public static double[] prev_keyHead1Block2_m_vec = new double[75];
        public static double[] prev_keyHead1Block2_v_vec = new double[75];
        public static double[] keyHead1Block2_adapted_rate = new double[75];

        public static double[] keyHead1_m_vec = new double[75];
        public static double[] keyHead1_v_vec = new double[75];
        public static double[] keyHead1_m_hat_vec = new double[75];
        public static double[] prev_keyHead1_m_vec = new double[75];
        public static double[] prev_keyHead1_v_vec = new double[75];
        public static double[] keyHead1_adapted_rate = new double[75];

        public static double[] keyHead2Block2_m_vec = new double[75];
        public static double[] keyHead2Block2_v_vec = new double[75];
        public static double[] keyHead2Block2_m_hat_vec = new double[75];
        public static double[] prev_keyHead2Block2_m_vec = new double[75];
        public static double[] prev_keyHead2Block2_v_vec = new double[75];
        public static double[] keyHead2Block2_adapted_rate = new double[75];

        public static double[] keyHead2_m_vec = new double[75];
        public static double[] keyHead2_v_vec = new double[75];
        public static double[] keyHead2_m_hat_vec = new double[75];
        public static double[] prev_keyHead2_m_vec = new double[75];
        public static double[] prev_keyHead2_v_vec = new double[75];
        public static double[] keyHead2_adapted_rate = new double[75];

        public static double[] keyHead3Block2_m_vec = new double[75];
        public static double[] keyHead3Block2_v_vec = new double[75];
        public static double[] keyHead3Block2_m_hat_vec = new double[75];
        public static double[] prev_keyHead3Block2_m_vec = new double[75];
        public static double[] prev_keyHead3Block2_v_vec = new double[75];
        public static double[] keyHead3Block2_adapted_rate = new double[75];

        public static double[] keyHead3_m_vec = new double[75];
        public static double[] keyHead3_v_vec = new double[75];
        public static double[] keyHead3_m_hat_vec = new double[75];
        public static double[] prev_keyHead3_m_vec = new double[75];
        public static double[] prev_keyHead3_v_vec = new double[75];
        public static double[] keyHead3_adapted_rate = new double[75];

        public static double[] valueHead1Block2_m_vec = new double[75];
        public static double[] valueHead1Block2_v_vec = new double[75];
        public static double[] valueHead1Block2_m_hat_vec = new double[75];
        public static double[] prev_valueHead1Block2_m_vec = new double[75];
        public static double[] prev_valueHead1Block2_v_vec = new double[75];
        public static double[] valueHead1Block2_adapted_rate = new double[75];

        public static double[] valueHead1_m_vec = new double[75];
        public static double[] valueHead1_v_vec = new double[75];
        public static double[] valueHead1_m_hat_vec = new double[75];
        public static double[] prev_valueHead1_m_vec = new double[75];
        public static double[] prev_valueHead1_v_vec = new double[75];
        public static double[] valueHead1_adapted_rate = new double[75];

        public static double[] valueHead2Block2_m_vec = new double[75];
        public static double[] valueHead2Block2_v_vec = new double[75];
        public static double[] valueHead2Block2_m_hat_vec = new double[75];
        public static double[] prev_valueHead2Block2_m_vec = new double[75];
        public static double[] prev_valueHead2Block2_v_vec = new double[75];
        public static double[] valueHead2Block2_adapted_rate = new double[75];

        public static double[] valueHead2_m_vec = new double[75];
        public static double[] valueHead2_v_vec = new double[75];
        public static double[] valueHead2_m_hat_vec = new double[75];
        public static double[] prev_valueHead2_m_vec = new double[75];
        public static double[] prev_valueHead2_v_vec = new double[75];
        public static double[] valueHead2_adapted_rate = new double[75];

        public static double[] valueHead3Block2_m_vec = new double[75];
        public static double[] valueHead3Block2_v_vec = new double[75];
        public static double[] valueHead3Block2_m_hat_vec = new double[75];
        public static double[] prev_valueHead3Block2_m_vec = new double[75];
        public static double[] prev_valueHead3Block2_v_vec = new double[75];
        public static double[] valueHead3Block2_adapted_rate = new double[75];

        public static double[] valueHead3_m_vec = new double[75];
        public static double[] valueHead3_v_vec = new double[75];
        public static double[] valueHead3_m_hat_vec = new double[75];
        public static double[] prev_valueHead3_m_vec = new double[75];
        public static double[] prev_valueHead3_v_vec = new double[75];
        public static double[] valueHead3_adapted_rate = new double[75];

        public static double[] deltaQueryWeightsHead2 = new double[75];
        public static double[] deltaKeyWeightsHead2 = new double[75];
        public static double[] deltaValueWeightsHead2 = new double[75];

        public static double[] deltaQueryWeightsHead3 = new double[75];
        public static double[] deltaKeyWeightsHead3 = new double[75];
        public static double[] deltaValueWeightsHead3 = new double[75];

        public static double[] deltaConvLayer5Kernel1_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel1_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel2_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel2_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel3_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel3_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel4_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel4_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel5_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel5_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel6_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel6_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel7_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel7_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel8_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel8_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel9_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel9_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel10_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel10_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel11_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel11_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel12_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel12_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel13_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel13_depth2 = new double[14];
        public static double[] deltaConvLayer5Kernel14_depth1 = new double[14];
        public static double[] deltaConvLayer5Kernel14_depth2 = new double[14];

        public static double[] deltaConvLayer4Kernel1_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel1_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel2_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel2_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel3_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel3_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel4_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel4_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel5_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel5_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel6_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel6_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel7_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel7_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel8_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel8_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel9_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel9_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel10_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel10_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel11_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel11_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel12_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel12_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel13_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel13_depth2 = new double[14];
        public static double[] deltaConvLayer4Kernel14_depth1 = new double[14];
        public static double[] deltaConvLayer4Kernel14_depth2 = new double[14];

        public static double[] deltaConvLayer3Kernel1_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel1_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel2_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel2_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel3_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel3_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel4_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel4_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel5_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel5_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel6_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel6_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel7_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel7_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel8_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel8_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel9_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel9_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel10_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel10_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel11_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel11_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel12_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel12_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel13_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel13_depth2 = new double[14];
        public static double[] deltaConvLayer3Kernel14_depth1 = new double[14];
        public static double[] deltaConvLayer3Kernel14_depth2 = new double[14];

        public static double[] deltaConvLayer2Kernel1_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel1_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel2_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel2_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel3_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel3_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel4_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel4_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel5_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel5_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel6_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel6_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel7_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel7_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel8_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel8_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel9_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel9_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel10_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel10_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel11_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel11_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel12_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel12_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel13_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel13_depth2 = new double[14];
        public static double[] deltaConvLayer2Kernel14_depth1 = new double[14];
        public static double[] deltaConvLayer2Kernel14_depth2 = new double[14];

        public static double[] deltaConvLayer1Kernel1_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel1_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel1_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel1_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel2_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel2_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel2_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel2_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel3_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel3_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel3_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel3_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel4_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel4_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel4_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel4_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel5_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel5_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel5_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel5_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel6_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel6_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel6_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel6_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel7_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel7_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel7_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel7_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel8_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel8_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel8_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel8_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel9_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel9_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel9_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel9_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel10_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel10_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel10_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel10_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel11_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel11_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel11_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel11_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel12_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel12_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel12_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel12_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel13_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel13_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel13_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel13_depth4 = new double[32];
        public static double[] deltaConvLayer1Kernel14_depth1 = new double[32];
        public static double[] deltaConvLayer1Kernel14_depth2 = new double[32];
        public static double[] deltaConvLayer1Kernel14_depth3 = new double[32];
        public static double[] deltaConvLayer1Kernel14_depth4 = new double[32];

        public static double[] deltaConvLayer5PReLUParams = new double[1400];
        public static double[] deltaConvLayer5Biases = new double[1400];
        public static double[] deltaConvLayer5NormGamma = new double[1400];
        public static double[] deltaConvLayer5NormBeta = new double[1400];

        public static double[] deltaConvLayer4PReLUParams = new double[1400];
        public static double[] deltaConvLayer4Biases = new double[1400];

        public static double[] deltaConvLayer3PReLUParams = new double[1400];
        public static double[] deltaConvLayer3Biases = new double[1400];

        public static double[] deltaConvLayer2PReLUParams = new double[1400];
        public static double[] deltaConvLayer2Biases = new double[1400];

        public static double[] deltaConvLayer1PReLUParams = new double[1400];
        public static double[] deltaConvLayer1Biases = new double[1400];

        public static double[] convLayer5Bias_m_vec = new double[1400];
        public static double[] convLayer5Bias_v_vec = new double[1400];
        public static double[] convLayer5Bias_m_hat_vec = new double[1400];
        public static double[] prev_convLayer5Bias_m_vec = new double[1400];
        public static double[] prev_convLayer5Bias_v_vec = new double[1400];
        public static double[] convLayer5Bias_adapted_rate = new double[1400];

        public static double[] convLayer5Beta_m_vec = new double[1400];
        public static double[] convLayer5Beta_v_vec = new double[1400];
        public static double[] convLayer5Beta_m_hat_vec = new double[1400];
        public static double[] prev_convLayer5Beta_m_vec = new double[1400];
        public static double[] prev_convLayer5Beta_v_vec = new double[1400];
        public static double[] convLayer5Beta_adapted_rate = new double[1400];

        public static double[] convLayer5Gamma_m_vec = new double[1400];
        public static double[] convLayer5Gamma_v_vec = new double[1400];
        public static double[] convLayer5Gamma_m_hat_vec = new double[1400];
        public static double[] prev_convLayer5Gamma_m_vec = new double[1400];
        public static double[] prev_convLayer5Gamma_v_vec = new double[1400];
        public static double[] convLayer5Gamma_adapted_rate = new double[1400];

        public static double[] convLayer5Kernel1Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel1Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel1Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel1Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel1Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel1Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel1Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel1Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel1Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel1Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel1Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel1Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel2Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel2Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel2Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel2Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel2Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel2Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel2Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel2Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel2Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel2Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel2Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel2Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel3Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel3Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel3Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel3Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel3Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel3Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel3Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel3Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel3Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel3Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel3Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel3Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel4Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel4Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel4Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel4Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel4Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel4Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel4Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel4Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel4Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel4Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel4Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel4Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel5Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel5Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel5Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel5Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel5Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel5Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel5Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel5Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel5Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel5Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel5Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel5Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel6Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel6Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel6Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel6Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel6Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel6Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel6Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel6Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel6Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel6Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel6Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel6Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel7Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel7Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel7Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel7Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel7Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel7Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel7Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel7Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel7Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel7Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel7Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel7Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel8Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel8Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel8Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel8Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel8Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel8Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel8Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel8Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel8Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel8Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel8Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel8Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel9Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel9Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel9Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel9Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel9Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel9Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel9Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel9Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel9Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel9Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel9Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel9Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel10Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel10Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel10Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel10Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel10Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel10Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel10Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel10Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel10Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel10Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel10Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel10Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel11Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel11Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel11Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel11Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel11Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel11Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel11Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel11Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel11Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel11Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel11Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel11Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel12Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel12Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel12Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel12Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel12Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel12Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel12Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel12Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel12Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel12Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel12Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel12Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel13Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel13Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel13Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel13Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel13Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel13Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel13Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel13Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel13Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel13Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel13Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel13Depth2_adapted_rate = new double[14];

        public static double[] convLayer5Kernel14Depth1_m_vec = new double[14];
        public static double[] convLayer5Kernel14Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel14Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel14Depth1_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel14Depth1_v_vec = new double[14];
        public static double[] convLayer5Kernel14Depth1_adapted_rate = new double[14];

        public static double[] convLayer5Kernel14Depth2_m_vec = new double[14];
        public static double[] convLayer5Kernel14Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel14Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer5Kernel14Depth2_m_vec = new double[14];
        public static double[] prev_convLayer5Kernel14Depth2_v_vec = new double[14];
        public static double[] convLayer5Kernel14Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Bias_m_vec = new double[1400];
        public static double[] convLayer4Bias_v_vec = new double[1400];
        public static double[] convLayer4Bias_m_hat_vec = new double[1400];
        public static double[] prev_convLayer4Bias_m_vec = new double[1400];
        public static double[] prev_convLayer4Bias_v_vec = new double[1400];
        public static double[] convLayer4Bias_adapted_rate = new double[1400];

        public static double[] convLayer4Kernel1Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel1Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel1Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel1Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel1Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel1Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel1Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel1Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel1Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel1Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel1Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel1Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel2Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel2Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel2Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel2Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel2Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel2Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel2Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel2Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel2Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel2Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel2Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel2Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel3Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel3Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel3Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel3Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel3Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel3Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel3Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel3Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel3Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel3Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel3Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel3Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel4Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel4Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel4Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel4Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel4Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel4Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel4Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel4Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel4Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel4Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel4Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel4Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel5Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel5Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel5Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel5Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel5Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel5Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel5Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel5Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel5Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel5Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel5Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel5Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel6Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel6Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel6Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel6Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel6Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel6Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel6Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel6Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel6Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel6Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel6Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel6Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel7Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel7Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel7Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel7Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel7Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel7Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel7Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel7Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel7Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel7Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel7Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel7Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel8Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel8Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel8Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel8Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel8Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel8Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel8Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel8Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel8Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel8Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel8Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel8Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel9Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel9Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel9Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel9Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel9Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel9Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel9Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel9Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel9Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel9Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel9Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel9Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel10Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel10Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel10Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel10Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel10Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel10Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel10Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel10Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel10Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel10Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel10Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel10Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel11Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel11Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel11Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel11Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel11Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel11Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel11Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel11Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel11Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel11Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel11Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel11Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel12Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel12Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel12Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel12Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel12Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel12Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel12Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel12Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel12Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel12Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel12Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel12Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel13Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel13Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel13Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel13Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel13Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel13Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel13Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel13Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel13Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel13Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel13Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel13Depth2_adapted_rate = new double[14];

        public static double[] convLayer4Kernel14Depth1_m_vec = new double[14];
        public static double[] convLayer4Kernel14Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel14Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel14Depth1_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel14Depth1_v_vec = new double[14];
        public static double[] convLayer4Kernel14Depth1_adapted_rate = new double[14];

        public static double[] convLayer4Kernel14Depth2_m_vec = new double[14];
        public static double[] convLayer4Kernel14Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel14Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer4Kernel14Depth2_m_vec = new double[14];
        public static double[] prev_convLayer4Kernel14Depth2_v_vec = new double[14];
        public static double[] convLayer4Kernel14Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Bias_m_vec = new double[1400];
        public static double[] convLayer3Bias_v_vec = new double[1400];
        public static double[] convLayer3Bias_m_hat_vec = new double[1400];
        public static double[] prev_convLayer3Bias_m_vec = new double[1400];
        public static double[] prev_convLayer3Bias_v_vec = new double[1400];
        public static double[] convLayer3Bias_adapted_rate = new double[1400];

        public static double[] convLayer3Kernel1Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel1Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel1Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel1Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel1Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel1Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel1Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel1Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel1Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel1Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel1Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel1Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel2Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel2Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel2Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel2Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel2Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel2Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel2Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel2Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel2Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel2Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel2Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel2Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel3Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel3Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel3Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel3Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel3Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel3Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel3Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel3Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel3Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel3Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel3Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel3Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel4Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel4Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel4Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel4Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel4Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel4Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel4Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel4Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel4Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel4Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel4Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel4Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel5Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel5Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel5Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel5Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel5Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel5Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel5Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel5Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel5Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel5Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel5Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel5Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel6Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel6Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel6Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel6Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel6Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel6Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel6Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel6Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel6Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel6Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel6Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel6Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel7Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel7Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel7Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel7Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel7Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel7Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel7Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel7Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel7Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel7Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel7Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel7Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel8Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel8Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel8Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel8Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel8Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel8Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel8Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel8Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel8Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel8Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel8Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel8Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel9Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel9Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel9Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel9Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel9Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel9Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel9Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel9Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel9Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel9Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel9Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel9Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel10Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel10Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel10Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel10Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel10Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel10Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel10Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel10Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel10Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel10Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel10Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel10Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel11Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel11Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel11Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel11Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel11Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel11Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel11Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel11Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel11Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel11Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel11Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel11Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel12Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel12Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel12Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel12Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel12Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel12Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel12Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel12Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel12Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel12Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel12Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel12Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel13Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel13Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel13Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel13Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel13Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel13Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel13Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel13Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel13Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel13Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel13Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel13Depth2_adapted_rate = new double[14];

        public static double[] convLayer3Kernel14Depth1_m_vec = new double[14];
        public static double[] convLayer3Kernel14Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel14Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel14Depth1_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel14Depth1_v_vec = new double[14];
        public static double[] convLayer3Kernel14Depth1_adapted_rate = new double[14];

        public static double[] convLayer3Kernel14Depth2_m_vec = new double[14];
        public static double[] convLayer3Kernel14Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel14Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer3Kernel14Depth2_m_vec = new double[14];
        public static double[] prev_convLayer3Kernel14Depth2_v_vec = new double[14];
        public static double[] convLayer3Kernel14Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Bias_m_vec = new double[1400];
        public static double[] convLayer2Bias_v_vec = new double[1400];
        public static double[] convLayer2Bias_m_hat_vec = new double[1400];
        public static double[] prev_convLayer2Bias_m_vec = new double[1400];
        public static double[] prev_convLayer2Bias_v_vec = new double[1400];
        public static double[] convLayer2Bias_adapted_rate = new double[1400];

        public static double[] convLayer2Kernel1Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel1Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel1Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel1Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel1Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel1Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel1Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel1Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel1Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel1Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel1Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel1Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel2Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel2Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel2Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel2Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel2Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel2Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel2Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel2Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel2Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel2Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel2Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel2Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel3Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel3Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel3Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel3Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel3Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel3Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel3Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel3Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel3Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel3Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel3Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel3Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel4Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel4Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel4Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel4Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel4Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel4Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel4Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel4Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel4Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel4Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel4Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel4Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel5Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel5Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel5Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel5Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel5Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel5Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel5Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel5Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel5Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel5Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel5Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel5Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel6Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel6Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel6Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel6Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel6Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel6Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel6Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel6Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel6Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel6Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel6Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel6Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel7Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel7Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel7Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel7Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel7Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel7Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel7Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel7Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel7Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel7Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel7Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel7Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel8Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel8Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel8Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel8Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel8Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel8Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel8Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel8Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel8Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel8Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel8Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel8Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel9Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel9Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel9Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel9Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel9Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel9Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel9Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel9Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel9Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel9Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel9Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel9Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel10Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel10Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel10Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel10Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel10Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel10Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel10Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel10Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel10Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel10Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel10Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel10Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel11Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel11Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel11Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel11Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel11Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel11Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel11Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel11Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel11Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel11Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel11Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel11Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel12Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel12Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel12Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel12Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel12Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel12Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel12Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel12Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel12Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel12Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel12Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel12Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel13Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel13Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel13Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel13Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel13Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel13Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel13Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel13Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel13Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel13Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel13Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel13Depth2_adapted_rate = new double[14];

        public static double[] convLayer2Kernel14Depth1_m_vec = new double[14];
        public static double[] convLayer2Kernel14Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel14Depth1_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel14Depth1_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel14Depth1_v_vec = new double[14];
        public static double[] convLayer2Kernel14Depth1_adapted_rate = new double[14];

        public static double[] convLayer2Kernel14Depth2_m_vec = new double[14];
        public static double[] convLayer2Kernel14Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel14Depth2_m_hat_vec = new double[14];
        public static double[] prev_convLayer2Kernel14Depth2_m_vec = new double[14];
        public static double[] prev_convLayer2Kernel14Depth2_v_vec = new double[14];
        public static double[] convLayer2Kernel14Depth2_adapted_rate = new double[14];

        public static double[] convLayer1Kernel1Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel1Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel1Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel1Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel1Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel1Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel1Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel1Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel1Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel1Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel1Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel1Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel1Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel1Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel1Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel1Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel1Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel1Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel1Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel1Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel1Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel1Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel1Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel1Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel2Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel2Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel2Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel2Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel2Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel2Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel2Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel2Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel2Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel2Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel2Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel2Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel2Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel2Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel2Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel2Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel2Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel2Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel2Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel2Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel2Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel2Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel2Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel2Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel3Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel3Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel3Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel3Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel3Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel3Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel3Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel3Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel3Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel3Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel3Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel3Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel3Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel3Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel3Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel3Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel3Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel3Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel3Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel3Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel3Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel3Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel3Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel3Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel4Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel4Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel4Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel4Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel4Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel4Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel4Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel4Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel4Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel4Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel4Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel4Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel4Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel4Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel4Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel4Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel4Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel4Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel4Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel4Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel4Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel4Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel4Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel4Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel5Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel5Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel5Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel5Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel5Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel5Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel5Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel5Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel5Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel5Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel5Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel5Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel5Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel5Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel5Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel5Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel5Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel5Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel5Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel5Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel5Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel5Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel5Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel5Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel6Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel6Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel6Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel6Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel6Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel6Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel6Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel6Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel6Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel6Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel6Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel6Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel6Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel6Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel6Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel6Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel6Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel6Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel6Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel6Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel6Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel6Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel6Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel6Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel7Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel7Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel7Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel7Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel7Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel7Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel7Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel7Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel7Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel7Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel7Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel7Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel7Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel7Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel7Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel7Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel7Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel7Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel7Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel7Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel7Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel7Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel7Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel7Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel8Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel8Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel8Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel8Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel8Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel8Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel8Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel8Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel8Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel8Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel8Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel8Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel8Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel8Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel8Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel8Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel8Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel8Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel8Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel8Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel8Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel8Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel8Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel8Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel9Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel9Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel9Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel9Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel9Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel9Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel9Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel9Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel9Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel9Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel9Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel9Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel9Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel9Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel9Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel9Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel9Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel9Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel9Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel9Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel9Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel9Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel9Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel9Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel10Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel10Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel10Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel10Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel10Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel10Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel10Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel10Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel10Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel10Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel10Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel10Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel10Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel10Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel10Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel10Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel10Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel10Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel10Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel10Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel10Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel10Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel10Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel10Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel11Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel11Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel11Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel11Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel11Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel11Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel11Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel11Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel11Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel11Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel11Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel11Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel11Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel11Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel11Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel11Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel11Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel11Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel11Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel11Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel11Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel11Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel11Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel11Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel12Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel12Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel12Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel12Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel12Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel12Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel12Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel12Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel12Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel12Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel12Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel12Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel12Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel12Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel12Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel12Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel12Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel12Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel12Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel12Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel12Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel12Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel12Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel12Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel13Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel13Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel13Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel13Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel13Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel13Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel13Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel13Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel13Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel13Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel13Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel13Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel13Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel13Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel13Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel13Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel13Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel13Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel13Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel13Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel13Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel13Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel13Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel13Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Kernel14Depth1_m_vec = new double[32];
        public static double[] convLayer1Kernel14Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel14Depth1_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel14Depth1_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel14Depth1_v_vec = new double[32];
        public static double[] convLayer1Kernel14Depth1_adapted_rate = new double[32];

        public static double[] convLayer1Kernel14Depth2_m_vec = new double[32];
        public static double[] convLayer1Kernel14Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel14Depth2_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel14Depth2_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel14Depth2_v_vec = new double[32];
        public static double[] convLayer1Kernel14Depth2_adapted_rate = new double[32];

        public static double[] convLayer1Kernel14Depth3_m_vec = new double[32];
        public static double[] convLayer1Kernel14Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel14Depth3_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel14Depth3_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel14Depth3_v_vec = new double[32];
        public static double[] convLayer1Kernel14Depth3_adapted_rate = new double[32];

        public static double[] convLayer1Kernel14Depth4_m_vec = new double[32];
        public static double[] convLayer1Kernel14Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel14Depth4_m_hat_vec = new double[32];
        public static double[] prev_convLayer1Kernel14Depth4_m_vec = new double[32];
        public static double[] prev_convLayer1Kernel14Depth4_v_vec = new double[32];
        public static double[] convLayer1Kernel14Depth4_adapted_rate = new double[32];

        public static double[] convLayer1Bias_m_vec = new double[1400];
        public static double[] convLayer1Bias_v_vec = new double[1400];
        public static double[] convLayer1Bias_m_hat_vec = new double[1400];
        public static double[] prev_convLayer1Bias_m_vec = new double[1400];
        public static double[] prev_convLayer1Bias_v_vec = new double[1400];
        public static double[] convLayer1Bias_adapted_rate = new double[1400];

        public void cross_entropy_loss_per_Ex(int example)
        {
            double[] predicted = new double[3];
            double[] expected = new double[3];

            double cross_entropy = 0.0;

            if (predictorGui.predictorGui1.activateTraining.Checked == true)
            {
                Array.Copy(MLP.secondLayerOut, 0, predicted, 0, 3);
                Array.Copy(actualOutcomes, 0, expected, 0, 3);
            }
            else
            {
                Array.Copy(predictorGui.prevPrediction, 0, predicted, 0, 3);
                Array.Copy(actualOutcomes, 0, expected, 0, 3);
            }

            
            for (int i = 0; i < 3; i++)
            {
                /*
                if(expected[i] == 0.0F)
                {
                    expected[i] = (epsilon * 1000000) / (3.0 - 1.0);
                }
                else
                {
                    expected[i] = 1.0 - (epsilon * 1000000);
                }
                */
                cross_entropy += -expected[i] * (Math.Log(predicted[i]));
                cross_entropy_per_neuron[i] = -expected[i] * (Math.Log(predicted[i]));
            }
            

            cross_entropy_loss = cross_entropy;
            cross_entropy_losses_mini_batch[predictorGui.miniBatchIdx] = cross_entropy;

            //initialize ones matrix
            for(int i = 0; i < 1400; i++)
            {
                ones[i] = 1;
            }
            for (int i = 0; i < 1500; i++)
            {
                onesTrans[i] = 1;
            }
        }

        public void affineMLPCalculateAdjustments(int blockNum)
        {
            double[] temp;
            int M = 1500;
            int K = 64;
            int N = 1;

            if (blockNum == 2)
            {
                temp = matOps.transposeMat(MLP.firstLayerWeights, M, K);
                Array.Copy(temp, 0, firstLayerWeightsTransposed, 0, 96000);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\firstLayerWeightsTransposed.txt");
                    for (int i = 0; i < 96000; i++)
                    {
                        output.WriteLine(firstLayerWeightsTransposed[i].ToString());
                    }
                    output.Close();
                }

                temp = matOps.matrixMul(firstLayerWeightsTransposed, error_of_first_layer, M, K, N);
                Array.Copy(temp, 0, error_of_affine_outBlock2, 0, 1500);
            }
            else
            {
                Array.Copy(avg_error_of_input_from_conv_module, 0, error_of_affine_outBlock2, 0, 1500);
                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\avg_error_of_input_from_conv_module_error_of_affine_outBlock2.txt");
                    for (int i = 0; i < 1500; i++)
                    {
                        output.WriteLine(avg_error_of_input_from_conv_module[i].ToString());
                    }
                    output.Close();
                }
            }

            //normalization backprop
            int normIdx;
            int meanVarIdx = 0;
            /*
            for (int i = 0; i < 1500; i++)
            {
                //calculate the delta for the beta of add and norm layer
                deltaAddAndNormBeta2[i] = error_of_affine_outBlock2[i];
            }
            
            //calculate for gamma next
            for (int i = 0; i < 1500; i++)
            {
                Transformer_Implementation.transformerBlockFinalOutput[i] -= Transformer_Implementation.addAndNorm2Beta[i];
                Transformer_Implementation.transformerBlockFinalOutput[i] /= Transformer_Implementation.addAndNorm2Gamma[i];
                deltaAddAndNormGamma2[i] = error_of_affine_outBlock2[i] * Transformer_Implementation.transformerBlockFinalOutput[i];
                dxhatTrans[i] = error_of_affine_outBlock2[i] * Transformer_Implementation.addAndNorm2Gamma[i];
            }

            if (blockNum == 2)
            {
                double summation = 0;
                //calculate divar
                singleDivar = 0;
                
                for (int i = 0; i < 1500; i++)
                {
                    singleDivar += dxhatTrans[i] * (Transformer_Implementation.transformerBlockFinalOutputIntermediate2[i] - Transformer_Implementation.mean2_block2[meanVarIdx]);
                    dxmuTrans[i] = dxhatTrans[i] * (1.0 / Math.Sqrt(Transformer_Implementation.variance2_block2[meanVarIdx] + Transformer_Implementation.epsilon));
                }
                meanVarIdx = 0;
                normIdx = 0;

                singleDsqrtvar = -1.0 / (Transformer_Implementation.variance2_block2[meanVarIdx] + Transformer_Implementation.epsilon) * singleDivar;
                singleDvar = 0.5 * (1.0 / Math.Sqrt(Transformer_Implementation.variance2_block2[meanVarIdx] + Transformer_Implementation.epsilon)) * singleDsqrtvar;

                for (int i = 0; i < 1500; i++)
                {
                    dsqTrans[i] = 1.0 / (100 * singleDvar);
                    dxmu2Trans[i] = 2.0 * (Transformer_Implementation.transformerBlockFinalOutputIntermediate2[i] - Transformer_Implementation.mean2_block2[meanVarIdx]) * dsqTrans[i];
                    dx1Trans[i] = dxmuTrans[i] + dxmu2Trans[i];
                }

                for (int i = 0; i < 1500; i++)
                {
                    summation += dxmuTrans[i] + dxmu2Trans[i];
                }

                dmuTrans = -1.0 / summation;

                for (int i = 0; i < 1500; i++)
                {
                    dx2Trans[i] = 1.0 / (100 * dmuTrans);
                    error_of_affine_outBlock2[i] = dx1Trans[i] + dx2Trans[i];
                }
            }
            else
            {
                double summation = 0;
                //calculate divar
                singleDivar = 0;
                for (int i = 0; i < 1500; i++)
                {
                    singleDivar += dxhatTrans[i] * (Transformer_Implementation.transformerBlockFinalOutputIntermediate1[i] - Transformer_Implementation.mean2[meanVarIdx]);
                    dxmuTrans[i] = dxhatTrans[i] * (1.0 / Math.Sqrt(Transformer_Implementation.variance2[meanVarIdx] + Transformer_Implementation.epsilon));
                }
                meanVarIdx = 0;
                normIdx = 0;

                singleDsqrtvar = -1.0 / (Transformer_Implementation.variance2[meanVarIdx] + Transformer_Implementation.epsilon) * singleDivar;
                singleDvar = 0.5 * (1.0 / Math.Sqrt(Transformer_Implementation.variance2[meanVarIdx] + Transformer_Implementation.epsilon)) * singleDsqrtvar;

                for(int i = 0; i < 1500; i++)
                {
                    dsqTrans[i] = 1.0 / (100 * singleDvar);
                    dxmu2Trans[i] = 2.0 * (Transformer_Implementation.transformerBlockFinalOutputIntermediate1[i] - Transformer_Implementation.mean2[meanVarIdx]) * dsqTrans[i];
                    dx1Trans[i] = dxmuTrans[i] + dxmu2Trans[i];
                }

                for (int i = 0; i < 1500; i++)
                {
                    summation += dxmuTrans[i] + dxmu2Trans[i];
                }

                dmuTrans = -1.0 / summation;

                for (int i = 0; i < 1500; i++)
                {
                    dx2Trans[i] = 1.0 / (100 * dmuTrans);
                    error_of_affine_outBlock2[i] = dx1Trans[i] + dx2Trans[i];
                }
            }

            for (int i = 0; i < 1500; i++)
            {
                Transformer_Implementation.transformerBlockFinalOutput[i] *= Transformer_Implementation.addAndNorm2Gamma[i];
                Transformer_Implementation.transformerBlockFinalOutput[i] += Transformer_Implementation.addAndNorm2Beta[i];
            }
            */
            //elementwise multiplication of the derivative of the output layer to the error_of_affine_out2Pass2Block2
            for (int i = 0; i < 1500; i++)
            {
                derivativeOfAffine2Output[i] = error_of_affine_outBlock2[i];
            }

            for(int i = 0; i < 1500; i++)
            {
                deltaAffineMLPBias2[i] = derivativeOfAffine2Output[i];
            }

            M = 60;
            K = 100;

            if (blockNum == 2)
            {
                temp = matOps.transposeMat(Transformer_Implementation.affineIntermediateRes2, M, K);
                Array.Copy(temp, 0, affineIntermediateRes2Transposed, 0, 6000);

                //we then multiply the matrices together to get the delta for the second block second pass affine transform
                M = 60;
                K = 100;
                N = 15;

                temp = matOps.matrixMul(affineIntermediateRes2Transposed, derivativeOfAffine2Output, M, K, N);
                Array.Copy(temp, 0, deltaAffineWeights2, 0, 900);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\deltaAffineWeights2.txt");
                    for (int i = 0; i < 900; i++)
                    {
                        output.WriteLine(deltaAffineWeights2[i].ToString());
                    }
                    output.Close();
                }
            }
            else
            {
                temp = matOps.transposeMat(Transformer_Implementation.affineIntermediateRes, M, K);
                Array.Copy(temp, 0, affineIntermediateRes1Transposed, 0, 6000);

                //we then multiply the matrices together to get the delta for the second block second pass affine transform
                M = 60;
                K = 100;
                N = 15;

                temp = matOps.matrixMul(affineIntermediateRes1Transposed, derivativeOfAffine2Output, M, K, N);
                Array.Copy(temp, 0, deltaAffineWeights2, 0, 900);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\deltaAffineWeights2.txt");
                    for (int i = 0; i < 900; i++)
                    {
                        output.WriteLine(deltaAffineWeights2[i].ToString());
                    }
                    output.Close();
                }
            }

            //we now need to calculate the error of the intermediate result
            //we need to transpose the affineTransWeights2 matrix and multiply that with the error matrix from earlier
            //first transpose affineTransWeights4 by calling library transposeMat func
            M = 15;
            K = 60;
            temp = matOps.transposeMat(Transformer_Implementation.affineTransWeights2, M, K);
            Array.Copy(temp, 0, affineTrans2WeightsTransposed, 0, 900);

            //then we multiply the error matrix from earlier to this transposed weight matrix
            M = 100;
            K = 15;
            N = 60;
            temp = matOps.matrixMul(error_of_affine_outBlock2, affineTrans2WeightsTransposed, M, K, N);
            Array.Copy(temp, 0, error_of_affine_intermediate_Block2, 0, 6000);

            //take derivative of the res layer
            for (int i = 0; i < 6000; i++)
            {
                derivativeOfAffine1Output[i] = 0;
            }
            if (blockNum == 2)
            {
                //PReLU_derivative(4);
                Mish_derivative(3);
            }
            else
            {
                //PReLU_derivative(5);
                Mish_derivative(2);
            }
            //elementwise multiplication of the derivative of the output layer to the error_of_affine_intermediate_Block2
            for (int i = 0; i < 6000; i++)
            {
                derivativeOfAffine1Output[i] *= error_of_affine_intermediate_Block2[i];
            }

            M = 15;
            K = 100;

            if (blockNum == 2)
            {
                temp = matOps.transposeMat(Transformer_Implementation.residualConnectionOutputNorm, M, K);
                Array.Copy(temp, 0, residualConnectionOutputNorm2Transposed, 0, 1500);
            }
            else
            {
                temp = matOps.transposeMat(Transformer_Implementation.residualConnectionOutputNormCpy, M, K);
                Array.Copy(temp, 0, residualConnectionOutputNorm1Transposed, 0, 1500);
            }

            M = 15;
            K = 100;
            N = 60;

            if (blockNum == 2)
            {
                temp = matOps.matrixMul(residualConnectionOutputNorm2Transposed, derivativeOfAffine1Output, M, K, N);
                Array.Copy(temp, 0, deltaAffineWeights1, 0, 900);
            }
            else
            {
                temp = matOps.matrixMul(residualConnectionOutputNorm1Transposed, derivativeOfAffine1Output, M, K, N);
                Array.Copy(temp, 0, deltaAffineWeights1, 0, 900);
            }

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\deltaAffineWeights1.txt");
                for (int i = 0; i < 900; i++)
                {
                    output.WriteLine(deltaAffineWeights1[i].ToString());
                }
                output.Close();
            }

            for(int i = 0; i < 6000; i++)
            {
                deltaAffineMLPBias1[i] = derivativeOfAffine1Output[i];
                deltaAffineMLPPreluParam1[i] = derivativeOfAffine1Output[i];
            }

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\deltaAddAndNormGamma2.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output.WriteLine(deltaAddAndNormGamma2[i].ToString());
                }
                output.Close();
            }

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\deltaAffineMLPBias1.txt");
                for (int i = 0; i < 6000; i++)
                {
                    output.WriteLine(deltaAffineMLPBias1[i].ToString());
                }
                output.Close();
            }
        }

        public void finalLinearLayerCalculateAdjustments(int blockNum)
        {
            double[] temp;
            int M = 60;
            int K = 15;
            int N = 15;

            //we calculate the error of the previous layer before the affine transformation MLP by multiplying
            //the transpose of affineTransWeights1 with the error matrix from earlier to get the error of the output of attention block
            temp = matOps.transposeMat(Transformer_Implementation.affineTransWeights1, M, K);
            Array.Copy(temp, 0, affineTrans1WeightsTransposed, 0, 900);

            //do multiplication to find next error matrix
            M = 100;
            K = 60;

            temp = matOps.matrixMul(error_of_affine_intermediate_Block2, affineTrans1WeightsTransposed, M, K, N);
            Array.Copy(temp, 0, error_of_output_attention_block, 0, 1500);

            for(int i = 0; i < 1500; i++)
            {
                error_of_output_attention_block[i] += error_of_affine_outBlock2[i];
                error_of_output_attention_block[i] /= 2;
            }
            /*
            //normalization backprop
            int normIdx;
            int meanVarIdx = 0;

            for (int i = 0; i < 1500; i++)
            {
                //calculate the delta for the beta of add and norm layer
                deltaAddAndNormBeta1[i] = error_of_output_attention_block[i];
            }

            //calculate for gamma next
            for (int i = 0; i < 1500; i++)
            {
                if(blockNum == 2)
                {
                    Transformer_Implementation.residualConnectionOutputNorm[i] -= Transformer_Implementation.addAndNorm1Beta[i];
                    Transformer_Implementation.residualConnectionOutputNorm[i] /= Transformer_Implementation.addAndNorm1Gamma[i];
                    deltaAddAndNormGamma1[i] = error_of_output_attention_block[i] * Transformer_Implementation.residualConnectionOutputNorm[i];
                    dxhatTrans[i] = error_of_output_attention_block[i] * Transformer_Implementation.addAndNorm1Gamma[i];
                }
                else
                {
                    Transformer_Implementation.residualConnectionOutputNormCpy[i] -= Transformer_Implementation.addAndNorm1Beta[i];
                    Transformer_Implementation.residualConnectionOutputNormCpy[i] /= Transformer_Implementation.addAndNorm1Gamma[i];
                    deltaAddAndNormGamma1[i] = error_of_output_attention_block[i] * Transformer_Implementation.residualConnectionOutputNormCpy[i];
                    dxhatTrans[i] = error_of_output_attention_block[i] * Transformer_Implementation.addAndNorm1Gamma[i];
                }
            }

            if (blockNum == 2)
            {
                double summation = 0;
                //calculate divar
                singleDivar = 0;
                for (int i = 0; i < 1500; i++)
                {
                    singleDivar += dxhatTrans[i] * (Transformer_Implementation.residualConnectionOutputNormIntermediate2[i] - Transformer_Implementation.mean1_block2[meanVarIdx]);
                    dxmuTrans[i] = dxhatTrans[i] * (1.0 / Math.Sqrt(Transformer_Implementation.variance1_block2[meanVarIdx] + Transformer_Implementation.epsilon));
                }
                meanVarIdx = 0;
                normIdx = 0;

                singleDsqrtvar = -1.0 / (Transformer_Implementation.variance1_block2[meanVarIdx] + Transformer_Implementation.epsilon) * singleDivar;
                singleDvar = 0.5 * (1.0 / Math.Sqrt(Transformer_Implementation.variance1_block2[meanVarIdx] + Transformer_Implementation.epsilon)) * singleDsqrtvar;

                for (int i = 0; i < 1500; i++)
                {
                    dsqTrans[i] = 1.0 / (100 * singleDvar);
                    dxmu2Trans[i] = 2.0 * (Transformer_Implementation.residualConnectionOutputNormIntermediate2[i] - Transformer_Implementation.mean1_block2[meanVarIdx]) * dsqTrans[i];
                    dx1Trans[i] = dxmuTrans[i] + dxmu2Trans[i];
                }

                for (int i = 0; i < 1500; i++)
                {
                    summation += dxmuTrans[i] + dxmu2Trans[i];
                }

                dmuTrans = -1.0 / summation;

                for (int i = 0; i < 1500; i++)
                {
                    dx2Trans[i] = 1.0 / (100 * dmuTrans);
                    error_of_output_attention_block[i] = dx1Trans[i] + dx2Trans[i];
                }
            }
            else
            {
                double summation = 0;
                //calculate divar
                singleDivar = 0;
                for (int i = 0; i < 1500; i++)
                {
                    singleDivar += dxhatTrans[i] * (Transformer_Implementation.residualConnectionOutputNormIntermediate1[i] - Transformer_Implementation.mean1[meanVarIdx]);
                    dxmuTrans[i] = dxhatTrans[i] * (1.0 / Math.Sqrt(Transformer_Implementation.variance1[meanVarIdx] + Transformer_Implementation.epsilon));
                }
                meanVarIdx = 0;
                normIdx = 0;

                singleDsqrtvar = -1.0 / (Transformer_Implementation.variance1[meanVarIdx] + Transformer_Implementation.epsilon) * singleDivar;
                singleDvar = 0.5 * (1.0 / Math.Sqrt(Transformer_Implementation.variance1[meanVarIdx] + Transformer_Implementation.epsilon)) * singleDsqrtvar;

                for (int i = 0; i < 1500; i++)
                {
                    dsqTrans[i] = 1.0 / (100 * singleDvar);
                    dxmu2Trans[i] = 2.0 * (Transformer_Implementation.residualConnectionOutputNormIntermediate1[i] - Transformer_Implementation.mean1[meanVarIdx]) * dsqTrans[i];
                    dx1Trans[i] = dxmuTrans[i] + dxmu2Trans[i];
                }

                for (int i = 0; i < 1500; i++)
                {
                    summation += dxmuTrans[i] + dxmu2Trans[i];
                }

                dmuTrans = -1.0 / summation;

                for (int i = 0; i < 1500; i++)
                {
                    dx2Trans[i] = 1.0 / (100 * dmuTrans);
                    error_of_output_attention_block[i] = dx1Trans[i] + dx2Trans[i];
                }
            }

            for (int i = 0; i < 1500; i++)
            {
                if (blockNum == 2)
                {
                    Transformer_Implementation.residualConnectionOutputNorm[i] *= Transformer_Implementation.addAndNorm1Gamma[i];
                    Transformer_Implementation.residualConnectionOutputNorm[i] += Transformer_Implementation.addAndNorm1Beta[i];
                }
                else
                {
                    Transformer_Implementation.residualConnectionOutputNormCpy[i] *= Transformer_Implementation.addAndNorm1Gamma[i];
                    Transformer_Implementation.residualConnectionOutputNormCpy[i] += Transformer_Implementation.addAndNorm1Beta[i];
                }
            }
            */
            Array.Copy(error_of_output_attention_block, 0, derivativeOfAttentionBlockOutput, 0, 1500);
            //multiply to get delta for the final linear layer weight matrix
            M = 15;
            K = 100;
            N = 15;

            if (blockNum == 2)
            {
                temp = matOps.matrixMul(residualConnectionOutputNorm2Transposed, derivativeOfAttentionBlockOutput, M, K, N);
                Array.Copy(temp, 0, deltaFinalLinearLayerWeights, 0, 225);
            }
            else
            {
                temp = matOps.matrixMul(residualConnectionOutputNorm1Transposed, derivativeOfAttentionBlockOutput, M, K, N);
                Array.Copy(temp, 0, deltaFinalLinearLayerWeights, 0, 225);
            }

            if(predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\deltaAddAndNormGamma1.txt");
                for(int i = 0; i < 1500; i++)
                {
                    output.WriteLine(deltaAddAndNormGamma1[i].ToString());
                }
                output.Close();
            }

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\deltaFinalLinearLayerWeights.txt");
                for (int i = 0; i < 225; i++)
                {
                    output.WriteLine(deltaFinalLinearLayerWeights[i].ToString());
                }
                output.Close();
            }
        }

        public void queryKeyValueCalculateAdjustments()
        {
            double[] temp;
            int M = 15;
            int K = 15;
            int N = 15;

            //we calculate the error of the concatenated filtered value matrix by taking the transpose of the final linear layer weight matrix
            //and multiplying the final output with the final output of the attention head
            temp = matOps.transposeMat(Transformer_Implementation.finalLinearLayerWeights, M, K);
            Array.Copy(temp, 0, finalLinearLayerWeightsTransposed, 0, 225);

            M = 100;
            //multiply this with the error output of the attention block
            temp = matOps.matrixMul(error_of_output_attention_block, finalLinearLayerWeightsTransposed, M, K, N);
            Array.Copy(temp, 0, error_of_concat_filteredVal_output, 0, 1500);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\error_of_concat_filteredVal_output.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output.WriteLine(error_of_concat_filteredVal_output[i].ToString());
                }
                output.Close();
            }

            //we deconcatenate this matrix into 3 separate matrices each 100 x 5 dimensions
            funcs.deconcatenateFilteredValMatrices();
            if(predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\deconcatMat1.txt");
                StreamWriter output2 = File.AppendText(@"X:\deconcatMat2.txt");
                StreamWriter output3 = File.AppendText(@"X:\deconcatMat3.txt");
                for(int i = 0; i < 500; i++)
                {
                    output.WriteLine(deconcatMat1[i].ToString());
                    output2.WriteLine(deconcatMat2[i].ToString());
                    output3.WriteLine(deconcatMat3[i].ToString());
                }
                output.Close();
                output2.Close();
                output3.Close();
            }

            attentionHeadsDeltaCalculations();

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\deltaQueryWeightsHead1.txt");
                StreamWriter output2 = File.AppendText(@"X:\deltaKeyWeightsHead1.txt");
                StreamWriter output3 = File.AppendText(@"X:\deltaValueWeightsHead1.txt");

                for (int i = 0; i < 75; i++)
                {
                    output.WriteLine(deltaQueryWeightsHead1[i].ToString());
                    output2.WriteLine(deltaKeyWeightsHead1[i].ToString());
                    output3.WriteLine(deltaValueWeightsHead1[i].ToString());
                }
                output.Close();
                output2.Close();
                output3.Close();
            }
            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output4 = File.AppendText(@"X:\deltaQueryWeightsHead2.txt");
                StreamWriter output5 = File.AppendText(@"X:\deltaKeyWeightsHead2.txt");
                StreamWriter output6 = File.AppendText(@"X:\deltaValueWeightsHead2.txt");

                for (int i = 0; i < 75; i++)
                {
                    output4.WriteLine(deltaQueryWeightsHead2[i].ToString());
                    output5.WriteLine(deltaKeyWeightsHead2[i].ToString());
                    output6.WriteLine(deltaValueWeightsHead2[i].ToString());
                }
                output4.Close();
                output5.Close();
                output6.Close();
            }
            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output7 = File.AppendText(@"X:\deltaQueryWeightsHead3.txt");
                StreamWriter output8 = File.AppendText(@"X:\deltaKeyWeightsHead3.txt");
                StreamWriter output9 = File.AppendText(@"X:\deltaValueWeightsHead3.txt");

                for (int i = 0; i < 75; i++)
                {
                    output7.WriteLine(deltaQueryWeightsHead3[i].ToString());
                    output8.WriteLine(deltaKeyWeightsHead3[i].ToString());
                    output9.WriteLine(deltaValueWeightsHead3[i].ToString());
                }
                output7.Close();
                output8.Close();
                output9.Close();
            }
        }

        public void attentionHeadsDeltaCalculations()
        {
            double[] temp;
            double[] attentionFilterTrans1 = new double[10000];
            double[] attentionFilterTrans2 = new double[10000];
            double[] attentionFilterTrans3 = new double[10000];
            int M = 15;
            int K = 100;
            int N = 5;
            double scaleVal = Math.Sqrt(5);

            temp = matOps.transposeMat(Transformer_Implementation.inputFromConvModule, M, K);
            Array.Copy(temp, 0, inputFromConvModuleTransposed, 0, 1500);

            Parallel.For(1, 4, (j, state) =>
            {
                //find softmax derivative for attention filters
                if (j == 1)
                {
                    temp = matOps.transposeMat(Transformer_Implementation.attention_filter_head1, 100, 100);
                    Array.Copy(temp, 0, attentionFilterTrans1, 0, 10000);
                    //temp = matOps.matrixMul(attentionFilterTrans1, deconcatMat1, 100, 100, 5);
                    temp = matOps.matrixMul(Transformer_Implementation.attention_filter_head1, deconcatMat1, 100, 100, 5);
                    Array.Copy(temp, 0, valueHead1Err, 0, 500);

                    temp = softmax_derivative(Transformer_Implementation.attention_filter_head1);
                    Array.Copy(temp, 0, attention_filter_head1_der, 0, 10000);

                    temp = matOps.transposeMat(attention_filter_head1_der, 100, 100);
                    Array.Copy(temp, 0, attention_filter_head1_trans, 0, 10000);

                    temp = matOps.transposeMat(deconcatMat1, 5, 100);
                    Array.Copy(temp, 0, deconcatMat1, 0, 500);
                    temp = matOps.matrixMul(Transformer_Implementation.value_head1, deconcatMat1, 100, 5, 100);
                    Array.Copy(temp, 0, scores_query_gradient_head1, 0, 10000);
                    temp = matOps.transposeMat(deconcatMat1, 100, 5);
                    Array.Copy(temp, 0, deconcatMat1, 0, 500);
                    temp = matOps.transposeMat(Transformer_Implementation.value_head1, 5, 100);
                    Array.Copy(temp, 0, Transformer_Implementation.value_head1, 0, 500);
                    temp = matOps.matrixMul(deconcatMat1, Transformer_Implementation.value_head1, 100, 5, 100);
                    Array.Copy(temp, 0, scores_key_gradient_head1, 0, 10000);
                    temp = matOps.transposeMat(Transformer_Implementation.value_head1, 100, 5);
                    Array.Copy(temp, 0, Transformer_Implementation.value_head1, 0, 500);

                    for (int i = 0; i < 500; i++)
                    {
                        Transformer_Implementation.query_head1[i] /= scaleVal;
                        Transformer_Implementation.key_head1[i] /= scaleVal;
                    }

                    for (int i = 0; i < 10000; i++)
                    {
                        scores_query_gradient_head1[i] *= attention_filter_head1_trans[i];
                        scores_key_gradient_head1[i] *= attention_filter_head1_der[i];
                    }
                    //temp = matOps.matrixMul(attention_filter_head1_trans, scores_query_gradient_head1, 100, 100, 100);
                    //Array.Copy(temp, 0, scores_query_gradient_head1, 0, 10000);
                    temp = matOps.matrixMul(scores_query_gradient_head1, Transformer_Implementation.key_head1, 100, 100, 5);
                    Array.Copy(temp, 0, queryHead1Err, 0, 500);
                    //temp = matOps.matrixMul(attention_filter_head1_der, scores_key_gradient_head1, 100, 100, 100);
                    //Array.Copy(temp, 0, scores_key_gradient_head1, 0, 10000);
                    temp = matOps.matrixMul(scores_key_gradient_head1, Transformer_Implementation.query_head1, 100, 100, 5);
                    Array.Copy(temp, 0, keyHead1Err, 0, 500);

                    temp = matOps.matrixMul(inputFromConvModuleTransposed, queryHead1Err, M, K, N);
                    Array.Copy(temp, 0, deltaQueryWeightsHead1, 0, 75);
                    temp = matOps.matrixMul(inputFromConvModuleTransposed, keyHead1Err, M, K, N);
                    Array.Copy(temp, 0, deltaKeyWeightsHead1, 0, 75);
                    temp = matOps.matrixMul(inputFromConvModuleTransposed, valueHead1Err, M, K, N);
                    Array.Copy(temp, 0, deltaValueWeightsHead1, 0, 75);
                }
                else if (j == 2)
                {
                    temp = matOps.transposeMat(Transformer_Implementation.attention_filter_head2, 100, 100);
                    Array.Copy(temp, 0, attentionFilterTrans2, 0, 10000);
                    //temp = matOps.matrixMul(attentionFilterTrans2, deconcatMat2, 100, 100, 5);
                    temp = matOps.matrixMul(Transformer_Implementation.attention_filter_head2, deconcatMat2, 100, 100, 5);
                    Array.Copy(temp, 0, valueHead2Err, 0, 500);

                    temp = softmax_derivative(Transformer_Implementation.attention_filter_head2);
                    Array.Copy(temp, 0, attention_filter_head2_der, 0, 10000);

                    temp = matOps.transposeMat(attention_filter_head2_der, 100, 100);
                    Array.Copy(temp, 0, attention_filter_head2_trans, 0, 10000);

                    temp = matOps.transposeMat(deconcatMat2, 5, 100);
                    Array.Copy(temp, 0, deconcatMat2, 0, 500);
                    temp = matOps.matrixMul(Transformer_Implementation.value_head2, deconcatMat2, 100, 5, 100);
                    Array.Copy(temp, 0, scores_query_gradient_head2, 0, 10000);
                    temp = matOps.transposeMat(deconcatMat2, 100, 5);
                    Array.Copy(temp, 0, deconcatMat2, 0, 500);
                    temp = matOps.transposeMat(Transformer_Implementation.value_head2, 5, 100);
                    Array.Copy(temp, 0, Transformer_Implementation.value_head2, 0, 500);
                    temp = matOps.matrixMul(deconcatMat2, Transformer_Implementation.value_head2, 100, 5, 100);
                    Array.Copy(temp, 0, scores_key_gradient_head2, 0, 10000);
                    temp = matOps.transposeMat(Transformer_Implementation.value_head2, 100, 5);
                    Array.Copy(temp, 0, Transformer_Implementation.value_head2, 0, 500);

                    for (int i = 0; i < 500; i++)
                    {
                        Transformer_Implementation.query_head2[i] /= scaleVal;
                        Transformer_Implementation.key_head2[i] /= scaleVal;
                    }

                    for (int i = 0; i < 10000; i++)
                    {
                        scores_query_gradient_head2[i] *= attention_filter_head2_trans[i];
                        scores_key_gradient_head2[i] *= attention_filter_head2_der[i];
                    }
                    //temp = matOps.matrixMul(attention_filter_head2_trans, scores_query_gradient_head2, 100, 100, 100);
                    //Array.Copy(temp, 0, scores_query_gradient_head2, 0, 10000);
                    temp = matOps.matrixMul(scores_query_gradient_head2, Transformer_Implementation.key_head2, 100, 100, 5);
                    Array.Copy(temp, 0, queryHead2Err, 0, 500);
                    //temp = matOps.matrixMul(attention_filter_head2_der, scores_key_gradient_head2, 100, 100, 100);
                    //Array.Copy(temp, 0, scores_key_gradient_head2, 0, 10000);
                    temp = matOps.matrixMul(scores_key_gradient_head2, Transformer_Implementation.query_head2, 100, 100, 5);
                    Array.Copy(temp, 0, keyHead2Err, 0, 500);

                    temp = matOps.matrixMul(inputFromConvModuleTransposed, queryHead2Err, M, K, N);
                    Array.Copy(temp, 0, deltaQueryWeightsHead2, 0, 75);
                    temp = matOps.matrixMul(inputFromConvModuleTransposed, keyHead2Err, M, K, N);
                    Array.Copy(temp, 0, deltaKeyWeightsHead2, 0, 75);
                    temp = matOps.matrixMul(inputFromConvModuleTransposed, valueHead2Err, M, K, N);
                    Array.Copy(temp, 0, deltaValueWeightsHead2, 0, 75);
                }
                else if (j == 3)
                {
                    temp = matOps.transposeMat(Transformer_Implementation.attention_filter_head3, 100, 100);
                    Array.Copy(temp, 0, attentionFilterTrans3, 0, 10000);
                    //temp = matOps.matrixMul(attentionFilterTrans3, deconcatMat3, 100, 100, 5);
                    temp = matOps.matrixMul(Transformer_Implementation.attention_filter_head3, deconcatMat3, 100, 100, 5);
                    Array.Copy(temp, 0, valueHead3Err, 0, 500);

                    temp = softmax_derivative(Transformer_Implementation.attention_filter_head3);
                    Array.Copy(temp, 0, attention_filter_head3_der, 0, 10000);

                    temp = matOps.transposeMat(attention_filter_head3_der, 100, 100);
                    Array.Copy(temp, 0, attention_filter_head3_trans, 0, 10000);

                    temp = matOps.transposeMat(deconcatMat3, 5, 100);
                    Array.Copy(temp, 0, deconcatMat3, 0, 500);
                    temp = matOps.matrixMul(Transformer_Implementation.value_head3, deconcatMat3, 100, 5, 100);
                    Array.Copy(temp, 0, scores_query_gradient_head3, 0, 10000);
                    temp = matOps.transposeMat(deconcatMat3, 100, 5);
                    Array.Copy(temp, 0, deconcatMat3, 0, 500);
                    temp = matOps.transposeMat(Transformer_Implementation.value_head3, 5, 100);
                    Array.Copy(temp, 0, Transformer_Implementation.value_head3, 0, 500);
                    temp = matOps.matrixMul(deconcatMat3, Transformer_Implementation.value_head3, 100, 5, 100);
                    Array.Copy(temp, 0, scores_key_gradient_head3, 0, 10000);
                    temp = matOps.transposeMat(Transformer_Implementation.value_head3, 100, 5);
                    Array.Copy(temp, 0, Transformer_Implementation.value_head3, 0, 500);

                    for (int i = 0; i < 500; i++)
                    {
                        Transformer_Implementation.query_head3[i] /= scaleVal;
                        Transformer_Implementation.key_head3[i] /= scaleVal;
                    }

                    for (int i = 0; i < 10000; i++)
                    {
                        scores_query_gradient_head3[i] *= attention_filter_head3_trans[i];
                        scores_key_gradient_head3[i] *= attention_filter_head3_der[i];
                    }
                    //temp = matOps.matrixMul(attention_filter_head3_trans, scores_query_gradient_head3, 100, 100, 100);
                    //Array.Copy(temp, 0, scores_query_gradient_head3, 0, 10000);
                    temp = matOps.matrixMul(scores_query_gradient_head3, Transformer_Implementation.key_head3, 100, 100, 5);
                    Array.Copy(temp, 0, queryHead3Err, 0, 500);
                    //temp = matOps.matrixMul(attention_filter_head3_der, scores_key_gradient_head3, 100, 100, 100);
                    //Array.Copy(temp, 0, scores_key_gradient_head3, 0, 10000);
                    temp = matOps.matrixMul(scores_key_gradient_head3, Transformer_Implementation.query_head3, 100, 100, 5);
                    Array.Copy(temp, 0, keyHead3Err, 0, 500);

                    temp = matOps.matrixMul(inputFromConvModuleTransposed, queryHead3Err, M, K, N);
                    Array.Copy(temp, 0, deltaQueryWeightsHead3, 0, 75);
                    temp = matOps.matrixMul(inputFromConvModuleTransposed, keyHead3Err, M, K, N);
                    Array.Copy(temp, 0, deltaKeyWeightsHead3, 0, 75);
                    temp = matOps.matrixMul(inputFromConvModuleTransposed, valueHead3Err, M, K, N);
                    Array.Copy(temp, 0, deltaValueWeightsHead3, 0, 75);
                }

                //if(predictorGui.predictorGui1.enableOutputs.Checked == true)
                //{
                //    StreamWriter output = File.AppendText(@"X:\attention_filter_head1_der.txt");
                //    for(int i = 0; i < 10000; i++)
                //    {
                //        output.WriteLine(attention_filter_head1_der[i].ToString());
                //    }
                //    output.Close();
                //}

                /*if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\scores_query_gradient_head1.txt");
                    for (int i = 0; i < 10000; i++)
                    {
                        output.WriteLine(scores_query_gradient_head1[i].ToString());
                    }
                    output.Close();
                }
                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\scores_key_gradient_head1.txt");
                    for (int i = 0; i < 10000; i++)
                    {
                        output.WriteLine(scores_key_gradient_head1[i].ToString());
                    }
                    output.Close();
                }*/
            });
        }

        public double[] softmax_derivative(double[] attentionFilter)
        {
            double[] temp = new double[10000];
            int row = 0;
            int col = 0;
            for(int i = 0; i < 10000; i++)
            {
                if (i % 100 == 0 && i != 0)
                {
                    row++;
                    col = 0;
                }
                if (row != col)
                {
                    if (attentionFilter[i] == 0 || attentionFilter[i] == 1)
                    {
                        temp[i] = 0.0001;
                    }
                    else
                    {
                        temp[i] = -attentionFilter[i] * attentionFilter[i];
                    }
                }
                else
                {
                    if (attentionFilter[i] == 0 || attentionFilter[i] == 1)
                    {
                        temp[i] = 0.0001;
                    }
                    else
                    {
                        temp[i] = attentionFilter[i] * (1.0 - attentionFilter[i]);
                    }
                }
                col++;
            }
            //if(predictorGui.predictorGui1.enableOutputs.Checked == true)
            //{
            //    StreamWriter output = File.AppendText(@"X:\attentionFilterDerivative.txt");
            //    for(int i = 0; i < 10000; i++)
            //    {
            //        output.WriteLine(temp[i].ToString());
            //    }
            //    output.Close();
            //}
            return temp;
        }

        public void mlpCalculateAdjustments()
        {
            int M = 3;
            int K = 3;
            int N = 1;
            double[] temp;

            double sumOfSquares = 0;

            //calculate derivative of cross entropy loss wrt softmax from given one hot encoding to calculate error of output layer
            for(int i = 0; i < 3; i++)
            {
                /*
                if(actualOutcomes[i] == 0.0F)
                {
                    actualOutcomes[i] = (epsilon * 1000000) / (3.0 - 1.0);
                }
                else
                {
                    actualOutcomes[i] = 1.0 - (epsilon * 1000000);
                }
                */
                dE_dZ_WRTThirdLayerInputs[i] = secondLayerOutCpy[i] - actualOutcomes[i];
                if(predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\errorOfMLPSecondLayer.txt");
                    output.WriteLine(secondLayerOutCpy[i].ToString() + " - " + actualOutcomes[i].ToString() + " = " + dE_dZ_WRTThirdLayerInputs[i].ToString());
                    output.Close();
                }
            }

            for(int i = 0; i < 192; i++)
            {
                sumOfSquares += (MLP.secondLayerWeights[i] * MLP.secondLayerWeights[i]);
            }

            sumOfSquares *= L2RegLambda;

            for(int i = 0; i < 3; i++)
            {
                dE_dZ_WRTThirdLayerInputs[i] += sumOfSquares;
                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\errorOfMLPSecondLayer_L2Reg.txt");
                    output.WriteLine(secondLayerOutCpy[i].ToString() + " - " + actualOutcomes[i].ToString() + " + " + sumOfSquares + " = " + dE_dZ_WRTThirdLayerInputs[i].ToString());
                    output.Close();
                }
            }

            //first we transpose the weight matrix for the second layer
            M = 64;
            K = 3;

            temp = matOps.transposeMat(MLP.secondLayerWeights, M, K);
            Array.Copy(temp, 0, secondLayerWeightsTransposed, 0, 192);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\mlpSecondLayerWeightsTransposed.txt");
                for (int i = 0; i < 192; i++)
                {
                    output.WriteLine(secondLayerWeightsTransposed[i].ToString());
                }
                output.Close();
            }

            //multiply secondLayerWeightsTransposed by the error_of_second_layer to get error_of_first_layer
            N = 1;

            temp = matOps.matrixMul(secondLayerWeightsTransposed, dE_dZ_WRTThirdLayerInputs, M, K, N);
            Array.Copy(temp, 0, error_of_first_layer, 0, 64);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\errorOfMLPFirstLayer.txt");
                for (int i = 0; i < 64; i++)
                {
                    output.WriteLine(error_of_first_layer[i].ToString());
                }
                output.Close();
            }

            //we transpose the firstLayerOut matrix for the second layer adjustments
            M = 1;
            K = 64;

            //temp = matOps.transposeMat(firstLayerOutCpy, M, K);
            Array.Copy(firstLayerOutCpy, 0, firstLayerOutTransposed, 0, 64);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\firstLayerOutTransposed.txt");
                for (int i = 0; i < 64; i++)
                {
                    output.WriteLine(firstLayerOutTransposed[i].ToString());
                }
                output.Close();
            }

            //multiply derivativeofSecondLayerOut with firstLayerOutTransposed
            M = 3;
            K = 1;
            N = 64;

            temp = matOps.matrixMul(dE_dZ_WRTThirdLayerInputs, firstLayerOutTransposed, M, K, N);
            Array.Copy(temp, 0, deltaSecondWeights, 0, 192);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\deltaSecondWeights.txt");
                for (int i = 0; i < 192; i++)
                {
                    output.WriteLine(deltaSecondWeights[i].ToString());
                }
                output.Close();
            }

            sumOfSquares = 0;
            for(int i = 0; i < 96000; i++)
            {
                sumOfSquares += (MLP.firstLayerWeights[i] * MLP.firstLayerWeights[i]);
            }
            sumOfSquares *= L2RegLambda;

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\errorOfMLPFirstLayer_L2Reg.txt");
                for (int i = 0; i < 64; i++)
                {
                    output.WriteLine(error_of_first_layer[i].ToString() + " + " + sumOfSquares);
                }
                output.Close();
            }
            for (int i = 0; i < 64; i++)
            {
                error_of_first_layer[i] += sumOfSquares;
            }

            //now that we have deltaSecondWeights we need to calculate deltaThirdWeights
            //we transpose the input to the MLP

            M = 1;
            K = 1500;

            //temp = matOps.transposeMat(firstLayerInCpy, M, K);
            Array.Copy(firstLayerInCpy, 0, firstLayerInputTransposed, 0, 1500);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\firstLayerInputCpyTransposed.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output.WriteLine(firstLayerInputTransposed[i].ToString());
                }
                output.Close();
            }

            //calculate derivative of first layer out
            if (predictorGui.predictorGui1.preluSelectFinalMLP.Checked == true)
            {
                PReLU_derivative(1);
            }
            else if (predictorGui.predictorGui1.mishSelectFinalMLP.Checked == true)
            {
                Mish_derivative(1);
            }

            //element wise multiplication of error of first layer to derivative of first layer
            for (int i = 0; i < 64; i++)
            {
                derivativeOfFirstLayerOut[i] *= error_of_first_layer[i] * MLP.dropout_mask[i]; //added 5/8/2022 potential dropout bug with earlier implementation
            }

            if(predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\derivativeOfFirstLayerOut.txt");
                for(int i = 0; i < 64; i++)
                {
                    output.WriteLine(derivativeOfFirstLayerOut[i].ToString());
                }
                output.Close();
            }

            //multiply derivative by input transposed
            M = 64;
            K = 1;
            N = 1500;

            temp = matOps.matrixMul(derivativeOfFirstLayerOut, firstLayerInputTransposed, M, K, N);
            Array.Copy(temp, 0, deltaFirstWeights, 0, 96000);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\deltaFirstWeights.txt");
                for(int i = 0; i < 96000; i++)
                {
                    output.WriteLine(deltaFirstWeights[i].ToString());
                }
                output.Close();
            }

            //calculate derivative of first layer out
            if (predictorGui.predictorGui1.preluSelectFinalMLP.Checked == true)
            {
                PReLU_derivative(1);
            }
            else if (predictorGui.predictorGui1.mishSelectFinalMLP.Checked == true)
            {
                Mish_derivative(1);
            }

            //element wise multiplication of error of first layer to derivative of first layer
            for (int i = 0; i < 64; i++)
            {
                error_of_first_layer[i] -= sumOfSquares;
                derivativeOfFirstLayerOut[i] *= error_of_first_layer[i] * MLP.dropout_mask[i]; //added 5/8/2022 potential dropout bug with earlier implementation
            }

            for (int i = 0; i < 64; i++)
            {
                deltaFirstBias[i] = derivativeOfFirstLayerOut[i];
                deltaFirstPReLU[i] = derivativeOfFirstLayerOut[i];
            }

            for (int i = 0; i < 64; i++)
            {
                error_of_first_layer[i] += sumOfSquares;
            }
        }
        public void calculateErrorMatForTransBlock1()
        {
            double[] temp;

            //transpose weight matrices for each attention head
            temp = matOps.transposeMat(Transformer_Implementation.queryLinearLayerWeights_head1, 5, 15);
            Array.Copy(temp, 0, queryLinearLayerWeightsHead1Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.keyLinearLayerWeights_head1, 5, 15);
            Array.Copy(temp, 0, keyLinearLayerWeightsHead1Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.valueLinearLayerWeights_head1, 5, 15);
            Array.Copy(temp, 0, valueLinearLayerWeightsHead1Trans, 0, 75);

            temp = matOps.transposeMat(Transformer_Implementation.queryLinearLayerWeights_head2, 5, 15);
            Array.Copy(temp, 0, queryLinearLayerWeightsHead2Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.keyLinearLayerWeights_head2, 5, 15);
            Array.Copy(temp, 0, keyLinearLayerWeightsHead2Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.valueLinearLayerWeights_head2, 5, 15);
            Array.Copy(temp, 0, valueLinearLayerWeightsHead2Trans, 0, 75);

            temp = matOps.transposeMat(Transformer_Implementation.queryLinearLayerWeights_head3, 5, 15);
            Array.Copy(temp, 0, queryLinearLayerWeightsHead3Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.keyLinearLayerWeights_head3, 5, 15);
            Array.Copy(temp, 0, keyLinearLayerWeightsHead3Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.valueLinearLayerWeights_head3, 5, 15);
            Array.Copy(temp, 0, valueLinearLayerWeightsHead3Trans, 0, 75);

            //multiply with the output of the linear layers for each attention head to get error of the input from conv module
            temp = matOps.matrixMul(Transformer_Implementation.query_head1, queryLinearLayerWeightsHead1Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module1, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.key_head1, keyLinearLayerWeightsHead1Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module2, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.value_head1, valueLinearLayerWeightsHead1Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module3, 0, 1500);

            temp = matOps.matrixMul(Transformer_Implementation.query_head2, queryLinearLayerWeightsHead2Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module4, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.key_head2, keyLinearLayerWeightsHead2Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module5, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.value_head2, valueLinearLayerWeightsHead2Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module6, 0, 1500);

            temp = matOps.matrixMul(Transformer_Implementation.query_head3, queryLinearLayerWeightsHead3Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module7, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.key_head3, keyLinearLayerWeightsHead3Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module8, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.value_head3, valueLinearLayerWeightsHead3Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module9, 0, 1500);

            if(predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\error_of_input_from_conv_module1.txt");
                for(int i = 0; i < 1500; i++)
                {
                    output.WriteLine(error_of_input_from_conv_module1[i].ToString());
                }
                output.Close();

                StreamWriter output2 = File.AppendText(@"X:\error_of_input_from_conv_module2.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output2.WriteLine(error_of_input_from_conv_module2[i].ToString());
                }
                output2.Close();

                StreamWriter output3 = File.AppendText(@"X:\error_of_input_from_conv_module3.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output3.WriteLine(error_of_input_from_conv_module3[i].ToString());
                }
                output3.Close();

                StreamWriter output4 = File.AppendText(@"X:\error_of_input_from_conv_module4.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output4.WriteLine(error_of_input_from_conv_module4[i].ToString());
                }
                output4.Close();

                StreamWriter output5 = File.AppendText(@"X:\error_of_input_from_conv_module5.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output5.WriteLine(error_of_input_from_conv_module5[i].ToString());
                }
                output5.Close();

                StreamWriter output6 = File.AppendText(@"X:\error_of_input_from_conv_module6.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output6.WriteLine(error_of_input_from_conv_module6[i].ToString());
                }
                output6.Close();

                StreamWriter output7 = File.AppendText(@"X:\error_of_input_from_conv_module7.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output7.WriteLine(error_of_input_from_conv_module7[i].ToString());
                }
                output7.Close();

                StreamWriter output8 = File.AppendText(@"X:\error_of_input_from_conv_module8.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output8.WriteLine(error_of_input_from_conv_module8[i].ToString());
                }
                output8.Close();

                StreamWriter output9 = File.AppendText(@"X:\error_of_input_from_conv_module9.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output9.WriteLine(error_of_input_from_conv_module9[i].ToString());
                }
                output9.Close();
            }

            //find the average of all the errors
            for (int i = 0; i < 1500; i++)
            {
                avg_error_of_input_from_conv_module[i] = (error_of_input_from_conv_module1[i] +
                                                          error_of_input_from_conv_module2[i] +
                                                          error_of_input_from_conv_module3[i] +
                                                          error_of_input_from_conv_module4[i] +
                                                          error_of_input_from_conv_module5[i] +
                                                          error_of_input_from_conv_module6[i] +
                                                          error_of_input_from_conv_module7[i] +
                                                          error_of_input_from_conv_module8[i] +
                                                          error_of_input_from_conv_module9[i]) / 9;
            }

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\derivativeOfAttentionBlockOutput.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output.WriteLine(derivativeOfAttentionBlockOutput[i].ToString());
                }
                output.Close();

                StreamWriter output2 = File.AppendText(@"X:\avg_error_of_input_from_conv_module_fresh_calc.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output2.WriteLine(avg_error_of_input_from_conv_module[i].ToString());
                }
                output2.Close();
            }

            for (int i = 0; i < 1500; i++)
            {
                avg_error_of_input_from_conv_module[i] += derivativeOfAttentionBlockOutput[i];
                avg_error_of_input_from_conv_module[i] /= 2;
            }
        }

        public void convModuleFilterAdjustments()
        {
            double[] temp;

            //transpose weight matrices for each attention head
            temp = matOps.transposeMat(Transformer_Implementation.queryLinearLayerWeights_head1, 5, 15);
            Array.Copy(temp, 0, queryLinearLayerWeightsHead1Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.keyLinearLayerWeights_head1, 5, 15);
            Array.Copy(temp, 0, keyLinearLayerWeightsHead1Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.valueLinearLayerWeights_head1, 5, 15);
            Array.Copy(temp, 0, valueLinearLayerWeightsHead1Trans, 0, 75);

            temp = matOps.transposeMat(Transformer_Implementation.queryLinearLayerWeights_head2, 5, 15);
            Array.Copy(temp, 0, queryLinearLayerWeightsHead2Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.keyLinearLayerWeights_head2, 5, 15);
            Array.Copy(temp, 0, keyLinearLayerWeightsHead2Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.valueLinearLayerWeights_head2, 5, 15);
            Array.Copy(temp, 0, valueLinearLayerWeightsHead2Trans, 0, 75);

            temp = matOps.transposeMat(Transformer_Implementation.queryLinearLayerWeights_head3, 5, 15);
            Array.Copy(temp, 0, queryLinearLayerWeightsHead3Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.keyLinearLayerWeights_head3, 5, 15);
            Array.Copy(temp, 0, keyLinearLayerWeightsHead3Trans, 0, 75);
            temp = matOps.transposeMat(Transformer_Implementation.valueLinearLayerWeights_head3, 5, 15);
            Array.Copy(temp, 0, valueLinearLayerWeightsHead3Trans, 0, 75);

            //multiply with the output of the linear layers for each attention head to get error of the input from conv module
            temp = matOps.matrixMul(Transformer_Implementation.query_head1, queryLinearLayerWeightsHead1Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module1, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.key_head1, keyLinearLayerWeightsHead1Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module2, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.value_head1, valueLinearLayerWeightsHead1Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module3, 0, 1500);

            temp = matOps.matrixMul(Transformer_Implementation.query_head2, queryLinearLayerWeightsHead2Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module4, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.key_head2, keyLinearLayerWeightsHead2Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module5, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.value_head2, valueLinearLayerWeightsHead2Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module6, 0, 1500);

            temp = matOps.matrixMul(Transformer_Implementation.query_head3, queryLinearLayerWeightsHead3Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module7, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.key_head3, keyLinearLayerWeightsHead3Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module8, 0, 1500);
            temp = matOps.matrixMul(Transformer_Implementation.value_head3, valueLinearLayerWeightsHead3Trans, 100, 5, 15);
            Array.Copy(temp, 0, error_of_input_from_conv_module9, 0, 1500);

            //find the average of all the errors
            for(int i = 0; i < 1500; i++)
            {
                avg_error_of_input_from_conv_module[i] = (error_of_input_from_conv_module1[i] +
                                                          error_of_input_from_conv_module2[i] +
                                                          error_of_input_from_conv_module3[i] +
                                                          error_of_input_from_conv_module4[i] +
                                                          error_of_input_from_conv_module5[i] +
                                                          error_of_input_from_conv_module6[i] +
                                                          error_of_input_from_conv_module7[i] +
                                                          error_of_input_from_conv_module8[i] +
                                                          error_of_input_from_conv_module9[i]) / 9;
            }

            for (int i = 0; i < 1500; i++)
            {
                avg_error_of_input_from_conv_module[i] += derivativeOfAttentionBlockOutput[i];
                avg_error_of_input_from_conv_module[i] /= 2;
            }

            //downsize the M dimension to bring down the error matrix to match the dimensions of the conv layer 5 output
            //in theory this is removing the "error" for the temporal encoding which is not needed for backprop of the conv module
            downsizeConvErrorMat();

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\detempencoded_avg_error_mat.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output.WriteLine(detempencoded_avg_error_mat[i].ToString());
                }
                output.Close();
            }

            if(predictorGui.predictorGui1.seluSelect.Checked == true)
            {
                for(int i = 0; i < 1400; i++)
                {
                    detempencoded_avg_error_mat[i] *= 100;
                }
            }

            //transpose as the original conv output was transposed before getting temporally encoded
            temp = matOps.transposeMat(detempencoded_avg_error_mat, 14, 100);
            Array.Copy(temp, 0, detempencoded_avg_error_mat, 14, 100);

            if (predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output = File.AppendText(@"X:\detempencoded_avg_error_mat_transposed.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output.WriteLine(detempencoded_avg_error_mat[i].ToString());
                }
                output.Close();
            }

            if (predictorGui.predictorGui1.preluSelect.Checked == true || predictorGui.predictorGui1.mishSelect.Checked == true)
            {
                double summation;
                int meanVarIdx = 0;
                int normIdx;

                for (int i = 0; i < 1400; i++)
                {
                    //calculate the delta for the beta of add and norm layer
                    deltaConvLayer5NormBeta[i] = detempencoded_avg_error_mat[i];
                }

                //calculate for gamma next
                for (int i = 0; i < 1400; i++)
                {
                    predictorGui.convStructs[0].convLayer5OutputNorm[i] -= predictorGui.convStructs[0].convLayer5OutputNormBeta[i];
                    predictorGui.convStructs[0].convLayer5OutputNorm[i] /= predictorGui.convStructs[0].convLayer5OutputNormGamma[i];
                    deltaConvLayer5NormGamma[i] = detempencoded_avg_error_mat[i] * predictorGui.convStructs[0].convLayer5OutputNorm[i];
                    dxhat[i] = detempencoded_avg_error_mat[i] * predictorGui.convStructs[0].convLayer5OutputNormGamma[i];
                }

                normIdx = 0;
                //calculate divar
                for (int i = 0; i < 1400; i++)
                {
                    if (i % 100 == 0)
                    {
                        divar[meanVarIdx] = 0;
                        for (int j = normIdx; j < normIdx + 100; j++)
                        {
                            divar[meanVarIdx] += dxhat[j] * (predictorGui.convStructs[0].convLayer5Output[j] - predictorGui.convStructs[0].mean[meanVarIdx]);
                        }
                        meanVarIdx++;
                        normIdx += 100;
                    }
                    dxmu[i] = dxhat[i] * (1.0 / Math.Sqrt(predictorGui.convStructs[0].variance[meanVarIdx - 1] + predictorGui.convStructs[0].epsilon));
                }
                meanVarIdx = 0;
                normIdx = 0;

                //calculate dsqrtvar and dvar and dsq and dxmu2
                for (int i = 0; i < 14; i++)
                {
                    dsqrtvar[i] = -1.0 / (predictorGui.convStructs[0].variance[i] + predictorGui.convStructs[0].epsilon) * divar[i];
                    dvar[i] = 0.5 * (1.0 / Math.Sqrt(predictorGui.convStructs[0].variance[i] + predictorGui.convStructs[0].epsilon)) * dsqrtvar[i];
                }

                temp = matOps.matrixMul(dvar, ones, 14, 1, 100);
                Array.Copy(temp, 0, dsq, 0, 1400);

                for (int i = 0; i < 1400; i++)
                {
                    if (i % 100 == 0 && i != 0)
                    {
                        meanVarIdx++;
                    }

                    dxmu2[i] = 2 * (predictorGui.convStructs[0].convLayer5Output[i] - predictorGui.convStructs[0].mean[meanVarIdx]) * dsq[i];
                    dx1[i] = dxmu[i] + dxmu2[i];
                }

                for (int i = 0; i < 14; i++)
                {
                    summation = 0;
                    for (int j = normIdx; j < normIdx + 100; j++)
                    {
                        summation += dxmu[j] + dxmu2[j];
                    }
                    dmu[i] = -1 * summation;
                    normIdx += 100;
                }

                temp = matOps.matrixMul(dmu, ones, 14, 1, 100);
                Array.Copy(temp, 0, dx2, 0, 1400);

                for (int i = 0; i < 1400; i++)
                {
                    detempencoded_avg_error_mat[i] = dx1[i] + dx2[i];
                }

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\divar3.txt");
                    StreamWriter output2 = File.AppendText(@"X:\dsqrtvar3.txt");
                    StreamWriter output3 = File.AppendText(@"X:\dvar3.txt");
                    StreamWriter output4 = File.AppendText(@"X:\dxmu2_3.txt");
                    for (int i = 0; i < 14; i++)
                    {
                        output.WriteLine(divar[i].ToString());
                        output2.WriteLine(dsqrtvar[i].ToString());
                        output3.WriteLine(dvar[i].ToString());
                    }
                    for (int i = 0; i < 1400; i++)
                    {
                        output4.WriteLine(dxmu2[i].ToString());
                    }
                    output.Close();
                    output2.Close();
                    output3.Close();
                    output4.Close();
                }

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\detempencoded_avg_error_mat_after_norm.txt");
                    for (int i = 0; i < 1400; i++)
                    {
                        output.WriteLine(detempencoded_avg_error_mat[i].ToString());
                    }
                    output.Close();
                }
            }
            //calculate deltas for the 14 conv layer 5 kernels
            calculateConvLayerWeightAdjustments(5);

            //calculate deltas for the 14 conv layer 4 kernels
            calculateConvLayerWeightAdjustments(4);

            //calculate deltas for the 14 conv layer 3 kernels
            calculateConvLayerWeightAdjustments(3);

            //calculate deltas for the 14 conv layer 2 kernels
            calculateConvLayerWeightAdjustments(2);

            //calculate deltas for the 14 conv layer 1 kernels
            calculateConvLayerWeightAdjustments(1);
        }

        public void calculateConvLayerWeightAdjustments(int layerNum)
        {  
            if (layerNum == 5)
            {
                double[] transConvOut;
                double[] transDerivative;
                transConvOut = matOps.transposeMat(predictorGui.convStructs[0].convLayer5Output, 14, 100);
                Array.Copy(transConvOut, 0, predictorGui.convStructs[0].convLayer5Output, 0, 1400);

                if (predictorGui.predictorGui1.preluSelect.Checked == true)
                {
                    PReLU_derivative(8);
                }
                else if (predictorGui.predictorGui1.seluSelect.Checked == true)
                {
                    SeLU_derivative(8);
                }
                else if (predictorGui.predictorGui1.mishSelect.Checked == true)
                {
                    Mish_derivative(8);
                }
                for (int i = 0; i < 1400; i++)
                {
                    derivativeOfConvLayerOut[i] *= detempencoded_avg_error_mat[i];
                }

                //find err matrix for next layer updates
                transDerivative = matOps.transposeMat(derivativeOfConvLayerOut, 100, 14);
                double[] errNextLayer = matOps.find_conv_layer_err4(transDerivative);
                Array.Copy(errNextLayer, 0, error_of_convLayer4, 0, 1400);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\error_of_convLayer4.txt");
                    for (int i = 0; i < 1400; i++)
                    {
                        output.WriteLine(error_of_convLayer4[i].ToString());
                    }
                    output.Close();
                }

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\derivativeOfConvLayer5Out.txt");
                    for(int i = 0; i < 1400; i++)
                    {
                        output.WriteLine(derivativeOfConvLayerOut[i].ToString());
                    }
                    output.Close();
                }

                double[] padded_input;
                padded_input = matOps.transposeMat(predictorGui.convStructs[0].convLayer4OutPadded, 14, 116);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\convLayer4OutPadded.txt");
                    for (int i = 0; i < 1624; i++)
                    {
                        output.WriteLine(padded_input[i].ToString());
                    }
                    output.Close();
                }

                Parallel.For(0, 14, (j, state) =>
                {
                    if (j == 0)
                    {
                        int startIdx = 0;
                        deltaConvLayer5Kernel1_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel1_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 1)
                    {
                        int startIdx = 100;
                        deltaConvLayer5Kernel2_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel2_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 2)
                    {
                        int startIdx = 200;
                        deltaConvLayer5Kernel3_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel3_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 3)
                    {
                        int startIdx = 300;
                        deltaConvLayer5Kernel4_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel4_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 4)
                    {
                        int startIdx = 400;
                        deltaConvLayer5Kernel5_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel5_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 5)
                    {
                        int startIdx = 500;
                        deltaConvLayer5Kernel6_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel6_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 6)
                    {
                        int startIdx = 600;
                        deltaConvLayer5Kernel7_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel7_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 7)
                    {
                        int startIdx = 700;
                        deltaConvLayer5Kernel8_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel8_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 8)
                    {
                        int startIdx = 800;
                        deltaConvLayer5Kernel9_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel9_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 9)
                    {
                        int startIdx = 900;
                        deltaConvLayer5Kernel10_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel10_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 10)
                    {
                        int startIdx = 1000;
                        deltaConvLayer5Kernel11_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel11_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 11)
                    {
                        int startIdx = 1100;
                        deltaConvLayer5Kernel12_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel12_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 12)
                    {
                        int startIdx = 1200;
                        deltaConvLayer5Kernel13_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel13_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 13)
                    {
                        int startIdx = 1300;
                        deltaConvLayer5Kernel14_depth1 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer5Kernel14_depth2 = matOps.conv5KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                });
                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\deltaConvLayer5Kernel1_depth1.txt");
                    StreamWriter output2 = File.AppendText(@"X:\deltaConvLayer5Kernel1_depth2.txt");
                    StreamWriter output3 = File.AppendText(@"X:\deltaConvLayer5Kernel14_depth1.txt");
                    StreamWriter output4 = File.AppendText(@"X:\deltaConvLayer5Kernel14_depth2.txt");
                    for (int i = 0; i < 14; i++)
                    {
                        output.WriteLine(deltaConvLayer5Kernel1_depth1[i].ToString());
                        output2.WriteLine(deltaConvLayer5Kernel1_depth2[i].ToString());
                        output3.WriteLine(deltaConvLayer5Kernel14_depth1[i].ToString());
                        output4.WriteLine(deltaConvLayer5Kernel14_depth2[i].ToString());
                    }
                    output.Close();
                    output2.Close();
                    output3.Close();
                    output4.Close();
                }

                for(int i = 0; i < 1400; i++)
                {
                    deltaConvLayer5Biases[i] = derivativeOfConvLayerOut[i];
                    deltaConvLayer5PReLUParams[i] = derivativeOfConvLayerOut[i];
                }
            }
            if (layerNum == 4)
            {
                double[] transConvOut;
                double[] transDerivative;
                transConvOut = matOps.transposeMat(predictorGui.convStructs[0].convLayer4Output, 14, 100);
                Array.Copy(transConvOut, 0, predictorGui.convStructs[0].convLayer4Output, 0, 1400);

                if (predictorGui.predictorGui1.preluSelect.Checked == true)
                {
                    PReLU_derivative(9);
                }
                else if (predictorGui.predictorGui1.seluSelect.Checked == true)
                {
                    SeLU_derivative(9);
                }
                else if (predictorGui.predictorGui1.mishSelect.Checked == true)
                {
                    Mish_derivative(9);
                }

                for (int i = 0; i < 1400; i++)
                {
                    derivativeOfConvLayerOut[i] *= error_of_convLayer4[i];
                }

                //find err matrix for next layer updates
                transDerivative = matOps.transposeMat(derivativeOfConvLayerOut, 100, 14);
                double[] errNextLayer = matOps.find_conv_layer_err3(transDerivative);
                Array.Copy(errNextLayer, 0, error_of_convLayer3, 0, 1400);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\error_of_convLayer3.txt");
                    for (int i = 0; i < 1400; i++)
                    {
                        output.WriteLine(error_of_convLayer3[i].ToString());
                    }
                    output.Close();
                }

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\derivativeOfConvLayer4Out.txt");
                    for (int i = 0; i < 1400; i++)
                    {
                        output.WriteLine(derivativeOfConvLayerOut[i].ToString());
                    }
                    output.Close();
                }

                double[] padded_input;
                padded_input = matOps.transposeMat(predictorGui.convStructs[0].convLayer3OutPadded, 14, 108);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\convLayer3OutPadded.txt");
                    for (int i = 0; i < 1512; i++)
                    {
                        output.WriteLine(padded_input[i].ToString());
                    }
                    output.Close();
                }

                Parallel.For(0, 14, (j, state) =>
                {
                    if (j == 0)
                    {
                        int startIdx = 0;
                        deltaConvLayer4Kernel1_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel1_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 1)
                    {
                        int startIdx = 100;
                        deltaConvLayer4Kernel2_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel2_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 2)
                    {
                        int startIdx = 200;
                        deltaConvLayer4Kernel3_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel3_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 3)
                    {
                        int startIdx = 300;
                        deltaConvLayer4Kernel4_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel4_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 4)
                    {
                        int startIdx = 400;
                        deltaConvLayer4Kernel5_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel5_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 5)
                    {
                        int startIdx = 500;
                        deltaConvLayer4Kernel6_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel6_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 6)
                    {
                        int startIdx = 600;
                        deltaConvLayer4Kernel7_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel7_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 7)
                    {
                        int startIdx = 700;
                        deltaConvLayer4Kernel8_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel8_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 8)
                    {
                        int startIdx = 800;
                        deltaConvLayer4Kernel9_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel9_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 9)
                    {
                        int startIdx = 900;
                        deltaConvLayer4Kernel10_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel10_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 10)
                    {
                        int startIdx = 1000;
                        deltaConvLayer4Kernel11_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel11_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 11)
                    {
                        int startIdx = 1100;
                        deltaConvLayer4Kernel12_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel12_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 12)
                    {
                        int startIdx = 1200;
                        deltaConvLayer4Kernel13_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel13_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 13)
                    {
                        int startIdx = 1300;
                        deltaConvLayer4Kernel14_depth1 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer4Kernel14_depth2 = matOps.conv4KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                });
                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\deltaConvLayer4Kernel1_depth1.txt");
                    StreamWriter output2 = File.AppendText(@"X:\deltaConvLayer4Kernel1_depth2.txt");
                    StreamWriter output3 = File.AppendText(@"X:\deltaConvLayer4Kernel14_depth1.txt");
                    StreamWriter output4 = File.AppendText(@"X:\deltaConvLayer4Kernel14_depth2.txt");
                    for (int i = 0; i < 14; i++)
                    {
                        output.WriteLine(deltaConvLayer4Kernel1_depth1[i].ToString());
                        output2.WriteLine(deltaConvLayer4Kernel1_depth2[i].ToString());
                        output3.WriteLine(deltaConvLayer4Kernel14_depth1[i].ToString());
                        output4.WriteLine(deltaConvLayer4Kernel14_depth2[i].ToString());
                    }
                    output.Close();
                    output2.Close();
                    output3.Close();
                    output4.Close();
                }

                for (int i = 0; i < 1400; i++)
                {
                    deltaConvLayer4Biases[i] = derivativeOfConvLayerOut[i];
                    deltaConvLayer4PReLUParams[i] = derivativeOfConvLayerOut[i];
                }
            }
            if (layerNum == 3)
            {
                double[] transConvOut;
                double[] transDerivative;
                transConvOut = matOps.transposeMat(predictorGui.convStructs[0].convLayer3Output, 14, 100);
                Array.Copy(transConvOut, 0, predictorGui.convStructs[0].convLayer3Output, 0, 1400);

                if (predictorGui.predictorGui1.preluSelect.Checked == true)
                {
                    PReLU_derivative(11);
                }
                else if (predictorGui.predictorGui1.seluSelect.Checked == true)
                {
                    SeLU_derivative(11);
                }
                else if (predictorGui.predictorGui1.mishSelect.Checked == true)
                {
                    Mish_derivative(11);
                }

                for (int i = 0; i < 1400; i++)
                {
                    derivativeOfConvLayerOut[i] *= error_of_convLayer3[i];
                }

                //find err matrix for next layer updates
                transDerivative = matOps.transposeMat(derivativeOfConvLayerOut, 100, 14);
                double[] errNextLayer = matOps.find_conv_layer_err2(transDerivative);
                Array.Copy(errNextLayer, 0, error_of_convLayer2, 0, 1400);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\error_of_convLayer2.txt");
                    for (int i = 0; i < 1400; i++)
                    {
                        output.WriteLine(error_of_convLayer2[i].ToString());
                    }
                    output.Close();
                }

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\derivativeOfConvLayer3Out.txt");
                    for (int i = 0; i < 1400; i++)
                    {
                        output.WriteLine(derivativeOfConvLayerOut[i].ToString());
                    }
                    output.Close();
                }

                double[] padded_input;
                padded_input = matOps.transposeMat(predictorGui.convStructs[0].convLayer2OutPadded, 14, 104);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\convLayer3OutPadded.txt");
                    for (int i = 0; i < 1456; i++)
                    {
                        output.WriteLine(padded_input[i].ToString());
                    }
                    output.Close();
                }

                Parallel.For(0, 14, (j, state) =>
                {
                    if (j == 0)
                    {
                        int startIdx = 0;
                        deltaConvLayer3Kernel1_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel1_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 1)
                    {
                        int startIdx = 100;
                        deltaConvLayer3Kernel2_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel2_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 2)
                    {
                        int startIdx = 200;
                        deltaConvLayer3Kernel3_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel3_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 3)
                    {
                        int startIdx = 300;
                        deltaConvLayer3Kernel4_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel4_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 4)
                    {
                        int startIdx = 400;
                        deltaConvLayer3Kernel5_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel5_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 5)
                    {
                        int startIdx = 500;
                        deltaConvLayer3Kernel6_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel6_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 6)
                    {
                        int startIdx = 600;
                        deltaConvLayer3Kernel7_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel7_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 7)
                    {
                        int startIdx = 700;
                        deltaConvLayer3Kernel8_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel8_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 8)
                    {
                        int startIdx = 800;
                        deltaConvLayer3Kernel9_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel9_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 9)
                    {
                        int startIdx = 900;
                        deltaConvLayer3Kernel10_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel10_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 10)
                    {
                        int startIdx = 1000;
                        deltaConvLayer3Kernel11_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel11_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 11)
                    {
                        int startIdx = 1100;
                        deltaConvLayer3Kernel12_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel12_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 12)
                    {
                        int startIdx = 1200;
                        deltaConvLayer3Kernel13_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel13_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 13)
                    {
                        int startIdx = 1300;
                        deltaConvLayer3Kernel14_depth1 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer3Kernel14_depth2 = matOps.conv3KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                });
                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\deltaConvLayer3Kernel1_depth1.txt");
                    StreamWriter output2 = File.AppendText(@"X:\deltaConvLayer3Kernel1_depth2.txt");
                    StreamWriter output3 = File.AppendText(@"X:\deltaConvLayer3Kernel14_depth1.txt");
                    StreamWriter output4 = File.AppendText(@"X:\deltaConvLayer3Kernel14_depth2.txt");
                    for (int i = 0; i < 14; i++)
                    {
                        output.WriteLine(deltaConvLayer3Kernel1_depth1[i].ToString());
                        output2.WriteLine(deltaConvLayer3Kernel1_depth2[i].ToString());
                        output3.WriteLine(deltaConvLayer3Kernel14_depth1[i].ToString());
                        output4.WriteLine(deltaConvLayer3Kernel14_depth2[i].ToString());
                    }
                    output.Close();
                    output2.Close();
                    output3.Close();
                    output4.Close();
                }

                for (int i = 0; i < 1400; i++)
                {
                    deltaConvLayer3Biases[i] = derivativeOfConvLayerOut[i];
                    deltaConvLayer3PReLUParams[i] = derivativeOfConvLayerOut[i];
                }
            }
            if (layerNum == 2)
            {
                double[] transConvOut;
                double[] transDerivative;
                transConvOut = matOps.transposeMat(predictorGui.convStructs[0].convLayer2Output, 14, 100);
                Array.Copy(transConvOut, 0, predictorGui.convStructs[0].convLayer2Output, 0, 1400);

                if (predictorGui.predictorGui1.preluSelect.Checked == true)
                {
                    PReLU_derivative(12);
                }
                else if (predictorGui.predictorGui1.seluSelect.Checked == true)
                {
                    SeLU_derivative(12);
                }
                else if (predictorGui.predictorGui1.mishSelect.Checked == true)
                {
                    Mish_derivative(12);
                }

                for (int i = 0; i < 1400; i++)
                {
                    derivativeOfConvLayerOut[i] *= error_of_convLayer2[i];
                }

                //find err matrix for next layer updates
                transDerivative = matOps.transposeMat(derivativeOfConvLayerOut, 100, 14);
                double[] errNextLayer = matOps.find_conv_layer_err1(transDerivative);
                Array.Copy(errNextLayer, 0, error_of_convLayer1, 0, 1400);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\error_of_convLayer1.txt");
                    for (int i = 0; i < 1400; i++)
                    {
                        output.WriteLine(error_of_convLayer1[i].ToString());
                    }
                    output.Close();
                }

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\derivativeOfConvLayer2Out.txt");
                    for (int i = 0; i < 1400; i++)
                    {
                        output.WriteLine(derivativeOfConvLayerOut[i].ToString());
                    }
                    output.Close();
                }

                double[] padded_input;
                padded_input = matOps.transposeMat(predictorGui.convStructs[0].convLayer1OutPadded, 14, 102);

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\convLayer1OutPadded.txt");
                    for (int i = 0; i < 1428; i++)
                    {
                        output.WriteLine(padded_input[i].ToString());
                    }
                    output.Close();
                }

                Parallel.For(0, 14, (j, state) =>
                {
                    if (j == 0)
                    {
                        int startIdx = 0;
                        deltaConvLayer2Kernel1_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel1_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 1)
                    {
                        int startIdx = 100;
                        deltaConvLayer2Kernel2_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel2_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 2)
                    {
                        int startIdx = 200;
                        deltaConvLayer2Kernel3_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel3_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 3)
                    {
                        int startIdx = 300;
                        deltaConvLayer2Kernel4_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel4_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 4)
                    {
                        int startIdx = 400;
                        deltaConvLayer2Kernel5_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel5_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 5)
                    {
                        int startIdx = 500;
                        deltaConvLayer2Kernel6_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel6_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 6)
                    {
                        int startIdx = 600;
                        deltaConvLayer2Kernel7_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel7_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 7)
                    {
                        int startIdx = 700;
                        deltaConvLayer2Kernel8_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel8_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 8)
                    {
                        int startIdx = 800;
                        deltaConvLayer2Kernel9_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel9_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 9)
                    {
                        int startIdx = 900;
                        deltaConvLayer2Kernel10_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel10_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 10)
                    {
                        int startIdx = 1000;
                        deltaConvLayer2Kernel11_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel11_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 11)
                    {
                        int startIdx = 1100;
                        deltaConvLayer2Kernel12_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel12_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 12)
                    {
                        int startIdx = 1200;
                        deltaConvLayer2Kernel13_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel13_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                    else if (j == 13)
                    {
                        int startIdx = 1300;
                        deltaConvLayer2Kernel14_depth1 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, true);
                        deltaConvLayer2Kernel14_depth2 = matOps.conv2KernelBackProp(padded_input, derivativeOfConvLayerOut, startIdx, false);
                    }
                });

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\deltaConvLayer2Kernel1_depth1.txt");
                    StreamWriter output2 = File.AppendText(@"X:\deltaConvLayer2Kernel1_depth2.txt");
                    StreamWriter output3 = File.AppendText(@"X:\deltaConvLayer2Kernel14_depth1.txt");
                    StreamWriter output4 = File.AppendText(@"X:\deltaConvLayer2Kernel14_depth2.txt");
                    for (int i = 0; i < 14; i++)
                    {
                        output.WriteLine(deltaConvLayer2Kernel1_depth1[i].ToString());
                        output2.WriteLine(deltaConvLayer2Kernel1_depth2[i].ToString());
                        output3.WriteLine(deltaConvLayer2Kernel14_depth1[i].ToString());
                        output4.WriteLine(deltaConvLayer2Kernel14_depth2[i].ToString());
                    }
                    output.Close();
                    output2.Close();
                    output3.Close();
                    output4.Close();
                }

                for (int i = 0; i < 1400; i++)
                {
                    deltaConvLayer2Biases[i] = derivativeOfConvLayerOut[i];
                    deltaConvLayer2PReLUParams[i] = derivativeOfConvLayerOut[i];
                }
            }
            if (layerNum == 1)
            {
                double[] transConvOut;

                transConvOut = matOps.transposeMat(predictorGui.convStructs[0].convLayer1Output, 14, 100);
                Array.Copy(transConvOut, 0, predictorGui.convStructs[0].convLayer1Output, 0, 1400);

                if (predictorGui.predictorGui1.preluSelect.Checked == true)
                {
                    PReLU_derivative(13);
                }
                else if (predictorGui.predictorGui1.seluSelect.Checked == true)
                {
                    SeLU_derivative(13);
                }
                else if (predictorGui.predictorGui1.mishSelect.Checked == true)
                {
                    Mish_derivative(13);
                }

                for (int i = 0; i < 1400; i++)
                {
                    derivativeOfConvLayerOut[i] *= error_of_convLayer1[i];
                }

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\derivativeOfConvLayer1Out.txt");
                    for (int i = 0; i < 1400; i++)
                    {
                        output.WriteLine(derivativeOfConvLayerOut[i].ToString());
                    }
                    output.Close();
                }

                double[] padded_inputPrices = new double[3232];
                double[] padded_inputSizes = new double[3232];

                int rowIdx = 0;

                for(int i = 0; i < 101; i++)
                {
                    Array.Copy(predictorGui.eventsArray[i].prices, 0, padded_inputPrices, rowIdx, 32);
                    Array.Copy(predictorGui.eventsArray[i].sizes, 0, padded_inputSizes, rowIdx, 32);
                    rowIdx += 32;
                }

                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\padded_inputPrices.txt");
                    StreamWriter output2 = File.AppendText(@"X:\padded_inputSizes.txt");
                    for (int i = 0; i < 3232; i++)
                    {
                        output.WriteLine(padded_inputPrices[i].ToString());
                        output2.WriteLine(padded_inputSizes[i].ToString());
                    }
                    output.Close();
                    output2.Close();
                }

                padded_inputPrices = matOps.transposeMat(padded_inputPrices, 32, 101);
                padded_inputSizes = matOps.transposeMat(padded_inputSizes, 32, 101);

                if(predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\padded_inputPrices_transposed.txt");
                    StreamWriter output2 = File.AppendText(@"X:\padded_inputSizes_transposed.txt");
                    for(int i = 0; i < 3232; i++)
                    {
                        output.WriteLine(padded_inputPrices[i].ToString());
                        output2.WriteLine(padded_inputSizes[i].ToString());
                    }
                    output.Close();
                    output2.Close();
                }

                Parallel.For(0, 14, (j, state) =>
                {
                    if (j == 0)
                    {
                        int startIdx = 0;
                        deltaConvLayer1Kernel1_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel1_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel1_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel1_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 1)
                    {
                        int startIdx = 100;
                        deltaConvLayer1Kernel2_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel2_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel2_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel2_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 2)
                    {
                        int startIdx = 200;
                        deltaConvLayer1Kernel3_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel3_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel3_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel3_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 3)
                    {
                        int startIdx = 300;
                        deltaConvLayer1Kernel4_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel4_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel4_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel4_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 4)
                    {
                        int startIdx = 400;
                        deltaConvLayer1Kernel5_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel5_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel5_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel5_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 5)
                    {
                        int startIdx = 500;
                        deltaConvLayer1Kernel6_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel6_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel6_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel6_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 6)
                    {
                        int startIdx = 600;
                        deltaConvLayer1Kernel7_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel7_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel7_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel7_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 7)
                    {
                        int startIdx = 700;
                        deltaConvLayer1Kernel8_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel8_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel8_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel8_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 8)
                    {
                        int startIdx = 800;
                        deltaConvLayer1Kernel9_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel9_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel9_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel9_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 9)
                    {
                        int startIdx = 900;
                        deltaConvLayer1Kernel10_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel10_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel10_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel10_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 10)
                    {
                        int startIdx = 1000;
                        deltaConvLayer1Kernel11_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel11_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel11_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel11_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 11)
                    {
                        int startIdx = 1100;
                        deltaConvLayer1Kernel12_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel12_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel12_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel12_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 12)
                    {
                        int startIdx = 1200;
                        deltaConvLayer1Kernel13_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel13_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel13_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel13_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                    else if (j == 13)
                    {
                        int startIdx = 1300;
                        deltaConvLayer1Kernel14_depth1 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 1);
                        deltaConvLayer1Kernel14_depth2 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 2);
                        deltaConvLayer1Kernel14_depth3 = matOps.conv1KernelBackProp(padded_inputPrices, derivativeOfConvLayerOut, startIdx, 3);
                        deltaConvLayer1Kernel14_depth4 = matOps.conv1KernelBackProp(padded_inputSizes, derivativeOfConvLayerOut, startIdx, 4);
                    }
                });
                if (predictorGui.predictorGui1.enableOutputs.Checked == true)
                {
                    StreamWriter output = File.AppendText(@"X:\deltaConvLayer1Kernel1_depth1.txt");
                    StreamWriter output2 = File.AppendText(@"X:\deltaConvLayer1Kernel1_depth2.txt");
                    StreamWriter output3 = File.AppendText(@"X:\deltaConvLayer1Kernel14_depth1.txt");
                    StreamWriter output4 = File.AppendText(@"X:\deltaConvLayer1Kernel14_depth2.txt");
                    for (int i = 0; i < 32; i++)
                    {
                        output.WriteLine(deltaConvLayer1Kernel1_depth1[i].ToString());
                        output2.WriteLine(deltaConvLayer1Kernel1_depth2[i].ToString());
                        output3.WriteLine(deltaConvLayer1Kernel14_depth1[i].ToString());
                        output4.WriteLine(deltaConvLayer1Kernel14_depth2[i].ToString());
                    }
                    output.Close();
                    output2.Close();
                    output3.Close();
                    output4.Close();
                }

                for (int i = 0; i < 1400; i++)
                {
                    deltaConvLayer1Biases[i] = derivativeOfConvLayerOut[i];
                    deltaConvLayer1PReLUParams[i] = derivativeOfConvLayerOut[i];
                }
            }
        }

        public void downsizeConvErrorMat()
        {
            int j = 0;
            for(int i = 0; i < 1400; i++)
            {
                if(i % 14 == 0 && i != 0)
                {
                    j++;  
                }
                Array.Copy(avg_error_of_input_from_conv_module, j, detempencoded_avg_error_mat, i, 1);
                j++;
            }

            if(predictorGui.predictorGui1.enableOutputs.Checked == true)
            {
                StreamWriter output2 = File.AppendText(@"X:\avg_error_of_input_from_conv_module.txt");
                for(int i = 0; i < 1500; i++)
                {
                    output2.WriteLine(avg_error_of_input_from_conv_module[i].ToString());
                }
                output2.Close();
            }
        }

        public void Mish_derivative(int layerNum)
        {
            if (layerNum == 1)
            {
                double[] x = new double[64];
                
                for (int i = 0; i < 64; i++)
                {
                    x[i] = firstLayerOutCpy[i];
                    derivativeOfFirstLayerOut[i] = Math.Exp(x[i]) * ((4 * (x[i] + 1)) + (4 * Math.Exp(2 * x[i])) + Math.Exp(3 * x[i]) +
                        (Math.Exp(x[i]) * (4 * x[i] + 6))) / Math.Pow((2 * Math.Exp(x[i])) + Math.Exp(2 * x[i]) + 2, 2);
                }
            }
            if(layerNum == 2)
            {
                for (int i = 0; i < 6000; i++)
                {
                    derivativeOfAffine1Output[i] = Math.Exp(Transformer_Implementation.affineIntermediateRes[i]) * (4 * (Transformer_Implementation.affineIntermediateRes[i] + 1) + 
                        (4 * Math.Exp(2 * Transformer_Implementation.affineIntermediateRes[i])) +
                        Math.Exp(3 * Transformer_Implementation.affineIntermediateRes[i]) + Math.Exp(Transformer_Implementation.affineIntermediateRes[i]) * 
                        (4 * Transformer_Implementation.affineIntermediateRes[i] + 6)) /
                        Math.Pow(2 * Math.Exp(Transformer_Implementation.affineIntermediateRes[i]) + Math.Exp(2 * Transformer_Implementation.affineIntermediateRes[i]) + 2, 2);
                }
            }
            if (layerNum == 3)
            {
                for (int i = 0; i < 6000; i++)
                {
                    derivativeOfAffine1Output[i] = Math.Exp(Transformer_Implementation.affineIntermediateRes2[i]) * (4 * (Transformer_Implementation.affineIntermediateRes2[i] + 1) +
                        (4 * Math.Exp(2 * Transformer_Implementation.affineIntermediateRes2[i])) +
                        Math.Exp(3 * Transformer_Implementation.affineIntermediateRes2[i]) + Math.Exp(Transformer_Implementation.affineIntermediateRes2[i]) *
                        (4 * Transformer_Implementation.affineIntermediateRes2[i] + 6)) /
                        Math.Pow(2 * Math.Exp(Transformer_Implementation.affineIntermediateRes2[i]) + Math.Exp(2 * Transformer_Implementation.affineIntermediateRes2[i]) + 2, 2);
                }
            }
            if (layerNum == 13)
            {
                for (int i = 0; i < 1400; i++)
                {
                    derivativeOfConvLayerOut[i] = Math.Exp(predictorGui.convStructs[0].convLayer1Output[i]) * (4 * (predictorGui.convStructs[0].convLayer1Output[i] + 1) +
                        (4 * Math.Exp(2 * predictorGui.convStructs[0].convLayer1Output[i])) +
                        Math.Exp(3 * predictorGui.convStructs[0].convLayer1Output[i]) + Math.Exp(predictorGui.convStructs[0].convLayer1Output[i]) *
                        (4 * predictorGui.convStructs[0].convLayer1Output[i] + 6)) /
                        Math.Pow(2 * Math.Exp(predictorGui.convStructs[0].convLayer1Output[i]) + Math.Exp(2 * predictorGui.convStructs[0].convLayer1Output[i]) + 2, 2);
                }
            }
            if (layerNum == 12)
            {
                for (int i = 0; i < 1400; i++)
                {
                    derivativeOfConvLayerOut[i] = Math.Exp(predictorGui.convStructs[0].convLayer2Output[i]) * (4 * (predictorGui.convStructs[0].convLayer2Output[i] + 1) +
                        (4 * Math.Exp(2 * predictorGui.convStructs[0].convLayer2Output[i])) +
                        Math.Exp(3 * predictorGui.convStructs[0].convLayer2Output[i]) + Math.Exp(predictorGui.convStructs[0].convLayer2Output[i]) *
                        (4 * predictorGui.convStructs[0].convLayer2Output[i] + 6)) /
                        Math.Pow(2 * Math.Exp(predictorGui.convStructs[0].convLayer2Output[i]) + Math.Exp(2 * predictorGui.convStructs[0].convLayer2Output[i]) + 2, 2);
                }
            }
            if (layerNum == 11)
            {
                for (int i = 0; i < 1400; i++)
                {
                    derivativeOfConvLayerOut[i] = Math.Exp(predictorGui.convStructs[0].convLayer3Output[i]) * (4 * (predictorGui.convStructs[0].convLayer3Output[i] + 1) +
                        (4 * Math.Exp(2 * predictorGui.convStructs[0].convLayer3Output[i])) +
                        Math.Exp(3 * predictorGui.convStructs[0].convLayer3Output[i]) + Math.Exp(predictorGui.convStructs[0].convLayer3Output[i]) *
                        (4 * predictorGui.convStructs[0].convLayer3Output[i] + 6)) /
                        Math.Pow(2 * Math.Exp(predictorGui.convStructs[0].convLayer3Output[i]) + Math.Exp(2 * predictorGui.convStructs[0].convLayer3Output[i]) + 2, 2);
                }
            }
            if (layerNum == 9)
            {
                for (int i = 0; i < 1400; i++)
                {
                    derivativeOfConvLayerOut[i] = Math.Exp(predictorGui.convStructs[0].convLayer4Output[i]) * (4 * (predictorGui.convStructs[0].convLayer4Output[i] + 1) +
                        (4 * Math.Exp(2 * predictorGui.convStructs[0].convLayer4Output[i])) +
                        Math.Exp(3 * predictorGui.convStructs[0].convLayer4Output[i]) + Math.Exp(predictorGui.convStructs[0].convLayer4Output[i]) *
                        (4 * predictorGui.convStructs[0].convLayer4Output[i] + 6)) /
                        Math.Pow(2 * Math.Exp(predictorGui.convStructs[0].convLayer4Output[i]) + Math.Exp(2 * predictorGui.convStructs[0].convLayer4Output[i]) + 2, 2);
                }
            }
            if (layerNum == 8)
            {
                for (int i = 0; i < 1400; i++)
                {
                    derivativeOfConvLayerOut[i] = Math.Exp(predictorGui.convStructs[0].convLayer5Output[i]) * (4 * (predictorGui.convStructs[0].convLayer5Output[i] + 1) +
                        (4 * Math.Exp(2 * predictorGui.convStructs[0].convLayer5Output[i])) +
                        Math.Exp(3 * predictorGui.convStructs[0].convLayer5Output[i]) + Math.Exp(predictorGui.convStructs[0].convLayer5Output[i]) *
                        (4 * predictorGui.convStructs[0].convLayer5Output[i] + 6)) /
                        Math.Pow(2 * Math.Exp(predictorGui.convStructs[0].convLayer5Output[i]) + Math.Exp(2 * predictorGui.convStructs[0].convLayer5Output[i]) + 2, 2);
                }
            }
        }

        public void SeLU_derivative(int layerNum)
        {
            if (layerNum == 13)
            {
                for(int i = 0; i < 1400; i++)
                {
                    if(predictorGui.convStructs[0].convLayer1Output[i] > 0)
                    {
                        derivativeOfConvLayerOut[i] = convLayer.SeLU_lambda;
                    }
                    else
                    {
                        derivativeOfConvLayerOut[i] = convLayer.SeLU_lambda * convLayer.SeLU_alpha * Math.Exp(predictorGui.convStructs[0].convLayer1Output[i]);
                    }
                }
            }
            if (layerNum == 12)
            {
                for (int i = 0; i < 1400; i++)
                {
                    if (predictorGui.convStructs[0].convLayer2Output[i] > 0)
                    {
                        derivativeOfConvLayerOut[i] = convLayer.SeLU_lambda;
                    }
                    else
                    {
                        derivativeOfConvLayerOut[i] = convLayer.SeLU_lambda * convLayer.SeLU_alpha * Math.Exp(predictorGui.convStructs[0].convLayer2Output[i]);
                    }
                }
            }
            if (layerNum == 11)
            {
                for (int i = 0; i < 1400; i++)
                {
                    if (predictorGui.convStructs[0].convLayer3Output[i] > 0)
                    {
                        derivativeOfConvLayerOut[i] = convLayer.SeLU_lambda;
                    }
                    else
                    {
                        derivativeOfConvLayerOut[i] = convLayer.SeLU_lambda * convLayer.SeLU_alpha * Math.Exp(predictorGui.convStructs[0].convLayer3Output[i]);
                    }
                }
            }
            if (layerNum == 9)
            {
                for (int i = 0; i < 1400; i++)
                {
                    if (predictorGui.convStructs[0].convLayer4Output[i] > 0)
                    {
                        derivativeOfConvLayerOut[i] = convLayer.SeLU_lambda;
                    }
                    else
                    {
                        derivativeOfConvLayerOut[i] = convLayer.SeLU_lambda * convLayer.SeLU_alpha * Math.Exp(predictorGui.convStructs[0].convLayer4Output[i]);
                    }
                }
            }
            if (layerNum == 8)
            {
                for (int i = 0; i < 1400; i++)
                {
                    if (predictorGui.convStructs[0].convLayer5Output[i] > 0)
                    {
                        derivativeOfConvLayerOut[i] = convLayer.SeLU_lambda;
                    }
                    else
                    {
                        derivativeOfConvLayerOut[i] = convLayer.SeLU_lambda * convLayer.SeLU_alpha * Math.Exp(predictorGui.convStructs[0].convLayer5Output[i]);
                    }
                }
            }
        }

        public void PReLU_derivative(int layerNum)
        {
            if (layerNum == 13)
            {
                for (int i = 0; i < 1400; i++)
                {
                    if (predictorGui.convStructs[0].convLayer1Output[i] + predictorGui.convStructs[0].convLayer1Bias[i] >= 0)
                    {
                        derivativeOfConvLayerOut[i] = 1;
                    }
                    else
                    {
                        derivativeOfConvLayerOut[i] = /*predictorGui.convStructs[0].convLayer1PReLUParam[i]*/0;
                    }
                }
            }
            if (layerNum == 12)
            {
                for (int i = 0; i < 1400; i++)
                {
                    if (predictorGui.convStructs[0].convLayer2Output[i] + predictorGui.convStructs[0].convLayer2Bias[i] >= 0)
                    {
                        derivativeOfConvLayerOut[i] = 1;
                    }
                    else
                    {
                        derivativeOfConvLayerOut[i] = /*predictorGui.convStructs[0].convLayer2PReLUParam[i]*/0;
                    }
                }
            }
            if (layerNum == 11)
            {
                for (int i = 0; i < 1400; i++)
                {
                    if (predictorGui.convStructs[0].convLayer3Output[i] + predictorGui.convStructs[0].convLayer3Bias[i] >= 0)
                    {
                        derivativeOfConvLayerOut[i] = 1;
                    }
                    else
                    {
                        derivativeOfConvLayerOut[i] = /*predictorGui.convStructs[0].convLayer3PReLUParam[i]*/0;
                    }
                }
            }
            if (layerNum == 10)
            {
                for (int i = 0; i < 1500; i++)
                {
                    if (Transformer_Implementation.residualConnectionOutputNormCpy[i] >= 0)
                    {
                        derivativeOfAttentionBlockOutput[i] = 1;
                    }
                    else
                    {
                        derivativeOfAttentionBlockOutput[i] = 0;
                    }
                }
            }
            if (layerNum == 9)
            {
                for (int i = 0; i < 1400; i++)
                {
                    if (predictorGui.convStructs[0].convLayer4Output[i] + predictorGui.convStructs[0].convLayer4Bias[i] >= 0)
                    {
                        derivativeOfConvLayerOut[i] = 1;
                    }
                    else
                    {
                        derivativeOfConvLayerOut[i] = /*predictorGui.convStructs[0].convLayer4PReLUParam[i]*/0;
                    }
                }
            }
            if (layerNum == 8)
            {
                for (int i = 0; i < 1400; i++)
                {
                    if (predictorGui.convStructs[0].convLayer5Output[i] + predictorGui.convStructs[0].convLayer5Bias[i] >= 0)
                    {
                        derivativeOfConvLayerOut[i] = 1;
                    }
                    else
                    {
                        derivativeOfConvLayerOut[i] = /*predictorGui.convStructs[0].convLayer5PReLUParam[i]*/0;
                    }
                }
            }
            if (layerNum == 7)
            {
                for (int i = 0; i < 1500; i++)
                {
                    if (Transformer_Implementation.residualConnectionOutputNorm[i] >= 0)
                    {
                        derivativeOfAttentionBlockOutput[i] = 1;
                    }
                    else
                    {
                        derivativeOfAttentionBlockOutput[i] = 0;
                    }
                }
            }
            if (layerNum == 6)
            {
                for (int i = 0; i < 6000; i++)
                {
                    if (Transformer_Implementation.affineIntermediateRes3[i] >= 0)
                    {
                        derivativeOfAffine1Output[i] = 1;
                    }
                    else
                    {
                        derivativeOfAffine1Output[i] = 0;
                    }
                }
            }
            if (layerNum == 5)
            {
                for (int i = 0; i < 6000; i++)
                {
                    if (Transformer_Implementation.affineIntermediateRes[i] >= 0)
                    {
                        derivativeOfAffine1Output[i] = 1;
                    }
                    else
                    {
                        derivativeOfAffine1Output[i] = /*Transformer_Implementation.transPReLUParam[i]*/0;
                    }
                }
            }
            if (layerNum == 4)
            {
                for (int i = 0; i < 6000; i++)
                {
                    if (Transformer_Implementation.affineIntermediateRes2[i] > 0)
                    {
                        derivativeOfAffine1Output[i] = 1;
                    }
                    else
                    {
                        derivativeOfAffine1Output[i] = /*Transformer_Implementation.transPReLUParam[i]*/0;
                    }
                }
            }
            if (layerNum == 3)
            {
                for(int i = 0; i < 1500; i++)
                {
                    if (predictorGui.transformerBlock2Output[i] >= 0)
                    {
                        derivativeOfAffine2Output[i] = 1;
                    }
                    else
                    {
                        derivativeOfAffine2Output[i] = 0;
                    }
                }
            }
            if (layerNum == 2)
            {
                for(int i = 0; i < 3; i++)
                {
                    if(MLP.secondLayerOut[i] >= 0)
                    {
                        derivativeOfSecondLayerOut[i] = 1;
                    }
                    else
                    {
                        derivativeOfSecondLayerOut[i] = /*MLP.mlpLayer2PReLUParam[i]*/0;
                    }
                }
            }

            if(layerNum == 1)
            {
                for (int i = 0; i < 64; i++)
                {
                    if (MLP.firstLayerOut[i] >= 0)
                    {
                        derivativeOfFirstLayerOut[i] = 1;
                    }
                    else
                    {
                        derivativeOfFirstLayerOut[i] = /*MLP.mlpLayer1PReLUParam[i]*/0;
                    }
                }
            }
        }

        public void priceDirectionality(int exIdx, bool training)
        {
            if (predictorGui.predictorGui1.buildTrdata.Checked == true && training == false)
            {
                if(File.Exists(@"X:\trainingData\trainingTensorExample" + exIdx.ToString() + ".gt.txt"))
                {
                    File.Delete(@"X:\trainingData\trainingTensorExample" + exIdx.ToString() + ".gt.txt");
                }
                StreamWriter output;
                output = File.AppendText(@"X:\trainingData\trainingTensorExample" + exIdx.ToString() + ".gt.txt");

                exIdx = 0;
                //check for upward direction
                if (pricePercentChange > pricePercentChangeCmp)
                {
                    actualOutcomes[exIdx] = 1.0;
                    output.WriteLine(actualOutcomes[exIdx].ToString());
                    exIdx++;
                    actualOutcomes[exIdx] = 0.0;
                    output.WriteLine(actualOutcomes[exIdx].ToString());
                    exIdx++;
                    actualOutcomes[exIdx] = 0.0;
                    output.WriteLine(actualOutcomes[exIdx].ToString());
                }
                //check for downward direction
                else if (pricePercentChange < -pricePercentChangeCmp)
                {
                    actualOutcomes[exIdx] = 0.0;
                    output.WriteLine(actualOutcomes[exIdx].ToString());
                    exIdx++;
                    actualOutcomes[exIdx] = 0.0;
                    output.WriteLine(actualOutcomes[exIdx].ToString());
                    exIdx++;
                    actualOutcomes[exIdx] = 1.0;
                    output.WriteLine(actualOutcomes[exIdx].ToString());
                }
                //check for flat direction
                else
                {
                    actualOutcomes[exIdx] = 0.0;
                    output.WriteLine(actualOutcomes[exIdx].ToString());
                    exIdx++;
                    actualOutcomes[exIdx] = 1.0;
                    output.WriteLine(actualOutcomes[exIdx].ToString());
                    exIdx++;
                    actualOutcomes[exIdx] = 0.0;
                    output.WriteLine(actualOutcomes[exIdx].ToString());
                }
                output.Close();
            }
            else if(predictorGui.predictorGui1.buildTrdata.Checked == false && training == false)
            {
                exIdx = 0;
                //check for upward direction
                if (pricePercentChange > pricePercentChangeCmp)
                {
                    actualOutcomes[exIdx] = 1.0;
                    exIdx++;
                    actualOutcomes[exIdx] = 0.0;
                    exIdx++;
                    actualOutcomes[exIdx] = 0.0;
                }
                //check for downward direction
                else if (pricePercentChange < -pricePercentChangeCmp)
                {
                    actualOutcomes[exIdx] = 0.0;
                    exIdx++;
                    actualOutcomes[exIdx] = 0.0;
                    exIdx++;
                    actualOutcomes[exIdx] = 1.0;
                }
                //check for flat direction
                else
                {
                    actualOutcomes[exIdx] = 0.0;
                    exIdx++;
                    actualOutcomes[exIdx] = 1.0;
                    exIdx++;
                    actualOutcomes[exIdx] = 0.0;
                }
            }
            else if(predictorGui.predictorGui1.buildTrdata.Checked == false && training == true)
            {
                string[] inputLines;
                inputLines = File.ReadAllLines(@"X:\trainingData\trainingTensorExample" + exIdx.ToString() + ".gt.txt");
                actualOutcomes[0] = Convert.ToInt32(inputLines[0]);
                actualOutcomes[1] = Convert.ToInt32(inputLines[1]);
                actualOutcomes[2] = Convert.ToInt32(inputLines[2]);
            }
        }

        public void buildTrainingDataExamples(int exIdx, inputTensor prevTensor)
        {
            StreamWriter output = File.AppendText(@"X:\trainingData\trainingTensor" + exIdx.ToString() + ".ex.txt");
            //we are interested in comparing the prediction of the previous tensor to the one hot encoding
            //provided by the priceDirectionality function
            //i.e. trainingTensorExample0.txt and oneHotEncodingExample0.txt constitutes one training example
            for(int i = 0; i < 3200; i++)
            {
                output.WriteLine(prevTensor.price[i].ToString() + " " + prevTensor.size[i].ToString());
            }
            output.Close();
        }

        public void createFileList(int exIdx)
        {
            int[] arr = Enumerable.Range(0, exIdx + 1).OrderBy(c => predictorGui.rand.Next()).ToArray();
            ArrayList priceSet = new ArrayList();
            ArrayList sizeSet = new ArrayList();
            ArrayList normPrice = new ArrayList();
            ArrayList normSize = new ArrayList();
            string[] arr2;
            string[] arr3;
            double meanOfPrices = 0;
            double meanOfSizes = 0;
            double stdPrices = 0;
            double stdSizes = 0;

            for (int i = 0; i <= exIdx; i++)
            {
                int lineCount = 0;
                listOfTrainingExamples.Add(arr[i]);
                string[] pricesFileName;
                string[] sizesFileName;

                if (!File.Exists(@"X:\min_max_scaling_values.txt"))
                {
                    string[] lines = File.ReadAllLines(@"X:\trainingData\trainingTensor" + arr[i].ToString() + ".ex.txt");

                    pricesFileName = File.ReadAllLines(@"X:\trainingData\entireDayPrices.forEx" + arr[i].ToString() + ".txt");
                    sizesFileName = File.ReadAllLines(@"X:\trainingData\entireDaySizes.forEx" + arr[i].ToString() + ".txt");

                    arr2 = File.ReadAllLines(@"X:\trainingData\" + pricesFileName[0]);
                    arr3 = File.ReadAllLines(@"X:\trainingData\" + sizesFileName[0]);

                    meanOfPrices = 0;
                    meanOfSizes = 0;
                    stdPrices = 0;
                    stdSizes = 0;

                    for (int j = 0; j < arr2.Length; j++)
                    {
                        meanOfPrices += Convert.ToDouble(arr2[j]) / arr2.Length;
                        meanOfSizes += Convert.ToDouble(arr3[j]) / arr3.Length;
                    }
                    for (int k = 0; k < arr2.Length; k++)
                    {
                        stdPrices += Math.Pow(Convert.ToDouble(arr2[k]) - meanOfPrices, 2) / arr2.Length;
                        stdSizes += Math.Pow(Convert.ToDouble(arr3[k]) - meanOfSizes, 2) / arr3.Length;
                    }

                    stdPrices = Math.Sqrt(stdPrices);
                    stdSizes = Math.Sqrt(stdSizes);

                    foreach (string line in lines)
                    {
                        string[] lineElements;
                        lineElements = lines[lineCount].Split(' ');
                        //if(lineElements[0] == "0" || lineElements[1] == "0" || lineElements[0] == "64" || lineElements[0] == "65")
                        //{
                        //    StreamWriter output = File.AppendText(@"X:\detectedExampleErrors.txt");
                        //    output.WriteLine(arr[i].ToString() + "  " + lineCount);
                        //    output.Close();
                        //}

                        normPrice.Add((Convert.ToDouble(lineElements[0]) - meanOfPrices) / stdPrices);
                        normSize.Add((Convert.ToDouble(lineElements[1]) - meanOfSizes) / stdPrices);

                        priceSet.Add(Convert.ToDouble(lineElements[0]));
                        sizeSet.Add(Convert.ToDouble(lineElements[1]));
                        lineCount++;
                    }
                }
            }

            if (!File.Exists(@"X:\min_max_scaling_values.txt"))
            {
                StreamWriter output = File.AppendText(@"X:\min_max_scaling_values.txt");
                object[] priceSetArray = priceSet.ToArray();
                object[] sizeSetArray = sizeSet.ToArray();

                object[] normPriceSetArray = normPrice.ToArray();
                object[] normSizeSetArray = normSize.ToArray();
                predictorGui.globalMaxPrice = (double)priceSetArray.Max();
                predictorGui.globalMinPrice = (double)priceSetArray.Min();
                predictorGui.globalMaxSize = (double)sizeSetArray.Max();
                predictorGui.globalMinSize = (double)sizeSetArray.Min();

                predictorGui.globalScaledMaxPrice = (double)normPriceSetArray.Max();
                predictorGui.globalScaledMinPrice = (double)normPriceSetArray.Min();
                predictorGui.globalScaledMaxSize = (double)normSizeSetArray.Max();
                predictorGui.globalScaledMinSize = (double)normSizeSetArray.Min();

                output.WriteLine(predictorGui.globalScaledMaxPrice.ToString());
                output.WriteLine(predictorGui.globalScaledMinPrice.ToString());
                output.WriteLine(predictorGui.globalScaledMaxSize.ToString());
                output.WriteLine(predictorGui.globalScaledMinSize.ToString());
                output.Close();
            }
            else
            {
                string[] scaling_vals = File.ReadAllLines(@"X:\min_max_scaling_values.txt");
                predictorGui.globalScaledMaxPrice = Convert.ToDouble(scaling_vals[0]);
                predictorGui.globalScaledMinPrice = Convert.ToDouble(scaling_vals[1]);
                predictorGui.globalScaledMaxSize = Convert.ToDouble(scaling_vals[2]);
                predictorGui.globalScaledMinSize = Convert.ToDouble(scaling_vals[3]);
            }
        }
        public void reduceFileList()
        {
            while(listOfTrainingExamples.Count != 256)
            {
                for(int i = 0; i < listOfTrainingExamples.Count; i++)
                {
                    string[] inputLines = File.ReadAllLines(@"X:\trainingData\trainingTensorExample" + listOfTrainingExamples[i].ToString() + ".gt.txt");
                    int randomVal0To99 = predictorGui.rand.Next(100);
                    if(inputLines[1] == "1" && randomVal0To99 < 10)
                    {
                        listOfTrainingExamples.RemoveAt(i);
                        break;
                    }
                }
                shuffleFileList();
            }
        }

        public void increaseFileList()
        {
            while (listOfTrainingExamples.Count != 1024)
            {
                for (int i = 0; i < listOfTrainingExamples.Count; i++)
                {
                    string[] inputLines = File.ReadAllLines(@"X:\trainingData\trainingTensorExample" + listOfTrainingExamples[i].ToString() + ".gt.txt");
                    int randomVal0To99 = predictorGui.rand.Next(100);
                    if (inputLines[0] == "1" && randomVal0To99 < 10)
                    {
                        listOfTrainingExamples.Add(listOfTrainingExamples[i]);
                        break;
                    }
                }
                shuffleFileList();
            }
        }

        public void shuffleFileList()
        {
            int n = listOfTrainingExamples.Count;
            while (n > 1)
            {
                n--;
                int k = predictorGui.rand.Next(n + 1);
                var value = listOfTrainingExamples[k];
                listOfTrainingExamples[k] = listOfTrainingExamples[n];
                listOfTrainingExamples[n] = value;
            }
        }
    }
}
