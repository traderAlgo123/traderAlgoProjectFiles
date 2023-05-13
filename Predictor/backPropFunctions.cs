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

namespace Predictor
{
    public class backPropFunctions
    {
        public static void makeTransformer1InputsCopy()
        {
            Array.Copy(Transformer_Implementation.positionalEncodingArray, 0, Transformer_Implementation.positionalEncodingArrayCpy, 0, 1500);
            Array.Copy(Transformer_Implementation.inputFromConvModule, 0, Transformer_Implementation.inputFromConvModuleCpy, 0, 1500);

            Array.Copy(Transformer_Implementation.query_head1, 0, Transformer_Implementation.query_head1Cpy, 0, 500);
            Array.Copy(Transformer_Implementation.key_head1, 0, Transformer_Implementation.key_head1Cpy, 0, 500);
            Array.Copy(Transformer_Implementation.value_head1, 0, Transformer_Implementation.value_head1Cpy, 0, 500);

            Array.Copy(Transformer_Implementation.query_head2, 0, Transformer_Implementation.query_head2Cpy, 0, 500);
            Array.Copy(Transformer_Implementation.key_head2, 0, Transformer_Implementation.key_head2Cpy, 0, 500);
            Array.Copy(Transformer_Implementation.value_head2, 0, Transformer_Implementation.value_head2Cpy, 0, 500);

            Array.Copy(Transformer_Implementation.query_head3, 0, Transformer_Implementation.query_head3Cpy, 0, 500);
            Array.Copy(Transformer_Implementation.key_head3, 0, Transformer_Implementation.key_head3Cpy, 0, 500);
            Array.Copy(Transformer_Implementation.value_head3, 0, Transformer_Implementation.value_head3Cpy, 0, 500);

            Array.Copy(Transformer_Implementation.filtered_value_head1, 0, Transformer_Implementation.filtered_value_head1Cpy, 0, 500);
            Array.Copy(Transformer_Implementation.filtered_value_head2, 0, Transformer_Implementation.filtered_value_head2Cpy, 0, 500);
            Array.Copy(Transformer_Implementation.filtered_value_head3, 0, Transformer_Implementation.filtered_value_head3Cpy, 0, 500);

            Array.Copy(Transformer_Implementation.attention_filter_head1, 0, Transformer_Implementation.attention_filter_head1Cpy, 0, 10000);
            Array.Copy(Transformer_Implementation.attention_filter_head2, 0, Transformer_Implementation.attention_filter_head2Cpy, 0, 10000);
            Array.Copy(Transformer_Implementation.attention_filter_head3, 0, Transformer_Implementation.attention_filter_head3Cpy, 0, 10000);

            Array.Copy(Transformer_Implementation.finalAttentionBlockOutput, 0, Transformer_Implementation.finalAttentionBlockOutputCpy, 0, 1500);
        }
        public static void reloadTransformer1Inputs()
        {
            double[] temp;
            matrixOps matOps = new matrixOps();

            Array.Copy(Transformer_Implementation.positionalEncodingArrayCpy, 0, Transformer_Implementation.positionalEncodingArray, 0, 1500);
            Array.Copy(Transformer_Implementation.inputFromConvModuleCpy, 0, Transformer_Implementation.inputFromConvModule, 0, 1500);

            Array.Copy(Transformer_Implementation.query_head1Cpy, 0, Transformer_Implementation.query_head1, 0, 500);
            Array.Copy(Transformer_Implementation.key_head1Cpy, 0, Transformer_Implementation.key_head1, 0, 500);
            temp = matOps.transposeMat(Transformer_Implementation.key_head1, 100, 5);
            Array.Copy(temp, 0, Transformer_Implementation.key_head1, 0, 500);
            Array.Copy(Transformer_Implementation.value_head1Cpy, 0, Transformer_Implementation.value_head1, 0, 500);

            Array.Copy(Transformer_Implementation.query_head2Cpy, 0, Transformer_Implementation.query_head2, 0, 500);
            Array.Copy(Transformer_Implementation.key_head2Cpy, 0, Transformer_Implementation.key_head2, 0, 500);
            temp = matOps.transposeMat(Transformer_Implementation.key_head2, 100, 5);
            Array.Copy(temp, 0, Transformer_Implementation.key_head2, 0, 500);
            Array.Copy(Transformer_Implementation.value_head2Cpy, 0, Transformer_Implementation.value_head2, 0, 500);

            Array.Copy(Transformer_Implementation.query_head3Cpy, 0, Transformer_Implementation.query_head3, 0, 500);
            Array.Copy(Transformer_Implementation.key_head3Cpy, 0, Transformer_Implementation.key_head3, 0, 500);
            temp = matOps.transposeMat(Transformer_Implementation.key_head3, 100, 5);
            Array.Copy(temp, 0, Transformer_Implementation.key_head3, 0, 500);
            Array.Copy(Transformer_Implementation.value_head3Cpy, 0, Transformer_Implementation.value_head3, 0, 500);

            Array.Copy(Transformer_Implementation.filtered_value_head1Cpy, 0, Transformer_Implementation.filtered_value_head1, 0, 500);
            Array.Copy(Transformer_Implementation.filtered_value_head2Cpy, 0, Transformer_Implementation.filtered_value_head2, 0, 500);
            Array.Copy(Transformer_Implementation.filtered_value_head3Cpy, 0, Transformer_Implementation.filtered_value_head3, 0, 500);

            Array.Copy(Transformer_Implementation.attention_filter_head1Cpy, 0, Transformer_Implementation.attention_filter_head1, 0, 10000);
            Array.Copy(Transformer_Implementation.attention_filter_head2Cpy, 0, Transformer_Implementation.attention_filter_head2, 0, 10000);
            Array.Copy(Transformer_Implementation.attention_filter_head3Cpy, 0, Transformer_Implementation.attention_filter_head3, 0, 10000);

            Array.Copy(Transformer_Implementation.finalAttentionBlockOutputCpy, 0, Transformer_Implementation.finalAttentionBlockOutput, 0, 1500);
        }
        public double[] ranger_optimizer(double[] gradient, string layer, int blockNum, int vecListIdx)
        {
            double[] temp = gradient;
            return temp;
        }

        public void generic_rectified_adam_optimizer(double[] gradient, string layer, int blockNum, int vecListIdx)
        {
            int len = gradient.Length;
            double[] temp = new double[len];
            double[] adaptive_learning_rate = new double[len];
            double p_infinity = (2.0 / (1.0 - backProp.adam_beta_2)) - 1.0;
            double[] rectify_term = new double[len];
            double p_t;
            for (int i = 0; i < len; i++)
            {
                backProp.mlpThirdLayer_v_vec[i] = backProp.adam_beta_2 * backProp.prev_mlpThirdLayer_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                backProp.mlpThirdLayer_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_mlpThirdLayer_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                backProp.mlpThirdLayer_m_hat_vec[i] = backProp.mlpThirdLayer_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
            }

            p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
            if (p_t > 4)
            {
                for (int i = 0; i < len; i++)
                {
                    adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.mlpThirdLayer_v_vec[i] + backProp.epsilon));
                    rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                    temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.mlpThirdLayer_m_hat_vec[i] * adaptive_learning_rate[i];
                }
            }
            else
            {
                for (int i = 0; i < len; i++)
                {
                    temp[i] = backProp.initial_learning_rate * backProp.mlpThirdLayer_m_hat_vec[i];
                }
            }

            Array.Copy(backProp.mlpThirdLayer_m_vec, 0, backProp.prev_mlpThirdLayer_m_vec, 0, 9);
            Array.Copy(backProp.mlpThirdLayer_v_vec, 0, backProp.prev_mlpThirdLayer_v_vec, 0, 9);
            Array.Copy(temp, 0, backProp.mlpThirdLayer_adapted_rate, 0, 9);
        }

        public void rectified_adam_optimizer(double[] gradient, string layer, int blockNum)
        {
            if (layer.Equals("mlpSecondLayer"))
            {
                double[] temp = new double[192];
                double[] adaptive_learning_rate = new double[192];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[192];
                double p_t;
                for (int i = 0; i < 192; i++)
                {
                    backProp.mlpSecondLayer_v_vec[i] = backProp.adam_beta_2 * backProp.prev_mlpSecondLayer_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.mlpSecondLayer_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_mlpSecondLayer_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.mlpSecondLayer_m_hat_vec[i] = backProp.mlpSecondLayer_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 192; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.mlpSecondLayer_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.mlpSecondLayer_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 192; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.mlpSecondLayer_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.mlpSecondLayer_m_vec, 0, backProp.prev_mlpSecondLayer_m_vec, 0, 192);
                Array.Copy(backProp.mlpSecondLayer_v_vec, 0, backProp.prev_mlpSecondLayer_v_vec, 0, 192);
                Array.Copy(temp, 0, backProp.mlpSecondLayer_adapted_rate, 0, 192);
            }
            else if (layer.Equals("mlpFirstLayer"))
            {
                double[] temp = new double[96000];
                double[] adaptive_learning_rate = new double[96000];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[96000];
                double p_t;
                for (int i = 0; i < 96000; i++)
                {
                    backProp.mlpFirstLayer_v_vec[i] = backProp.adam_beta_2 * backProp.prev_mlpFirstLayer_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.mlpFirstLayer_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_mlpFirstLayer_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.mlpFirstLayer_m_hat_vec[i] = backProp.mlpFirstLayer_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 96000; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.mlpFirstLayer_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.mlpFirstLayer_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 96000; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.mlpFirstLayer_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.mlpFirstLayer_m_vec, 0, backProp.prev_mlpFirstLayer_m_vec, 0, 96000);
                Array.Copy(backProp.mlpFirstLayer_v_vec, 0, backProp.prev_mlpFirstLayer_v_vec, 0, 96000);
                Array.Copy(temp, 0, backProp.mlpFirstLayer_adapted_rate, 0, 96000);
            }
            else if (layer.Equals("mlpSecondLayerBiasPrelu"))
            {
                double[] temp = new double[3];
                double[] adaptive_learning_rate = new double[3];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[3];
                double p_t;
                for (int i = 0; i < 3; i++)
                {
                    backProp.mlpSecondLayerBias_v_vec[i] = backProp.adam_beta_2 * backProp.prev_mlpSecondLayerBias_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.mlpSecondLayerBias_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_mlpSecondLayerBias_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.mlpSecondLayerBias_m_hat_vec[i] = backProp.mlpSecondLayerBias_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.mlpSecondLayerBias_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.mlpSecondLayerBias_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 3; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.mlpSecondLayerBias_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.mlpSecondLayerBias_m_vec, 0, backProp.prev_mlpSecondLayerBias_m_vec, 0, 3);
                Array.Copy(backProp.mlpSecondLayerBias_v_vec, 0, backProp.prev_mlpSecondLayerBias_v_vec, 0, 3);
                Array.Copy(temp, 0, backProp.mlpSecondLayerBias_adapted_rate, 0, 3);
            }
            else if (layer.Equals("mlpFirstLayerBiasPrelu"))
            {
                double[] temp = new double[64];
                double[] adaptive_learning_rate = new double[64];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[64];
                double p_t;
                for (int i = 0; i < 64; i++)
                {
                    backProp.mlpFirstLayerBias_v_vec[i] = backProp.adam_beta_2 * backProp.prev_mlpFirstLayerBias_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.mlpFirstLayerBias_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_mlpFirstLayerBias_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.mlpFirstLayerBias_m_hat_vec[i] = backProp.mlpFirstLayerBias_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 64; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.mlpFirstLayerBias_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.mlpFirstLayerBias_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 64; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.mlpFirstLayerBias_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.mlpFirstLayerBias_m_vec, 0, backProp.prev_mlpFirstLayerBias_m_vec, 0, 64);
                Array.Copy(backProp.mlpFirstLayerBias_v_vec, 0, backProp.prev_mlpFirstLayerBias_v_vec, 0, 64);
                Array.Copy(temp, 0, backProp.mlpFirstLayerBias_adapted_rate, 0, 64);
            }
            else if (layer.Equals("affineMLPWeights4"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[900];
                    double[] adaptive_learning_rate = new double[900];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[900];
                    double p_t;
                    for (int i = 0; i < 900; i++)
                    {
                        backProp.affineMLPWeights4Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPWeights4Block2_v_vec[i]+ ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPWeights4Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPWeights4Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPWeights4Block2_m_hat_vec[i] = backProp.affineMLPWeights4Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPWeights4Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPWeights4Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPWeights4Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPWeights4Block2_m_vec, 0, backProp.prev_affineMLPWeights4Block2_m_vec, 0, 900);
                    Array.Copy(backProp.affineMLPWeights4Block2_v_vec, 0, backProp.prev_affineMLPWeights4Block2_v_vec, 0, 900);
                    Array.Copy(temp, 0, backProp.affineMLPWeights4Block2_adapted_rate, 0, 900);
                }
                else
                {
                    double[] temp = new double[900];
                    double[] adaptive_learning_rate = new double[900];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[900];
                    double p_t;
                    for (int i = 0; i < 900; i++)
                    {
                        backProp.affineMLPWeights4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPWeights4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPWeights4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPWeights4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);
                        backProp.affineMLPWeights4_m_hat_vec[i] = backProp.affineMLPWeights4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPWeights4_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPWeights4_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPWeights4_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPWeights4_m_vec, 0, backProp.prev_affineMLPWeights4_m_vec, 0, 900);
                    Array.Copy(backProp.affineMLPWeights4_v_vec, 0, backProp.prev_affineMLPWeights4_v_vec, 0, 900);
                    Array.Copy(temp, 0, backProp.affineMLPWeights4_adapted_rate, 0, 900);
                }
            }
            else if (layer.Equals("affineMLPWeights3"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[900];
                    double[] adaptive_learning_rate = new double[900];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[900];
                    double p_t;
                    for (int i = 0; i < 900; i++)
                    {
                        backProp.affineMLPWeights3Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPWeights3Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPWeights3Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPWeights3Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPWeights3Block2_m_hat_vec[i] = backProp.affineMLPWeights3Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPWeights3Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPWeights3Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPWeights3Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPWeights3Block2_m_vec, 0, backProp.prev_affineMLPWeights3Block2_m_vec, 0, 900);
                    Array.Copy(backProp.affineMLPWeights3Block2_v_vec, 0, backProp.prev_affineMLPWeights3Block2_v_vec, 0, 900);
                    Array.Copy(temp, 0, backProp.affineMLPWeights3Block2_adapted_rate, 0, 900);
                }
                else
                {
                    double[] temp = new double[900];
                    double[] adaptive_learning_rate = new double[900];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[900];
                    double p_t;
                    for (int i = 0; i < 900; i++)
                    {
                        backProp.affineMLPWeights3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPWeights3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPWeights3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPWeights3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPWeights3_m_hat_vec[i] = backProp.affineMLPWeights3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPWeights3_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPWeights3_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPWeights3_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPWeights3_m_vec, 0, backProp.prev_affineMLPWeights3_m_vec, 0, 900);
                    Array.Copy(backProp.affineMLPWeights3_v_vec, 0, backProp.prev_affineMLPWeights3_v_vec, 0, 900);
                    Array.Copy(temp, 0, backProp.affineMLPWeights3_adapted_rate, 0, 900);
                }
            }
            else if (layer.Equals("affineMLPWeights2"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[900];
                    double[] adaptive_learning_rate = new double[900];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[900];
                    double p_t;
                    for (int i = 0; i < 900; i++)
                    {
                        backProp.affineMLPWeights2Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPWeights2Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPWeights2Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPWeights2Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPWeights2Block2_m_hat_vec[i] = backProp.affineMLPWeights2Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPWeights2Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPWeights2Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPWeights2Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPWeights2Block2_m_vec, 0, backProp.prev_affineMLPWeights2Block2_m_vec, 0, 900);
                    Array.Copy(backProp.affineMLPWeights2Block2_v_vec, 0, backProp.prev_affineMLPWeights2Block2_v_vec, 0, 900);
                    Array.Copy(temp, 0, backProp.affineMLPWeights2Block2_adapted_rate, 0, 900);
                }
                else
                {
                    double[] temp = new double[900];
                    double[] adaptive_learning_rate = new double[900];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[900];
                    double p_t;
                    for (int i = 0; i < 900; i++)
                    {
                        backProp.affineMLPWeights2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPWeights2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPWeights2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPWeights2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPWeights2_m_hat_vec[i] = backProp.affineMLPWeights2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPWeights2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPWeights2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPWeights2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPWeights2_m_vec, 0, backProp.prev_affineMLPWeights2_m_vec, 0, 900);
                    Array.Copy(backProp.affineMLPWeights2_v_vec, 0, backProp.prev_affineMLPWeights2_v_vec, 0, 900);
                    Array.Copy(temp, 0, backProp.affineMLPWeights2_adapted_rate, 0, 900);
                }
            }
            else if (layer.Equals("affineMLPWeights1"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[900];
                    double[] adaptive_learning_rate = new double[900];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[900];
                    double p_t;
                    for (int i = 0; i < 900; i++)
                    {
                        backProp.affineMLPWeights1Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPWeights1Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPWeights1Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPWeights1Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPWeights1Block2_m_hat_vec[i] = backProp.affineMLPWeights1Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPWeights1Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPWeights1Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPWeights1Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPWeights1Block2_m_vec, 0, backProp.prev_affineMLPWeights1Block2_m_vec, 0, 900);
                    Array.Copy(backProp.affineMLPWeights1Block2_v_vec, 0, backProp.prev_affineMLPWeights1Block2_v_vec, 0, 900);
                    Array.Copy(temp, 0, backProp.affineMLPWeights1Block2_adapted_rate, 0, 900);
                }
                else
                {
                    double[] temp = new double[900];
                    double[] adaptive_learning_rate = new double[900];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[900];
                    double p_t;
                    for (int i = 0; i < 900; i++)
                    {
                        backProp.affineMLPWeights1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPWeights1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPWeights1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPWeights1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPWeights1_m_hat_vec[i] = backProp.affineMLPWeights1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPWeights1_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPWeights1_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 900; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPWeights1_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPWeights1_m_vec, 0, backProp.prev_affineMLPWeights1_m_vec, 0, 900);
                    Array.Copy(backProp.affineMLPWeights1_v_vec, 0, backProp.prev_affineMLPWeights1_v_vec, 0, 900);
                    Array.Copy(temp, 0, backProp.affineMLPWeights1_adapted_rate, 0, 900);
                }
            }
            else if (layer.Equals("affineMLPBiasPrelu"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[6000];
                    double[] adaptive_learning_rate = new double[6000];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[6000];
                    double p_t;
                    for (int i = 0; i < 6000; i++)
                    {
                        backProp.affineMLPBias2Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPBias2Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPBias2Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPBias2Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPBias2Block2_m_hat_vec[i] = backProp.affineMLPBias2Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 6000; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPBias2Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPBias2Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 6000; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPBias2Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPBias2Block2_m_vec, 0, backProp.prev_affineMLPBias2Block2_m_vec, 0, 6000);
                    Array.Copy(backProp.affineMLPBias2Block2_v_vec, 0, backProp.prev_affineMLPBias2Block2_v_vec, 0, 6000);
                    Array.Copy(temp, 0, backProp.affineMLPBias2Block2_adapted_rate, 0, 6000);
                }
            }
            else if (layer.Equals("affineMLPBeta2"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[1500];
                    double[] adaptive_learning_rate = new double[1500];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[1500];
                    double p_t;
                    for (int i = 0; i < 1500; i++)
                    {
                        backProp.affineMLPBeta2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPBeta2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPBeta2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPBeta2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPBeta2_m_hat_vec[i] = backProp.affineMLPBeta2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 1500; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPBeta2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPBeta2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 1500; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPBeta2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPBeta2_m_vec, 0, backProp.prev_affineMLPBeta2_m_vec, 0, 1500);
                    Array.Copy(backProp.affineMLPBeta2_v_vec, 0, backProp.prev_affineMLPBeta2_v_vec, 0, 1500);
                    Array.Copy(temp, 0, backProp.affineMLPBeta2_adapted_rate, 0, 1500);
                }
            }
            else if (layer.Equals("affineMLPGamma2"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[1500];
                    double[] adaptive_learning_rate = new double[1500];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[1500];
                    double p_t;
                    for (int i = 0; i < 1500; i++)
                    {
                        backProp.affineMLPGamma2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPGamma2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPGamma2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPGamma2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPGamma2_m_hat_vec[i] = backProp.affineMLPGamma2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 1500; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPGamma2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPGamma2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 1500; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPGamma2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPGamma2_m_vec, 0, backProp.prev_affineMLPGamma2_m_vec, 0, 1500);
                    Array.Copy(backProp.affineMLPGamma2_v_vec, 0, backProp.prev_affineMLPGamma2_v_vec, 0, 1500);
                    Array.Copy(temp, 0, backProp.affineMLPGamma2_adapted_rate, 0, 1500);
                }
            }
            else if (layer.Equals("affineMLPBias2"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[1500];
                    double[] adaptive_learning_rate = new double[1500];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[1500];
                    double p_t;
                    for (int i = 0; i < 1500; i++)
                    {
                        backProp.affineMLPBias2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPBias2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPBias2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPBias2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPBias2_m_hat_vec[i] = backProp.affineMLPBias2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 1500; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPBias2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPBias2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 1500; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPBias2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPBias2_m_vec, 0, backProp.prev_affineMLPBias2_m_vec, 0, 1500);
                    Array.Copy(backProp.affineMLPBias2_v_vec, 0, backProp.prev_affineMLPBias2_v_vec, 0, 1500);
                    Array.Copy(temp, 0, backProp.affineMLPBias2_adapted_rate, 0, 1500);
                }
            }
            else if (layer.Equals("affineMLPBeta1"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[1500];
                    double[] adaptive_learning_rate = new double[1500];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[1500];
                    double p_t;
                    for (int i = 0; i < 1500; i++)
                    {
                        backProp.affineMLPBeta1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPBeta1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPBeta1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPBeta1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPBeta1_m_hat_vec[i] = backProp.affineMLPBeta1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 1500; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPBeta1_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPBeta1_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 1500; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPBeta1_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPBeta1_m_vec, 0, backProp.prev_affineMLPBeta1_m_vec, 0, 1500);
                    Array.Copy(backProp.affineMLPBeta1_v_vec, 0, backProp.prev_affineMLPBeta1_v_vec, 0, 1500);
                    Array.Copy(temp, 0, backProp.affineMLPBeta1_adapted_rate, 0, 1500);
                }
            }
            else if (layer.Equals("affineMLPGamma1"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[1500];
                    double[] adaptive_learning_rate = new double[1500];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[1500];
                    double p_t;
                    for (int i = 0; i < 1500; i++)
                    {
                        backProp.affineMLPGamma1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_affineMLPGamma1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.affineMLPGamma1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_affineMLPGamma1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.affineMLPGamma1_m_hat_vec[i] = backProp.affineMLPGamma1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 1500; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.affineMLPGamma1_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.affineMLPGamma1_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 1500; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.affineMLPGamma1_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.affineMLPGamma1_m_vec, 0, backProp.prev_affineMLPGamma1_m_vec, 0, 1500);
                    Array.Copy(backProp.affineMLPGamma1_v_vec, 0, backProp.prev_affineMLPGamma1_v_vec, 0, 1500);
                    Array.Copy(temp, 0, backProp.affineMLPGamma1_adapted_rate, 0, 1500);
                }
            }
            else if (layer.Equals("finalLinearLayerWeights"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[225];
                    double[] adaptive_learning_rate = new double[225];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[225];
                    double p_t;
                    for (int i = 0; i < 225; i++)
                    {
                        backProp.finalLinearLayerBlock2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_finalLinearLayerBlock2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.finalLinearLayerBlock2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_finalLinearLayerBlock2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.finalLinearLayerBlock2_m_hat_vec[i] = backProp.finalLinearLayerBlock2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 225; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.finalLinearLayerBlock2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.finalLinearLayerBlock2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 225; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.finalLinearLayerBlock2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.finalLinearLayerBlock2_m_vec, 0, backProp.prev_finalLinearLayerBlock2_m_vec, 0, 225);
                    Array.Copy(backProp.finalLinearLayerBlock2_v_vec, 0, backProp.prev_finalLinearLayerBlock2_v_vec, 0, 225);
                    Array.Copy(temp, 0, backProp.finalLinearLayerBlock2_adapted_rate, 0, 225);
                }
                else
                {
                    double[] temp = new double[225];
                    double[] adaptive_learning_rate = new double[225];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[225];
                    double p_t;
                    for (int i = 0; i < 225; i++)
                    {
                        backProp.finalLinearLayer_v_vec[i] = backProp.adam_beta_2 * backProp.prev_finalLinearLayer_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.finalLinearLayer_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_finalLinearLayer_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.finalLinearLayer_m_hat_vec[i] = backProp.finalLinearLayer_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 225; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.finalLinearLayer_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.finalLinearLayer_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 225; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.finalLinearLayer_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.finalLinearLayer_m_vec, 0, backProp.prev_finalLinearLayer_m_vec, 0, 225);
                    Array.Copy(backProp.finalLinearLayer_v_vec, 0, backProp.prev_finalLinearLayer_v_vec, 0, 225);
                    Array.Copy(temp, 0, backProp.finalLinearLayer_adapted_rate, 0, 225);
                }
            }
            else if (layer.Equals("queryHead1Weights"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.queryHead1Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_queryHead1Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.queryHead1Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_queryHead1Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.queryHead1Block2_m_hat_vec[i] = backProp.queryHead1Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.queryHead1Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.queryHead1Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.queryHead1Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.queryHead1Block2_m_vec, 0, backProp.prev_queryHead1Block2_m_vec, 0, 75);
                    Array.Copy(backProp.queryHead1Block2_v_vec, 0, backProp.prev_queryHead1Block2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.queryHead1Block2_adapted_rate, 0, 75);
                }
                else
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.queryHead1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_queryHead1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.queryHead1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_queryHead1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.queryHead1_m_hat_vec[i] = backProp.queryHead1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.queryHead1_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.queryHead1_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.queryHead1_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.queryHead1_m_vec, 0, backProp.prev_queryHead1_m_vec, 0, 75);
                    Array.Copy(backProp.queryHead1_v_vec, 0, backProp.prev_queryHead1_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.queryHead1_adapted_rate, 0, 75);
                }
            }
            else if (layer.Equals("queryHead2Weights"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.queryHead2Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_queryHead2Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.queryHead2Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_queryHead2Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.queryHead2Block2_m_hat_vec[i] = backProp.queryHead2Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.queryHead2Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.queryHead2Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.queryHead2Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.queryHead2Block2_m_vec, 0, backProp.prev_queryHead2Block2_m_vec, 0, 75);
                    Array.Copy(backProp.queryHead2Block2_v_vec, 0, backProp.prev_queryHead2Block2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.queryHead2Block2_adapted_rate, 0, 75);
                }
                else
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.queryHead2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_queryHead2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.queryHead2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_queryHead2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.queryHead2_m_hat_vec[i] = backProp.queryHead2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.queryHead2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.queryHead2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.queryHead2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.queryHead2_m_vec, 0, backProp.prev_queryHead2_m_vec, 0, 75);
                    Array.Copy(backProp.queryHead2_v_vec, 0, backProp.prev_queryHead2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.queryHead2_adapted_rate, 0, 75);
                }
            }
            else if (layer.Equals("queryHead3Weights"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.queryHead3Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_queryHead3Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.queryHead3Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_queryHead3Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.queryHead3Block2_m_hat_vec[i] = backProp.queryHead3Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.queryHead3Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.queryHead3Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.queryHead3Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.queryHead3Block2_m_vec, 0, backProp.prev_queryHead3Block2_m_vec, 0, 75);
                    Array.Copy(backProp.queryHead3Block2_v_vec, 0, backProp.prev_queryHead3Block2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.queryHead3Block2_adapted_rate, 0, 75);
                }
                else
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.queryHead3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_queryHead3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.queryHead3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_queryHead3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.queryHead3_m_hat_vec[i] = backProp.queryHead3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.queryHead3_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.queryHead3_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.queryHead3_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.queryHead3_m_vec, 0, backProp.prev_queryHead3_m_vec, 0, 75);
                    Array.Copy(backProp.queryHead3_v_vec, 0, backProp.prev_queryHead3_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.queryHead3_adapted_rate, 0, 75);
                }
            }
            else if (layer.Equals("keyHead1Weights"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.keyHead1Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_keyHead1Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.keyHead1Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_keyHead1Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.keyHead1Block2_m_hat_vec[i] = backProp.keyHead1Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.keyHead1Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.keyHead1Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.keyHead1Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.keyHead1Block2_m_vec, 0, backProp.prev_keyHead1Block2_m_vec, 0, 75);
                    Array.Copy(backProp.keyHead1Block2_v_vec, 0, backProp.prev_keyHead1Block2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.keyHead1Block2_adapted_rate, 0, 75);
                }
                else
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.keyHead1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_keyHead1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.keyHead1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_keyHead1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.keyHead1_m_hat_vec[i] = backProp.keyHead1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.keyHead1_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.keyHead1_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.keyHead1_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.keyHead1_m_vec, 0, backProp.prev_keyHead1_m_vec, 0, 75);
                    Array.Copy(backProp.keyHead1_v_vec, 0, backProp.prev_keyHead1_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.keyHead1_adapted_rate, 0, 75);
                }
            }
            else if (layer.Equals("keyHead2Weights"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.keyHead2Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_keyHead2Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.keyHead2Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_keyHead2Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.keyHead2Block2_m_hat_vec[i] = backProp.keyHead2Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.keyHead2Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.keyHead2Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.keyHead2Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.keyHead2Block2_m_vec, 0, backProp.prev_keyHead2Block2_m_vec, 0, 75);
                    Array.Copy(backProp.keyHead2Block2_v_vec, 0, backProp.prev_keyHead2Block2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.keyHead2Block2_adapted_rate, 0, 75);
                }
                else
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.keyHead2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_keyHead2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.keyHead2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_keyHead2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.keyHead2_m_hat_vec[i] = backProp.keyHead2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.keyHead2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.keyHead2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.keyHead2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.keyHead2_m_vec, 0, backProp.prev_keyHead2_m_vec, 0, 75);
                    Array.Copy(backProp.keyHead2_v_vec, 0, backProp.prev_keyHead2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.keyHead2_adapted_rate, 0, 75);
                }
            }
            else if (layer.Equals("keyHead3Weights"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.keyHead3Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_keyHead3Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.keyHead3Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_keyHead3Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.keyHead3Block2_m_hat_vec[i] = backProp.keyHead3Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.keyHead3Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.keyHead3Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.keyHead3Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.keyHead3Block2_m_vec, 0, backProp.prev_keyHead3Block2_m_vec, 0, 75);
                    Array.Copy(backProp.keyHead3Block2_v_vec, 0, backProp.prev_keyHead3Block2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.keyHead3Block2_adapted_rate, 0, 75);
                }
                else
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.keyHead3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_keyHead3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.keyHead3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_keyHead3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.keyHead3_m_hat_vec[i] = backProp.keyHead3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.keyHead3_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.keyHead3_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.keyHead3_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.keyHead3_m_vec, 0, backProp.prev_keyHead3_m_vec, 0, 75);
                    Array.Copy(backProp.keyHead3_v_vec, 0, backProp.prev_keyHead3_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.keyHead3_adapted_rate, 0, 75);
                }
            }
            else if (layer.Equals("valueHead1Weights"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.valueHead1Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_valueHead1Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.valueHead1Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_valueHead1Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.valueHead1Block2_m_hat_vec[i] = backProp.valueHead1Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.valueHead1Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.valueHead1Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.valueHead1Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.valueHead1Block2_m_vec, 0, backProp.prev_valueHead1Block2_m_vec, 0, 75);
                    Array.Copy(backProp.valueHead1Block2_v_vec, 0, backProp.prev_valueHead1Block2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.valueHead1Block2_adapted_rate, 0, 75);
                }
                else
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.valueHead1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_valueHead1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.valueHead1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_valueHead1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.valueHead1_m_hat_vec[i] = backProp.keyHead1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.valueHead1_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.valueHead1_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.valueHead1_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.valueHead1_m_vec, 0, backProp.prev_valueHead1_m_vec, 0, 75);
                    Array.Copy(backProp.valueHead1_v_vec, 0, backProp.prev_valueHead1_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.valueHead1_adapted_rate, 0, 75);
                }
            }
            else if (layer.Equals("valueHead2Weights"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.valueHead2Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_valueHead2Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.valueHead2Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_valueHead2Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.valueHead2Block2_m_hat_vec[i] = backProp.valueHead2Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.valueHead2Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.valueHead2Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.valueHead2Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.valueHead2Block2_m_vec, 0, backProp.prev_valueHead2Block2_m_vec, 0, 75);
                    Array.Copy(backProp.valueHead2Block2_v_vec, 0, backProp.prev_valueHead2Block2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.valueHead2Block2_adapted_rate, 0, 75);
                }
                else
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.valueHead2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_valueHead2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.valueHead2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_valueHead2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.valueHead2_m_hat_vec[i] = backProp.valueHead2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.valueHead2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.valueHead2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.valueHead2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.valueHead2_m_vec, 0, backProp.prev_valueHead2_m_vec, 0, 75);
                    Array.Copy(backProp.valueHead2_v_vec, 0, backProp.prev_valueHead2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.valueHead2_adapted_rate, 0, 75);
                }
            }
            else if (layer.Equals("valueHead3Weights"))
            {
                if (blockNum == 2)
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.valueHead3Block2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_valueHead3Block2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.valueHead3Block2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_valueHead3Block2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.valueHead3Block2_m_hat_vec[i] = backProp.valueHead3Block2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.valueHead3Block2_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.valueHead3Block2_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.valueHead3Block2_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.valueHead3Block2_m_vec, 0, backProp.prev_valueHead3Block2_m_vec, 0, 75);
                    Array.Copy(backProp.valueHead3Block2_v_vec, 0, backProp.prev_valueHead3Block2_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.valueHead3Block2_adapted_rate, 0, 75);
                }
                else
                {
                    double[] temp = new double[75];
                    double[] adaptive_learning_rate = new double[75];
                    double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                    double[] rectify_term = new double[75];
                    double p_t;
                    for (int i = 0; i < 75; i++)
                    {
                        backProp.valueHead3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_valueHead3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                        backProp.valueHead3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_valueHead3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                        backProp.valueHead3_m_hat_vec[i] = backProp.valueHead3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                    }

                    p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                    if (p_t > 4)
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.valueHead3_v_vec[i] + backProp.epsilon));
                            rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                            temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.valueHead3_m_hat_vec[i] * adaptive_learning_rate[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < 75; i++)
                        {
                            temp[i] = backProp.initial_learning_rate * backProp.valueHead3_m_hat_vec[i];
                        }
                    }
                    Array.Copy(backProp.valueHead3_m_vec, 0, backProp.prev_valueHead3_m_vec, 0, 75);
                    Array.Copy(backProp.valueHead3_v_vec, 0, backProp.prev_valueHead3_v_vec, 0, 75);
                    Array.Copy(temp, 0, backProp.valueHead3_adapted_rate, 0, 75);
                }
            }
            else if (layer.Equals("convLayer5Bias"))
            {
                double[] temp = new double[1400];
                double[] adaptive_learning_rate = new double[1400];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[1400];
                double p_t;
                for (int i = 0; i < 1400; i++)
                {
                    backProp.convLayer5Bias_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Bias_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Bias_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Bias_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Bias_m_hat_vec[i] = backProp.convLayer5Bias_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Bias_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Bias_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Bias_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Bias_m_vec, 0, backProp.prev_convLayer5Bias_m_vec, 0, 1400);
                Array.Copy(backProp.convLayer5Bias_v_vec, 0, backProp.prev_convLayer5Bias_v_vec, 0, 1400);
                Array.Copy(temp, 0, backProp.convLayer5Bias_adapted_rate, 0, 1400);
            }
            else if (layer.Equals("convLayer5Gamma"))
            {
                double[] temp = new double[1400];
                double[] adaptive_learning_rate = new double[1400];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[1400];
                double p_t;
                for (int i = 0; i < 1400; i++)
                {
                    backProp.convLayer5Gamma_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Gamma_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Gamma_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Gamma_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Gamma_m_hat_vec[i] = backProp.convLayer5Gamma_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Gamma_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Gamma_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Gamma_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Gamma_m_vec, 0, backProp.prev_convLayer5Gamma_m_vec, 0, 1400);
                Array.Copy(backProp.convLayer5Gamma_v_vec, 0, backProp.prev_convLayer5Gamma_v_vec, 0, 1400);
                Array.Copy(temp, 0, backProp.convLayer5Gamma_adapted_rate, 0, 1400);
            }
            else if (layer.Equals("convLayer5Beta"))
            {
                double[] temp = new double[1400];
                double[] adaptive_learning_rate = new double[1400];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[1400];
                double p_t;
                for (int i = 0; i < 1400; i++)
                {
                    backProp.convLayer5Beta_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Beta_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Beta_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Beta_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Beta_m_hat_vec[i] = backProp.convLayer5Beta_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Beta_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Beta_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Beta_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Beta_m_vec, 0, backProp.prev_convLayer5Beta_m_vec, 0, 1400);
                Array.Copy(backProp.convLayer5Beta_v_vec, 0, backProp.prev_convLayer5Beta_v_vec, 0, 1400);
                Array.Copy(temp, 0, backProp.convLayer5Beta_adapted_rate, 0, 1400);
            }
            else if (layer.Equals("convLayer5WeightsKernel1Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel1Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel1Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel1Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel1Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel1Depth1_m_hat_vec[i] = backProp.convLayer5Kernel1Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel1Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel1Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel1Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel1Depth1_m_vec, 0, backProp.prev_convLayer5Kernel1Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel1Depth1_v_vec, 0, backProp.prev_convLayer5Kernel1Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel1Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel1Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel1Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel1Depth2_v_vec[i]+ ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel1Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel1Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel1Depth2_m_hat_vec[i] = backProp.convLayer5Kernel1Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel1Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel1Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel1Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel1Depth2_m_vec, 0, backProp.prev_convLayer5Kernel1Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel1Depth2_v_vec, 0, backProp.prev_convLayer5Kernel1Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel1Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel2Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel2Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel2Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel2Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel2Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel2Depth1_m_hat_vec[i] = backProp.convLayer5Kernel2Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel2Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel2Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel2Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel2Depth1_m_vec, 0, backProp.prev_convLayer5Kernel2Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel2Depth1_v_vec, 0, backProp.prev_convLayer5Kernel2Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel2Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel2Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel2Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel2Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel2Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel2Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel2Depth2_m_hat_vec[i] = backProp.convLayer5Kernel2Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel2Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel2Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel2Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel2Depth2_m_vec, 0, backProp.prev_convLayer5Kernel2Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel2Depth2_v_vec, 0, backProp.prev_convLayer5Kernel2Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel2Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel3Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel3Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel3Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel3Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel3Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel3Depth1_m_hat_vec[i] = backProp.convLayer5Kernel3Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel3Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel3Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel3Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel3Depth1_m_vec, 0, backProp.prev_convLayer5Kernel3Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel3Depth1_v_vec, 0, backProp.prev_convLayer5Kernel3Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel3Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel3Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel3Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel3Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel3Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel3Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel3Depth2_m_hat_vec[i] = backProp.convLayer5Kernel3Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel3Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel3Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel3Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel3Depth2_m_vec, 0, backProp.prev_convLayer5Kernel3Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel3Depth2_v_vec, 0, backProp.prev_convLayer5Kernel3Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel3Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel4Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel4Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel4Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel4Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel4Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel4Depth1_m_hat_vec[i] = backProp.convLayer5Kernel4Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel4Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel4Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel4Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel4Depth1_m_vec, 0, backProp.prev_convLayer5Kernel4Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel4Depth1_v_vec, 0, backProp.prev_convLayer5Kernel4Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel4Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel4Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel4Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel4Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel4Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel4Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel4Depth2_m_hat_vec[i] = backProp.convLayer5Kernel4Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel4Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel4Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel4Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel4Depth2_m_vec, 0, backProp.prev_convLayer5Kernel4Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel4Depth2_v_vec, 0, backProp.prev_convLayer5Kernel4Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel4Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel5Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel5Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel5Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel5Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel5Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel5Depth1_m_hat_vec[i] = backProp.convLayer5Kernel5Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel5Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel5Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel5Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel5Depth1_m_vec, 0, backProp.prev_convLayer5Kernel5Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel5Depth1_v_vec, 0, backProp.prev_convLayer5Kernel5Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel5Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel5Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel5Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel5Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel5Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel5Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel5Depth2_m_hat_vec[i] = backProp.convLayer5Kernel5Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel5Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel5Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel5Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel5Depth2_m_vec, 0, backProp.prev_convLayer5Kernel5Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel5Depth2_v_vec, 0, backProp.prev_convLayer5Kernel5Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel5Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel6Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel6Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel6Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel6Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel6Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel6Depth1_m_hat_vec[i] = backProp.convLayer5Kernel6Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel6Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel6Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel6Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel6Depth1_m_vec, 0, backProp.prev_convLayer5Kernel6Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel6Depth1_v_vec, 0, backProp.prev_convLayer5Kernel6Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel6Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel6Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel6Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel6Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel6Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel6Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel6Depth2_m_hat_vec[i] = backProp.convLayer5Kernel6Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel6Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel6Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel6Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel6Depth2_m_vec, 0, backProp.prev_convLayer5Kernel6Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel6Depth2_v_vec, 0, backProp.prev_convLayer5Kernel6Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel6Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel7Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel7Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel7Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel7Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel7Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel7Depth1_m_hat_vec[i] = backProp.convLayer5Kernel7Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel7Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel7Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel7Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel7Depth1_m_vec, 0, backProp.prev_convLayer5Kernel7Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel7Depth1_v_vec, 0, backProp.prev_convLayer5Kernel7Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel7Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel7Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel7Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel7Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel7Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel7Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel7Depth2_m_hat_vec[i] = backProp.convLayer5Kernel7Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel7Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel7Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel7Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel7Depth2_m_vec, 0, backProp.prev_convLayer5Kernel7Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel7Depth2_v_vec, 0, backProp.prev_convLayer5Kernel7Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel7Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel8Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel8Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel8Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel8Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel8Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel8Depth1_m_hat_vec[i] = backProp.convLayer5Kernel8Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel8Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel8Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel8Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel8Depth1_m_vec, 0, backProp.prev_convLayer5Kernel8Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel8Depth1_v_vec, 0, backProp.prev_convLayer5Kernel8Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel8Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel8Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel8Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel8Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel8Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel8Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel8Depth2_m_hat_vec[i] = backProp.convLayer5Kernel8Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel8Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel8Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel8Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel8Depth2_m_vec, 0, backProp.prev_convLayer5Kernel8Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel8Depth2_v_vec, 0, backProp.prev_convLayer5Kernel8Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel8Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel9Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel9Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel9Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel9Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel9Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel9Depth1_m_hat_vec[i] = backProp.convLayer5Kernel9Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel9Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel9Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel9Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel9Depth1_m_vec, 0, backProp.prev_convLayer5Kernel9Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel9Depth1_v_vec, 0, backProp.prev_convLayer5Kernel9Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel9Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel9Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel9Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel9Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel9Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel9Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel9Depth2_m_hat_vec[i] = backProp.convLayer5Kernel9Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel9Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel9Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel9Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel9Depth2_m_vec, 0, backProp.prev_convLayer5Kernel9Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel9Depth2_v_vec, 0, backProp.prev_convLayer5Kernel9Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel9Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel10Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel10Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel10Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel10Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel10Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel10Depth1_m_hat_vec[i] = backProp.convLayer5Kernel10Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel10Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel10Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel10Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel10Depth1_m_vec, 0, backProp.prev_convLayer5Kernel10Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel10Depth1_v_vec, 0, backProp.prev_convLayer5Kernel10Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel10Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel10Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel10Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel10Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel10Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel10Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel10Depth2_m_hat_vec[i] = backProp.convLayer5Kernel10Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel10Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel10Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel10Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel10Depth2_m_vec, 0, backProp.prev_convLayer5Kernel10Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel10Depth2_v_vec, 0, backProp.prev_convLayer5Kernel10Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel10Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel11Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel11Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel11Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel11Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel11Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel11Depth1_m_hat_vec[i] = backProp.convLayer5Kernel11Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel11Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel11Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel11Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel11Depth1_m_vec, 0, backProp.prev_convLayer5Kernel11Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel11Depth1_v_vec, 0, backProp.prev_convLayer5Kernel11Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel11Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel11Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel11Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel11Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel11Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel11Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel11Depth2_m_hat_vec[i] = backProp.convLayer5Kernel11Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel11Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel11Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel11Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel11Depth2_m_vec, 0, backProp.prev_convLayer5Kernel11Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel11Depth2_v_vec, 0, backProp.prev_convLayer5Kernel11Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel11Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel12Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel12Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel12Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel12Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel12Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel12Depth1_m_hat_vec[i] = backProp.convLayer5Kernel12Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel12Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel12Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel12Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel12Depth1_m_vec, 0, backProp.prev_convLayer5Kernel12Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel12Depth1_v_vec, 0, backProp.prev_convLayer5Kernel12Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel12Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel12Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel12Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel12Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel12Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel12Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel12Depth2_m_hat_vec[i] = backProp.convLayer5Kernel12Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel12Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel12Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel12Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel12Depth2_m_vec, 0, backProp.prev_convLayer5Kernel12Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel12Depth2_v_vec, 0, backProp.prev_convLayer5Kernel12Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel12Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel13Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel13Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel13Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel13Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel13Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel13Depth1_m_hat_vec[i] = backProp.convLayer5Kernel13Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel13Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel13Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel13Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel13Depth1_m_vec, 0, backProp.prev_convLayer5Kernel13Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel13Depth1_v_vec, 0, backProp.prev_convLayer5Kernel13Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel13Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel13Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel13Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel13Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel13Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel13Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel13Depth2_m_hat_vec[i] = backProp.convLayer5Kernel13Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel13Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel13Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel13Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel13Depth2_m_vec, 0, backProp.prev_convLayer5Kernel13Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel13Depth2_v_vec, 0, backProp.prev_convLayer5Kernel13Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel13Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel14Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel14Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel14Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel14Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel14Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel14Depth1_m_hat_vec[i] = backProp.convLayer5Kernel14Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel14Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel14Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel14Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel14Depth1_m_vec, 0, backProp.prev_convLayer5Kernel14Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel14Depth1_v_vec, 0, backProp.prev_convLayer5Kernel14Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel14Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer5WeightsKernel14Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1.0 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer5Kernel14Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer5Kernel14Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer5Kernel14Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer5Kernel14Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel14Depth2_m_hat_vec[i] = backProp.convLayer5Kernel14Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4.0)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer5Kernel14Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4.0) * (p_t - 2.0) * p_infinity) / ((p_infinity - 4.0) * (p_infinity - 2.0) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer5Kernel14Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer5Kernel14Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer5Kernel14Depth2_m_vec, 0, backProp.prev_convLayer5Kernel14Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer5Kernel14Depth2_v_vec, 0, backProp.prev_convLayer5Kernel14Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer5Kernel14Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4BiasPrelu"))
            {
                double[] temp = new double[1400];
                double[] adaptive_learning_rate = new double[1400];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[1400];
                double p_t;
                for (int i = 0; i < 1400; i++)
                {
                    backProp.convLayer4Bias_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Bias_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Bias_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Bias_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Bias_m_hat_vec[i] = backProp.convLayer4Bias_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Bias_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Bias_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Bias_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Bias_m_vec, 0, backProp.prev_convLayer4Bias_m_vec, 0, 1400);
                Array.Copy(backProp.convLayer4Bias_v_vec, 0, backProp.prev_convLayer4Bias_v_vec, 0, 1400);
                Array.Copy(temp, 0, backProp.convLayer4Bias_adapted_rate, 0, 1400);
            }
            else if (layer.Equals("convLayer4WeightsKernel1Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel1Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel1Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel1Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel1Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel1Depth1_m_hat_vec[i] = backProp.convLayer4Kernel1Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel1Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel1Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel1Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel1Depth1_m_vec, 0, backProp.prev_convLayer4Kernel1Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel1Depth1_v_vec, 0, backProp.prev_convLayer4Kernel1Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel1Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel1Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel1Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel1Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel1Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel1Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel1Depth2_m_hat_vec[i] = backProp.convLayer4Kernel1Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel1Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel1Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel1Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel1Depth2_m_vec, 0, backProp.prev_convLayer4Kernel1Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel1Depth2_v_vec, 0, backProp.prev_convLayer4Kernel1Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel1Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel2Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel2Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel2Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel2Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel2Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel2Depth1_m_hat_vec[i] = backProp.convLayer4Kernel2Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel2Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel2Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel2Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel2Depth1_m_vec, 0, backProp.prev_convLayer4Kernel2Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel2Depth1_v_vec, 0, backProp.prev_convLayer4Kernel2Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel2Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel2Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel2Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel2Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel2Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel2Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel2Depth2_m_hat_vec[i] = backProp.convLayer4Kernel2Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel2Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel2Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel2Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel2Depth2_m_vec, 0, backProp.prev_convLayer4Kernel2Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel2Depth2_v_vec, 0, backProp.prev_convLayer4Kernel2Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel2Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel3Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel3Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel3Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel3Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel3Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel3Depth1_m_hat_vec[i] = backProp.convLayer4Kernel3Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel3Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel3Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel3Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel3Depth1_m_vec, 0, backProp.prev_convLayer4Kernel3Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel3Depth1_v_vec, 0, backProp.prev_convLayer4Kernel3Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel3Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel3Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel3Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel3Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel3Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel3Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel3Depth2_m_hat_vec[i] = backProp.convLayer4Kernel3Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel3Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel3Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel3Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel3Depth2_m_vec, 0, backProp.prev_convLayer4Kernel3Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel3Depth2_v_vec, 0, backProp.prev_convLayer4Kernel3Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel3Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel4Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel4Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel4Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel4Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel4Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel4Depth1_m_hat_vec[i] = backProp.convLayer4Kernel4Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel4Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel4Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel4Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel4Depth1_m_vec, 0, backProp.prev_convLayer4Kernel4Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel4Depth1_v_vec, 0, backProp.prev_convLayer4Kernel4Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel4Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel4Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel4Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel4Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel4Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel4Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel4Depth2_m_hat_vec[i] = backProp.convLayer4Kernel4Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel4Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel4Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel4Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel4Depth2_m_vec, 0, backProp.prev_convLayer4Kernel4Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel4Depth2_v_vec, 0, backProp.prev_convLayer4Kernel4Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel4Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel5Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel5Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel5Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel5Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel5Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel5Depth1_m_hat_vec[i] = backProp.convLayer4Kernel5Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel5Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel5Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel5Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel5Depth1_m_vec, 0, backProp.prev_convLayer4Kernel5Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel5Depth1_v_vec, 0, backProp.prev_convLayer4Kernel5Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel5Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel5Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel5Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel5Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel5Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel5Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel5Depth2_m_hat_vec[i] = backProp.convLayer4Kernel5Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel5Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel5Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel5Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel5Depth2_m_vec, 0, backProp.prev_convLayer4Kernel5Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel5Depth2_v_vec, 0, backProp.prev_convLayer4Kernel5Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel5Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel6Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel6Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel6Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel6Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel6Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel6Depth1_m_hat_vec[i] = backProp.convLayer4Kernel6Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel6Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel6Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel6Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel6Depth1_m_vec, 0, backProp.prev_convLayer4Kernel6Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel6Depth1_v_vec, 0, backProp.prev_convLayer4Kernel6Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel6Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel6Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel6Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel6Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel6Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel6Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel6Depth2_m_hat_vec[i] = backProp.convLayer4Kernel6Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel6Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel6Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel6Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel6Depth2_m_vec, 0, backProp.prev_convLayer4Kernel6Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel6Depth2_v_vec, 0, backProp.prev_convLayer4Kernel6Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel6Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel7Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel7Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel7Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel7Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel7Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel7Depth1_m_hat_vec[i] = backProp.convLayer4Kernel7Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel7Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel7Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel7Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel7Depth1_m_vec, 0, backProp.prev_convLayer4Kernel7Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel7Depth1_v_vec, 0, backProp.prev_convLayer4Kernel7Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel7Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel7Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel7Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel7Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel7Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel7Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel7Depth2_m_hat_vec[i] = backProp.convLayer4Kernel7Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel7Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel7Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel7Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel7Depth2_m_vec, 0, backProp.prev_convLayer4Kernel7Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel7Depth2_v_vec, 0, backProp.prev_convLayer4Kernel7Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel7Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel8Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel8Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel8Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel8Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel8Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel8Depth1_m_hat_vec[i] = backProp.convLayer4Kernel8Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel8Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel8Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel8Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel8Depth1_m_vec, 0, backProp.prev_convLayer4Kernel8Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel8Depth1_v_vec, 0, backProp.prev_convLayer4Kernel8Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel8Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel8Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel8Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel8Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel8Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel8Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel8Depth2_m_hat_vec[i] = backProp.convLayer4Kernel8Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel8Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel8Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel8Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel8Depth2_m_vec, 0, backProp.prev_convLayer4Kernel8Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel8Depth2_v_vec, 0, backProp.prev_convLayer4Kernel8Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel8Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel9Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel9Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel9Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel9Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel9Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel9Depth1_m_hat_vec[i] = backProp.convLayer4Kernel9Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel9Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel9Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel9Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel9Depth1_m_vec, 0, backProp.prev_convLayer4Kernel9Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel9Depth1_v_vec, 0, backProp.prev_convLayer4Kernel9Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel9Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel9Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel9Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel9Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel9Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel9Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel9Depth2_m_hat_vec[i] = backProp.convLayer4Kernel9Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel9Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel9Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel9Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel9Depth2_m_vec, 0, backProp.prev_convLayer4Kernel9Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel9Depth2_v_vec, 0, backProp.prev_convLayer4Kernel9Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel9Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel10Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel10Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel10Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel10Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel10Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel10Depth1_m_hat_vec[i] = backProp.convLayer4Kernel10Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel10Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel10Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel10Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel10Depth1_m_vec, 0, backProp.prev_convLayer4Kernel10Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel10Depth1_v_vec, 0, backProp.prev_convLayer4Kernel10Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel10Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel10Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel10Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel10Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel10Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel10Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel10Depth2_m_hat_vec[i] = backProp.convLayer4Kernel10Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel10Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel10Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel10Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel10Depth2_m_vec, 0, backProp.prev_convLayer4Kernel10Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel10Depth2_v_vec, 0, backProp.prev_convLayer4Kernel10Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel10Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel11Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel11Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel11Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel11Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel11Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel11Depth1_m_hat_vec[i] = backProp.convLayer4Kernel11Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel11Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel11Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel11Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel11Depth1_m_vec, 0, backProp.prev_convLayer4Kernel11Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel11Depth1_v_vec, 0, backProp.prev_convLayer4Kernel11Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel11Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel11Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel11Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel11Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel11Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel11Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel11Depth2_m_hat_vec[i] = backProp.convLayer4Kernel11Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel11Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel11Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel11Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel11Depth2_m_vec, 0, backProp.prev_convLayer4Kernel11Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel11Depth2_v_vec, 0, backProp.prev_convLayer4Kernel11Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel11Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel12Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel12Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel12Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel12Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel12Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel12Depth1_m_hat_vec[i] = backProp.convLayer4Kernel12Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel12Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel12Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel12Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel12Depth1_m_vec, 0, backProp.prev_convLayer4Kernel12Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel12Depth1_v_vec, 0, backProp.prev_convLayer4Kernel12Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel12Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel12Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel12Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel12Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel12Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel12Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel12Depth2_m_hat_vec[i] = backProp.convLayer4Kernel12Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel12Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel12Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel12Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel12Depth2_m_vec, 0, backProp.prev_convLayer4Kernel12Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel12Depth2_v_vec, 0, backProp.prev_convLayer4Kernel12Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel12Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel13Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel13Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel13Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel13Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel13Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel13Depth1_m_hat_vec[i] = backProp.convLayer4Kernel13Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel13Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel13Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel13Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel13Depth1_m_vec, 0, backProp.prev_convLayer4Kernel13Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel13Depth1_v_vec, 0, backProp.prev_convLayer4Kernel13Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel13Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel13Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel13Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel13Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel13Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel13Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel13Depth2_m_hat_vec[i] = backProp.convLayer4Kernel13Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel13Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel13Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel13Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel13Depth2_m_vec, 0, backProp.prev_convLayer4Kernel13Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel13Depth2_v_vec, 0, backProp.prev_convLayer4Kernel13Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel13Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel14Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel14Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel14Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel14Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel14Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel14Depth1_m_hat_vec[i] = backProp.convLayer4Kernel14Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel14Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel14Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel14Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel14Depth1_m_vec, 0, backProp.prev_convLayer4Kernel14Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel14Depth1_v_vec, 0, backProp.prev_convLayer4Kernel14Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel14Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer4WeightsKernel14Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1.0 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer4Kernel14Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer4Kernel14Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer4Kernel14Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer4Kernel14Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer4Kernel14Depth2_m_hat_vec[i] = backProp.convLayer4Kernel14Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4.0)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer4Kernel14Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4.0) * (p_t - 2.0) * p_infinity) / ((p_infinity - 4.0) * (p_infinity - 2.0) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer4Kernel14Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer4Kernel14Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer4Kernel14Depth2_m_vec, 0, backProp.prev_convLayer4Kernel14Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer4Kernel14Depth2_v_vec, 0, backProp.prev_convLayer4Kernel14Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer4Kernel14Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3BiasPrelu"))
            {
                double[] temp = new double[1400];
                double[] adaptive_learning_rate = new double[1400];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[1400];
                double p_t;
                for (int i = 0; i < 1400; i++)
                {
                    backProp.convLayer3Bias_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Bias_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Bias_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Bias_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Bias_m_hat_vec[i] = backProp.convLayer3Bias_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Bias_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Bias_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Bias_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Bias_m_vec, 0, backProp.prev_convLayer3Bias_m_vec, 0, 1400);
                Array.Copy(backProp.convLayer3Bias_v_vec, 0, backProp.prev_convLayer3Bias_v_vec, 0, 1400);
                Array.Copy(temp, 0, backProp.convLayer3Bias_adapted_rate, 0, 1400);
            }
            else if (layer.Equals("convLayer3WeightsKernel1Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel1Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel1Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel1Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel1Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel1Depth1_m_hat_vec[i] = backProp.convLayer3Kernel1Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel1Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel1Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel1Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel1Depth1_m_vec, 0, backProp.prev_convLayer3Kernel1Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel1Depth1_v_vec, 0, backProp.prev_convLayer3Kernel1Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel1Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel1Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel1Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel1Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel1Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel1Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel1Depth2_m_hat_vec[i] = backProp.convLayer3Kernel1Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel1Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel1Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel1Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel1Depth2_m_vec, 0, backProp.prev_convLayer3Kernel1Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel1Depth2_v_vec, 0, backProp.prev_convLayer3Kernel1Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel1Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel2Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel2Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel2Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel2Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel2Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel2Depth1_m_hat_vec[i] = backProp.convLayer3Kernel2Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel2Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel2Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel2Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel2Depth1_m_vec, 0, backProp.prev_convLayer3Kernel2Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel2Depth1_v_vec, 0, backProp.prev_convLayer3Kernel2Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel2Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel2Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel2Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel2Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel2Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel2Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel2Depth2_m_hat_vec[i] = backProp.convLayer3Kernel2Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel2Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel2Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel2Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel2Depth2_m_vec, 0, backProp.prev_convLayer3Kernel2Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel2Depth2_v_vec, 0, backProp.prev_convLayer3Kernel2Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel2Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel3Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel3Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel3Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel3Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel3Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel3Depth1_m_hat_vec[i] = backProp.convLayer3Kernel3Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel3Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel3Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel3Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel3Depth1_m_vec, 0, backProp.prev_convLayer3Kernel3Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel3Depth1_v_vec, 0, backProp.prev_convLayer3Kernel3Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel3Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel3Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel3Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel3Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel3Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel3Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel3Depth2_m_hat_vec[i] = backProp.convLayer3Kernel3Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel3Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel3Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel3Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel3Depth2_m_vec, 0, backProp.prev_convLayer3Kernel3Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel3Depth2_v_vec, 0, backProp.prev_convLayer3Kernel3Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel3Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel4Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel4Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel4Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel4Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel4Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel4Depth1_m_hat_vec[i] = backProp.convLayer3Kernel4Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel4Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel4Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel4Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel4Depth1_m_vec, 0, backProp.prev_convLayer3Kernel4Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel4Depth1_v_vec, 0, backProp.prev_convLayer3Kernel4Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel4Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel4Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel4Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel4Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel4Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel4Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel4Depth2_m_hat_vec[i] = backProp.convLayer3Kernel4Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel4Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel4Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel4Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel4Depth2_m_vec, 0, backProp.prev_convLayer3Kernel4Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel4Depth2_v_vec, 0, backProp.prev_convLayer3Kernel4Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel4Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel5Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel5Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel5Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel5Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel5Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel5Depth1_m_hat_vec[i] = backProp.convLayer3Kernel5Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel5Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel5Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel5Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel5Depth1_m_vec, 0, backProp.prev_convLayer3Kernel5Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel5Depth1_v_vec, 0, backProp.prev_convLayer3Kernel5Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel5Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel5Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel5Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel5Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel5Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel5Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel5Depth2_m_hat_vec[i] = backProp.convLayer3Kernel5Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel5Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel5Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel5Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel5Depth2_m_vec, 0, backProp.prev_convLayer3Kernel5Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel5Depth2_v_vec, 0, backProp.prev_convLayer3Kernel5Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel5Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel6Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel6Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel6Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel6Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel6Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel6Depth1_m_hat_vec[i] = backProp.convLayer3Kernel6Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel6Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel6Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel6Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel6Depth1_m_vec, 0, backProp.prev_convLayer3Kernel6Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel6Depth1_v_vec, 0, backProp.prev_convLayer3Kernel6Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel6Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel6Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel6Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel6Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel6Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel6Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel6Depth2_m_hat_vec[i] = backProp.convLayer3Kernel6Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel6Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel6Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel6Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel6Depth2_m_vec, 0, backProp.prev_convLayer3Kernel6Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel6Depth2_v_vec, 0, backProp.prev_convLayer3Kernel6Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel6Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel7Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel7Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel7Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel7Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel7Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel7Depth1_m_hat_vec[i] = backProp.convLayer3Kernel7Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel7Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel7Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel7Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel7Depth1_m_vec, 0, backProp.prev_convLayer3Kernel7Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel7Depth1_v_vec, 0, backProp.prev_convLayer3Kernel7Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel7Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel7Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel7Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel7Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel7Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel7Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel7Depth2_m_hat_vec[i] = backProp.convLayer3Kernel7Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel7Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel7Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel7Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel7Depth2_m_vec, 0, backProp.prev_convLayer3Kernel7Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel7Depth2_v_vec, 0, backProp.prev_convLayer3Kernel7Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel7Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel8Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel8Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel8Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel8Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel8Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel8Depth1_m_hat_vec[i] = backProp.convLayer3Kernel8Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel8Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel8Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel8Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel8Depth1_m_vec, 0, backProp.prev_convLayer3Kernel8Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel8Depth1_v_vec, 0, backProp.prev_convLayer3Kernel8Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel8Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel8Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel8Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel8Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel8Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel8Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel8Depth2_m_hat_vec[i] = backProp.convLayer3Kernel8Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel8Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel8Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel8Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel8Depth2_m_vec, 0, backProp.prev_convLayer3Kernel8Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel8Depth2_v_vec, 0, backProp.prev_convLayer3Kernel8Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel8Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel9Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel9Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel9Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel9Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel9Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel9Depth1_m_hat_vec[i] = backProp.convLayer3Kernel9Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel9Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel9Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel9Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel9Depth1_m_vec, 0, backProp.prev_convLayer3Kernel9Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel9Depth1_v_vec, 0, backProp.prev_convLayer3Kernel9Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel9Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel9Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel9Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel9Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel9Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel9Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel9Depth2_m_hat_vec[i] = backProp.convLayer3Kernel9Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel9Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel9Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel9Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel9Depth2_m_vec, 0, backProp.prev_convLayer3Kernel9Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel9Depth2_v_vec, 0, backProp.prev_convLayer3Kernel9Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel9Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel10Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel10Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel10Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel10Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel10Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel10Depth1_m_hat_vec[i] = backProp.convLayer3Kernel10Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel10Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel10Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel10Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel10Depth1_m_vec, 0, backProp.prev_convLayer3Kernel10Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel10Depth1_v_vec, 0, backProp.prev_convLayer3Kernel10Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel10Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel10Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel10Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel10Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel10Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel10Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel10Depth2_m_hat_vec[i] = backProp.convLayer3Kernel10Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel10Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel10Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel10Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel10Depth2_m_vec, 0, backProp.prev_convLayer3Kernel10Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel10Depth2_v_vec, 0, backProp.prev_convLayer3Kernel10Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel10Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel11Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel11Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel11Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel11Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel11Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel11Depth1_m_hat_vec[i] = backProp.convLayer3Kernel11Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel11Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel11Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel11Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel11Depth1_m_vec, 0, backProp.prev_convLayer3Kernel11Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel11Depth1_v_vec, 0, backProp.prev_convLayer3Kernel11Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel11Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel11Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel11Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel11Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel11Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel11Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel11Depth2_m_hat_vec[i] = backProp.convLayer3Kernel11Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel11Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel11Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel11Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel11Depth2_m_vec, 0, backProp.prev_convLayer3Kernel11Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel11Depth2_v_vec, 0, backProp.prev_convLayer3Kernel11Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel11Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel12Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel12Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel12Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel12Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel12Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel12Depth1_m_hat_vec[i] = backProp.convLayer3Kernel12Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel12Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel12Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel12Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel12Depth1_m_vec, 0, backProp.prev_convLayer3Kernel12Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel12Depth1_v_vec, 0, backProp.prev_convLayer3Kernel12Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel12Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel12Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel12Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel12Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel12Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel12Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel12Depth2_m_hat_vec[i] = backProp.convLayer3Kernel12Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel12Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel12Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel12Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel12Depth2_m_vec, 0, backProp.prev_convLayer3Kernel12Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel12Depth2_v_vec, 0, backProp.prev_convLayer3Kernel12Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel12Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel13Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel13Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel13Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel13Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel13Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel13Depth1_m_hat_vec[i] = backProp.convLayer3Kernel13Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel13Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel13Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel13Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel13Depth1_m_vec, 0, backProp.prev_convLayer3Kernel13Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel13Depth1_v_vec, 0, backProp.prev_convLayer3Kernel13Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel13Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel13Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel13Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel13Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel13Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel13Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel13Depth2_m_hat_vec[i] = backProp.convLayer3Kernel13Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel13Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel13Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel13Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel13Depth2_m_vec, 0, backProp.prev_convLayer3Kernel13Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel13Depth2_v_vec, 0, backProp.prev_convLayer3Kernel13Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel13Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel14Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel14Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel14Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel14Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel14Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel14Depth1_m_hat_vec[i] = backProp.convLayer3Kernel14Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel14Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel14Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel14Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel14Depth1_m_vec, 0, backProp.prev_convLayer3Kernel14Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel14Depth1_v_vec, 0, backProp.prev_convLayer3Kernel14Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel14Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer3WeightsKernel14Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1.0 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer3Kernel14Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer3Kernel14Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer3Kernel14Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer3Kernel14Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer3Kernel14Depth2_m_hat_vec[i] = backProp.convLayer3Kernel14Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4.0)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer3Kernel14Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4.0) * (p_t - 2.0) * p_infinity) / ((p_infinity - 4.0) * (p_infinity - 2.0) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer3Kernel14Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer3Kernel14Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer3Kernel14Depth2_m_vec, 0, backProp.prev_convLayer3Kernel14Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer3Kernel14Depth2_v_vec, 0, backProp.prev_convLayer3Kernel14Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer3Kernel14Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2BiasPrelu"))
            {
                double[] temp = new double[1400];
                double[] adaptive_learning_rate = new double[1400];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[1400];
                double p_t;
                for (int i = 0; i < 1400; i++)
                {
                    backProp.convLayer2Bias_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Bias_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Bias_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Bias_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Bias_m_hat_vec[i] = backProp.convLayer2Bias_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Bias_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Bias_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Bias_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Bias_m_vec, 0, backProp.prev_convLayer2Bias_m_vec, 0, 1400);
                Array.Copy(backProp.convLayer2Bias_v_vec, 0, backProp.prev_convLayer2Bias_v_vec, 0, 1400);
                Array.Copy(temp, 0, backProp.convLayer2Bias_adapted_rate, 0, 1400);
            }
            else if (layer.Equals("convLayer2WeightsKernel1Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel1Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel1Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel1Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel1Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel1Depth1_m_hat_vec[i] = backProp.convLayer2Kernel1Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel1Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel1Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel1Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel1Depth1_m_vec, 0, backProp.prev_convLayer2Kernel1Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel1Depth1_v_vec, 0, backProp.prev_convLayer2Kernel1Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel1Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel1Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel1Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel1Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel1Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel1Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel1Depth2_m_hat_vec[i] = backProp.convLayer2Kernel1Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel1Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel1Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel1Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel1Depth2_m_vec, 0, backProp.prev_convLayer2Kernel1Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel1Depth2_v_vec, 0, backProp.prev_convLayer2Kernel1Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel1Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel2Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel2Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel2Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel2Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel2Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel2Depth1_m_hat_vec[i] = backProp.convLayer2Kernel2Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel2Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel2Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel2Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel2Depth1_m_vec, 0, backProp.prev_convLayer2Kernel2Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel2Depth1_v_vec, 0, backProp.prev_convLayer2Kernel2Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel2Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel2Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel2Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel2Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel2Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel2Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel2Depth2_m_hat_vec[i] = backProp.convLayer2Kernel2Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel2Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel2Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel2Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel2Depth2_m_vec, 0, backProp.prev_convLayer2Kernel2Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel2Depth2_v_vec, 0, backProp.prev_convLayer2Kernel2Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel2Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel3Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel3Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel3Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel3Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel3Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel3Depth1_m_hat_vec[i] = backProp.convLayer2Kernel3Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel3Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel3Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel3Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel3Depth1_m_vec, 0, backProp.prev_convLayer2Kernel3Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel3Depth1_v_vec, 0, backProp.prev_convLayer2Kernel3Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel3Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel3Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel3Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel3Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel3Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel3Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel3Depth2_m_hat_vec[i] = backProp.convLayer2Kernel3Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel3Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel3Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel3Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel3Depth2_m_vec, 0, backProp.prev_convLayer2Kernel3Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel3Depth2_v_vec, 0, backProp.prev_convLayer2Kernel3Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel3Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel4Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel4Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel4Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel4Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel4Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel4Depth1_m_hat_vec[i] = backProp.convLayer2Kernel4Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel4Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel4Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel4Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel4Depth1_m_vec, 0, backProp.prev_convLayer2Kernel4Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel4Depth1_v_vec, 0, backProp.prev_convLayer2Kernel4Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel4Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel4Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel4Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel4Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel4Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel4Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel4Depth2_m_hat_vec[i] = backProp.convLayer2Kernel4Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel4Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel4Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel4Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel4Depth2_m_vec, 0, backProp.prev_convLayer2Kernel4Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel4Depth2_v_vec, 0, backProp.prev_convLayer2Kernel4Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel4Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel5Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel5Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel5Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel5Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel5Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel5Depth1_m_hat_vec[i] = backProp.convLayer2Kernel5Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel5Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel5Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel5Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel5Depth1_m_vec, 0, backProp.prev_convLayer2Kernel5Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel5Depth1_v_vec, 0, backProp.prev_convLayer2Kernel5Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel5Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel5Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel5Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel5Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel5Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel5Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel5Depth2_m_hat_vec[i] = backProp.convLayer2Kernel5Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel5Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel5Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel5Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel5Depth2_m_vec, 0, backProp.prev_convLayer2Kernel5Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel5Depth2_v_vec, 0, backProp.prev_convLayer2Kernel5Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel5Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel6Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel6Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel6Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel6Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel6Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel6Depth1_m_hat_vec[i] = backProp.convLayer2Kernel6Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel6Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel6Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel6Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel6Depth1_m_vec, 0, backProp.prev_convLayer2Kernel6Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel6Depth1_v_vec, 0, backProp.prev_convLayer2Kernel6Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel6Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel6Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel6Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel6Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel6Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel6Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel6Depth2_m_hat_vec[i] = backProp.convLayer2Kernel6Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel6Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel6Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel6Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel6Depth2_m_vec, 0, backProp.prev_convLayer2Kernel6Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel6Depth2_v_vec, 0, backProp.prev_convLayer2Kernel6Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel6Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel7Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel7Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel7Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel7Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel7Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer5Kernel7Depth1_m_hat_vec[i] = backProp.convLayer2Kernel7Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel7Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel7Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel7Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel7Depth1_m_vec, 0, backProp.prev_convLayer2Kernel7Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel7Depth1_v_vec, 0, backProp.prev_convLayer2Kernel7Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel7Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel7Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel7Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel7Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel7Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel7Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel7Depth2_m_hat_vec[i] = backProp.convLayer2Kernel7Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel7Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel7Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel7Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel7Depth2_m_vec, 0, backProp.prev_convLayer2Kernel7Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel7Depth2_v_vec, 0, backProp.prev_convLayer2Kernel7Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel7Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel8Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel8Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel8Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel8Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel8Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel8Depth1_m_hat_vec[i] = backProp.convLayer2Kernel8Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel8Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel8Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel8Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel8Depth1_m_vec, 0, backProp.prev_convLayer2Kernel8Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel8Depth1_v_vec, 0, backProp.prev_convLayer2Kernel8Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel8Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel8Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel8Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel8Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel8Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel8Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel8Depth2_m_hat_vec[i] = backProp.convLayer2Kernel8Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel8Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel8Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel8Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel8Depth2_m_vec, 0, backProp.prev_convLayer2Kernel8Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel8Depth2_v_vec, 0, backProp.prev_convLayer2Kernel8Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel8Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel9Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel9Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel9Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel9Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel9Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel9Depth1_m_hat_vec[i] = backProp.convLayer2Kernel9Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel9Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel9Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel9Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel9Depth1_m_vec, 0, backProp.prev_convLayer2Kernel9Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel9Depth1_v_vec, 0, backProp.prev_convLayer2Kernel9Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel9Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel9Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel9Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel9Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel9Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel9Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel9Depth2_m_hat_vec[i] = backProp.convLayer2Kernel9Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel9Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel9Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel9Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel9Depth2_m_vec, 0, backProp.prev_convLayer2Kernel9Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel9Depth2_v_vec, 0, backProp.prev_convLayer2Kernel9Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel9Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel10Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel10Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel10Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel10Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel10Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel10Depth1_m_hat_vec[i] = backProp.convLayer2Kernel10Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel10Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel10Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel10Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel10Depth1_m_vec, 0, backProp.prev_convLayer2Kernel10Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel10Depth1_v_vec, 0, backProp.prev_convLayer2Kernel10Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel10Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel10Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel10Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel10Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel10Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel10Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel10Depth2_m_hat_vec[i] = backProp.convLayer2Kernel10Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel10Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel10Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel10Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel10Depth2_m_vec, 0, backProp.prev_convLayer2Kernel10Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel10Depth2_v_vec, 0, backProp.prev_convLayer2Kernel10Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel10Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel11Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel11Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel11Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel11Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel11Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel11Depth1_m_hat_vec[i] = backProp.convLayer2Kernel11Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel11Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel11Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel11Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel11Depth1_m_vec, 0, backProp.prev_convLayer2Kernel11Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel11Depth1_v_vec, 0, backProp.prev_convLayer2Kernel11Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel11Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel11Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel11Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel11Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel11Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel11Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel11Depth2_m_hat_vec[i] = backProp.convLayer2Kernel11Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel11Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel11Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel11Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel11Depth2_m_vec, 0, backProp.prev_convLayer2Kernel11Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel11Depth2_v_vec, 0, backProp.prev_convLayer2Kernel11Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel11Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel12Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel12Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel12Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel12Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel12Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel12Depth1_m_hat_vec[i] = backProp.convLayer2Kernel12Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel12Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel12Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel12Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel12Depth1_m_vec, 0, backProp.prev_convLayer2Kernel12Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel12Depth1_v_vec, 0, backProp.prev_convLayer2Kernel12Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel12Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel12Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel12Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel12Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel12Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel12Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel12Depth2_m_hat_vec[i] = backProp.convLayer2Kernel12Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel12Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel12Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel12Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel12Depth2_m_vec, 0, backProp.prev_convLayer2Kernel12Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel12Depth2_v_vec, 0, backProp.prev_convLayer2Kernel12Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel12Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel13Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel13Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel13Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel13Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel13Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel13Depth1_m_hat_vec[i] = backProp.convLayer2Kernel13Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel13Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel13Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel13Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel13Depth1_m_vec, 0, backProp.prev_convLayer2Kernel13Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel13Depth1_v_vec, 0, backProp.prev_convLayer2Kernel13Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel13Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel13Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel13Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel13Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel13Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel13Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel13Depth2_m_hat_vec[i] = backProp.convLayer2Kernel13Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel13Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel13Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel13Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel13Depth2_m_vec, 0, backProp.prev_convLayer2Kernel13Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel13Depth2_v_vec, 0, backProp.prev_convLayer2Kernel13Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel13Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel14Depth1"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel14Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel14Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel14Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel14Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel14Depth1_m_hat_vec[i] = backProp.convLayer2Kernel14Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel14Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel14Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel14Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel14Depth1_m_vec, 0, backProp.prev_convLayer2Kernel14Depth1_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel14Depth1_v_vec, 0, backProp.prev_convLayer2Kernel14Depth1_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel14Depth1_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer2WeightsKernel14Depth2"))
            {
                double[] temp = new double[14];
                double[] adaptive_learning_rate = new double[14];
                double p_infinity = (2.0 / (1.0 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[14];
                double p_t;
                for (int i = 0; i < 14; i++)
                {
                    backProp.convLayer2Kernel14Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer2Kernel14Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer2Kernel14Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer2Kernel14Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer2Kernel14Depth2_m_hat_vec[i] = backProp.convLayer2Kernel14Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4.0)
                {
                    for (int i = 0; i < 14; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer2Kernel14Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4.0) * (p_t - 2.0) * p_infinity) / ((p_infinity - 4.0) * (p_infinity - 2.0) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer2Kernel14Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 14; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer2Kernel14Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer2Kernel14Depth2_m_vec, 0, backProp.prev_convLayer2Kernel14Depth2_m_vec, 0, 14);
                Array.Copy(backProp.convLayer2Kernel14Depth2_v_vec, 0, backProp.prev_convLayer2Kernel14Depth2_v_vec, 0, 14);
                Array.Copy(temp, 0, backProp.convLayer2Kernel14Depth2_adapted_rate, 0, 14);
            }
            else if (layer.Equals("convLayer1BiasPrelu"))
            {
                double[] temp = new double[1400];
                double[] adaptive_learning_rate = new double[1400];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[1400];
                double p_t;
                for (int i = 0; i < 1400; i++)
                {
                    backProp.convLayer1Bias_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Bias_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Bias_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Bias_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Bias_m_hat_vec[i] = backProp.convLayer1Bias_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Bias_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Bias_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 1400; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Bias_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Bias_m_vec, 0, backProp.prev_convLayer1Bias_m_vec, 0, 1400);
                Array.Copy(backProp.convLayer1Bias_v_vec, 0, backProp.prev_convLayer1Bias_v_vec, 0, 1400);
                Array.Copy(temp, 0, backProp.convLayer1Bias_adapted_rate, 0, 1400);
            }
            else if (layer.Equals("convLayer1WeightsKernel1Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel1Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel1Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel1Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel1Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel1Depth1_m_hat_vec[i] = backProp.convLayer1Kernel1Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel1Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel1Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel1Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel1Depth1_m_vec, 0, backProp.prev_convLayer1Kernel1Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel1Depth1_v_vec, 0, backProp.prev_convLayer1Kernel1Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel1Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel1Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel1Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel1Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel1Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel1Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel1Depth2_m_hat_vec[i] = backProp.convLayer1Kernel1Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel1Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel1Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel1Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel1Depth2_m_vec, 0, backProp.prev_convLayer1Kernel1Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel1Depth2_v_vec, 0, backProp.prev_convLayer1Kernel1Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel1Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel1Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel1Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel1Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel1Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel1Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel1Depth3_m_hat_vec[i] = backProp.convLayer1Kernel1Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel1Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel1Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel1Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel1Depth3_m_vec, 0, backProp.prev_convLayer1Kernel1Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel1Depth3_v_vec, 0, backProp.prev_convLayer1Kernel1Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel1Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel1Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel1Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel1Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel1Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel1Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel1Depth4_m_hat_vec[i] = backProp.convLayer1Kernel1Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel1Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel1Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel1Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel1Depth4_m_vec, 0, backProp.prev_convLayer1Kernel1Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel1Depth4_v_vec, 0, backProp.prev_convLayer1Kernel1Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel1Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel2Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel2Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel2Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel2Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel2Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel2Depth1_m_hat_vec[i] = backProp.convLayer1Kernel2Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel2Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel2Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel2Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel2Depth1_m_vec, 0, backProp.prev_convLayer1Kernel2Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel2Depth1_v_vec, 0, backProp.prev_convLayer1Kernel2Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel2Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel2Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel2Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel2Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel2Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel2Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel2Depth2_m_hat_vec[i] = backProp.convLayer1Kernel2Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel2Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel2Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel2Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel2Depth2_m_vec, 0, backProp.prev_convLayer1Kernel2Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel2Depth2_v_vec, 0, backProp.prev_convLayer1Kernel2Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel2Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel2Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel2Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel2Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel2Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel2Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel2Depth3_m_hat_vec[i] = backProp.convLayer1Kernel2Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel2Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel2Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel2Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel2Depth3_m_vec, 0, backProp.prev_convLayer1Kernel2Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel2Depth3_v_vec, 0, backProp.prev_convLayer1Kernel2Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel2Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel2Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel2Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel2Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel2Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel2Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel2Depth4_m_hat_vec[i] = backProp.convLayer1Kernel2Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel2Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel2Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel2Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel2Depth4_m_vec, 0, backProp.prev_convLayer1Kernel2Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel2Depth4_v_vec, 0, backProp.prev_convLayer1Kernel2Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel2Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel3Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel3Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel3Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel3Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel3Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel3Depth1_m_hat_vec[i] = backProp.convLayer1Kernel3Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel3Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel3Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel3Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel3Depth1_m_vec, 0, backProp.prev_convLayer1Kernel3Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel3Depth1_v_vec, 0, backProp.prev_convLayer1Kernel3Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel3Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel3Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel3Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel3Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel3Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel3Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel3Depth2_m_hat_vec[i] = backProp.convLayer1Kernel3Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel3Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel3Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel3Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel3Depth2_m_vec, 0, backProp.prev_convLayer1Kernel3Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel3Depth2_v_vec, 0, backProp.prev_convLayer1Kernel3Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel3Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel3Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel3Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel3Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel3Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel3Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel3Depth3_m_hat_vec[i] = backProp.convLayer1Kernel3Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel3Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel3Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel3Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel3Depth3_m_vec, 0, backProp.prev_convLayer1Kernel3Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel3Depth3_v_vec, 0, backProp.prev_convLayer1Kernel3Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel3Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel3Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel3Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel3Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel3Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel3Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel3Depth4_m_hat_vec[i] = backProp.convLayer1Kernel3Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel3Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel3Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel3Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel3Depth4_m_vec, 0, backProp.prev_convLayer1Kernel3Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel3Depth4_v_vec, 0, backProp.prev_convLayer1Kernel3Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel3Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel4Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel4Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel4Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel4Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel4Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel4Depth1_m_hat_vec[i] = backProp.convLayer1Kernel4Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel4Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel4Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel4Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel4Depth1_m_vec, 0, backProp.prev_convLayer1Kernel4Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel4Depth1_v_vec, 0, backProp.prev_convLayer1Kernel4Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel4Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel4Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel4Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel4Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel4Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel4Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel4Depth2_m_hat_vec[i] = backProp.convLayer1Kernel4Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel4Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel4Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel4Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel4Depth2_m_vec, 0, backProp.prev_convLayer1Kernel4Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel4Depth2_v_vec, 0, backProp.prev_convLayer1Kernel4Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel4Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel4Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel4Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel4Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel4Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel4Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel4Depth3_m_hat_vec[i] = backProp.convLayer1Kernel4Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel4Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel4Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel4Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel4Depth3_m_vec, 0, backProp.prev_convLayer1Kernel4Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel4Depth3_v_vec, 0, backProp.prev_convLayer1Kernel4Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel4Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel4Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel4Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel4Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel4Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel4Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel4Depth4_m_hat_vec[i] = backProp.convLayer1Kernel4Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel4Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel4Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel4Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel4Depth4_m_vec, 0, backProp.prev_convLayer1Kernel4Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel4Depth4_v_vec, 0, backProp.prev_convLayer1Kernel4Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel4Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel5Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel5Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel5Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel5Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel5Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel5Depth1_m_hat_vec[i] = backProp.convLayer1Kernel5Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel5Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel5Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel5Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel5Depth1_m_vec, 0, backProp.prev_convLayer1Kernel5Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel5Depth1_v_vec, 0, backProp.prev_convLayer1Kernel5Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel5Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel5Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel5Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel5Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel5Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel5Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel5Depth2_m_hat_vec[i] = backProp.convLayer1Kernel5Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel5Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel5Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel5Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel5Depth2_m_vec, 0, backProp.prev_convLayer1Kernel5Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel5Depth2_v_vec, 0, backProp.prev_convLayer1Kernel5Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel5Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel5Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel5Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel5Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel5Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel5Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel5Depth3_m_hat_vec[i] = backProp.convLayer1Kernel5Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel5Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel5Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel5Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel5Depth3_m_vec, 0, backProp.prev_convLayer1Kernel5Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel5Depth3_v_vec, 0, backProp.prev_convLayer1Kernel5Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel5Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel5Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel5Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel5Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel5Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel5Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel5Depth4_m_hat_vec[i] = backProp.convLayer1Kernel5Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel5Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel5Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel5Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel5Depth4_m_vec, 0, backProp.prev_convLayer1Kernel5Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel5Depth4_v_vec, 0, backProp.prev_convLayer1Kernel5Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel5Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel6Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel6Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel6Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel6Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel6Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel6Depth1_m_hat_vec[i] = backProp.convLayer1Kernel6Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel6Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel6Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel6Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel6Depth1_m_vec, 0, backProp.prev_convLayer1Kernel6Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel6Depth1_v_vec, 0, backProp.prev_convLayer1Kernel6Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel6Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel6Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel6Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel6Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel6Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel6Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel6Depth2_m_hat_vec[i] = backProp.convLayer1Kernel6Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel6Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel6Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel6Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel6Depth2_m_vec, 0, backProp.prev_convLayer1Kernel6Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel6Depth2_v_vec, 0, backProp.prev_convLayer1Kernel6Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel6Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel6Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel6Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel6Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel6Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel6Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel6Depth3_m_hat_vec[i] = backProp.convLayer1Kernel6Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel6Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel6Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel6Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel6Depth3_m_vec, 0, backProp.prev_convLayer1Kernel6Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel6Depth3_v_vec, 0, backProp.prev_convLayer1Kernel6Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel6Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel6Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel6Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel6Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel6Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel6Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel6Depth4_m_hat_vec[i] = backProp.convLayer1Kernel6Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel6Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel6Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel6Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel6Depth4_m_vec, 0, backProp.prev_convLayer1Kernel6Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel6Depth4_v_vec, 0, backProp.prev_convLayer1Kernel6Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel6Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel7Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel7Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel7Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel7Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel7Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel7Depth1_m_hat_vec[i] = backProp.convLayer1Kernel7Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel7Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel7Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel7Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel7Depth1_m_vec, 0, backProp.prev_convLayer1Kernel7Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel7Depth1_v_vec, 0, backProp.prev_convLayer1Kernel7Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel7Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel7Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel7Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel7Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel7Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel7Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel7Depth2_m_hat_vec[i] = backProp.convLayer1Kernel7Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel7Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel7Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel7Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel7Depth2_m_vec, 0, backProp.prev_convLayer1Kernel7Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel7Depth2_v_vec, 0, backProp.prev_convLayer1Kernel7Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel7Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel7Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel7Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel7Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel7Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel7Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel7Depth3_m_hat_vec[i] = backProp.convLayer1Kernel7Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel7Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel7Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel7Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel7Depth3_m_vec, 0, backProp.prev_convLayer1Kernel7Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel7Depth3_v_vec, 0, backProp.prev_convLayer1Kernel7Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel7Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel7Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel7Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel7Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel7Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel7Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel7Depth4_m_hat_vec[i] = backProp.convLayer1Kernel7Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel7Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel7Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel7Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel7Depth4_m_vec, 0, backProp.prev_convLayer1Kernel7Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel7Depth4_v_vec, 0, backProp.prev_convLayer1Kernel7Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel7Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel8Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel8Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel8Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel8Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel8Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel8Depth1_m_hat_vec[i] = backProp.convLayer1Kernel8Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel8Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel8Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel8Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel8Depth1_m_vec, 0, backProp.prev_convLayer1Kernel8Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel8Depth1_v_vec, 0, backProp.prev_convLayer1Kernel8Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel8Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel8Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel8Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel8Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel8Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel8Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel8Depth2_m_hat_vec[i] = backProp.convLayer1Kernel8Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel8Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel8Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel8Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel8Depth2_m_vec, 0, backProp.prev_convLayer1Kernel8Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel8Depth2_v_vec, 0, backProp.prev_convLayer1Kernel8Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel8Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel8Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel8Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel8Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel8Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel8Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel8Depth3_m_hat_vec[i] = backProp.convLayer1Kernel8Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel8Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel8Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel8Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel8Depth3_m_vec, 0, backProp.prev_convLayer1Kernel8Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel8Depth3_v_vec, 0, backProp.prev_convLayer1Kernel8Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel8Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel8Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel8Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel8Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel8Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel8Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel8Depth4_m_hat_vec[i] = backProp.convLayer1Kernel8Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel8Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel8Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel8Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel8Depth4_m_vec, 0, backProp.prev_convLayer1Kernel8Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel8Depth4_v_vec, 0, backProp.prev_convLayer1Kernel8Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel8Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel9Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel9Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel9Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel9Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel9Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel9Depth1_m_hat_vec[i] = backProp.convLayer1Kernel9Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel9Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel9Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel9Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel9Depth1_m_vec, 0, backProp.prev_convLayer1Kernel9Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel9Depth1_v_vec, 0, backProp.prev_convLayer1Kernel9Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel9Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel9Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel9Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel9Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel9Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel9Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel9Depth2_m_hat_vec[i] = backProp.convLayer1Kernel9Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel9Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel9Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel9Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel9Depth2_m_vec, 0, backProp.prev_convLayer1Kernel9Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel9Depth2_v_vec, 0, backProp.prev_convLayer1Kernel9Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel9Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel9Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel9Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel9Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel9Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel9Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel9Depth3_m_hat_vec[i] = backProp.convLayer1Kernel9Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel9Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel9Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel9Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel9Depth3_m_vec, 0, backProp.prev_convLayer1Kernel9Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel9Depth3_v_vec, 0, backProp.prev_convLayer1Kernel9Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel9Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel9Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel9Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel9Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel9Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel9Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel9Depth4_m_hat_vec[i] = backProp.convLayer1Kernel9Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel9Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel9Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel9Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel9Depth4_m_vec, 0, backProp.prev_convLayer1Kernel9Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel9Depth4_v_vec, 0, backProp.prev_convLayer1Kernel9Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel9Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel10Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel10Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel10Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel10Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel10Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel10Depth1_m_hat_vec[i] = backProp.convLayer1Kernel10Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel10Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel10Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel10Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel10Depth1_m_vec, 0, backProp.prev_convLayer1Kernel10Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel10Depth1_v_vec, 0, backProp.prev_convLayer1Kernel10Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel10Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel10Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel10Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel10Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel10Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel10Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel10Depth2_m_hat_vec[i] = backProp.convLayer1Kernel10Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel10Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel10Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel10Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel10Depth2_m_vec, 0, backProp.prev_convLayer1Kernel10Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel10Depth2_v_vec, 0, backProp.prev_convLayer1Kernel10Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel10Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel10Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel10Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel10Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel10Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel10Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel10Depth3_m_hat_vec[i] = backProp.convLayer1Kernel10Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel10Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel10Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel10Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel10Depth3_m_vec, 0, backProp.prev_convLayer1Kernel10Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel10Depth3_v_vec, 0, backProp.prev_convLayer1Kernel10Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel10Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel10Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel10Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel10Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel10Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel10Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel10Depth4_m_hat_vec[i] = backProp.convLayer1Kernel10Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel10Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel10Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel10Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel10Depth4_m_vec, 0, backProp.prev_convLayer1Kernel10Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel10Depth4_v_vec, 0, backProp.prev_convLayer1Kernel10Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel10Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel11Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel11Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel11Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel11Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel11Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel11Depth1_m_hat_vec[i] = backProp.convLayer1Kernel11Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel11Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel11Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel11Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel11Depth1_m_vec, 0, backProp.prev_convLayer1Kernel11Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel11Depth1_v_vec, 0, backProp.prev_convLayer1Kernel11Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel11Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel11Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel11Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel11Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel11Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel11Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel11Depth2_m_hat_vec[i] = backProp.convLayer1Kernel11Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel11Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel11Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel11Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel11Depth2_m_vec, 0, backProp.prev_convLayer1Kernel11Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel11Depth2_v_vec, 0, backProp.prev_convLayer1Kernel11Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel11Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel11Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel11Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel11Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel11Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel11Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel11Depth3_m_hat_vec[i] = backProp.convLayer1Kernel11Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel11Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel11Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel11Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel11Depth3_m_vec, 0, backProp.prev_convLayer1Kernel11Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel11Depth3_v_vec, 0, backProp.prev_convLayer1Kernel11Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel11Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel11Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel11Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel11Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel11Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel11Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel11Depth4_m_hat_vec[i] = backProp.convLayer1Kernel11Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel11Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel11Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel11Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel11Depth4_m_vec, 0, backProp.prev_convLayer1Kernel11Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel11Depth4_v_vec, 0, backProp.prev_convLayer1Kernel11Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel11Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel12Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel12Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel12Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel12Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel12Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel12Depth1_m_hat_vec[i] = backProp.convLayer1Kernel12Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel12Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel12Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel12Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel12Depth1_m_vec, 0, backProp.prev_convLayer1Kernel12Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel12Depth1_v_vec, 0, backProp.prev_convLayer1Kernel12Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel12Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel12Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel12Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel12Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel12Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel12Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel12Depth2_m_hat_vec[i] = backProp.convLayer1Kernel12Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel12Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel12Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel12Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel12Depth2_m_vec, 0, backProp.prev_convLayer1Kernel12Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel12Depth2_v_vec, 0, backProp.prev_convLayer1Kernel12Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel12Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel12Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel12Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel12Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel12Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel12Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel12Depth3_m_hat_vec[i] = backProp.convLayer1Kernel12Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel12Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel12Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel12Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel12Depth3_m_vec, 0, backProp.prev_convLayer1Kernel12Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel12Depth3_v_vec, 0, backProp.prev_convLayer1Kernel12Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel12Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel12Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel12Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel12Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel12Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel12Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel12Depth4_m_hat_vec[i] = backProp.convLayer1Kernel12Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel12Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel12Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel12Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel12Depth4_m_vec, 0, backProp.prev_convLayer1Kernel12Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel12Depth4_v_vec, 0, backProp.prev_convLayer1Kernel12Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel12Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel13Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel13Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel13Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel13Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel13Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel13Depth1_m_hat_vec[i] = backProp.convLayer1Kernel13Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel13Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel13Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel13Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel13Depth1_m_vec, 0, backProp.prev_convLayer1Kernel13Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel13Depth1_v_vec, 0, backProp.prev_convLayer1Kernel13Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel13Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel13Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel13Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel13Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel13Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel13Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel13Depth2_m_hat_vec[i] = backProp.convLayer1Kernel13Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel13Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel13Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel13Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel13Depth2_m_vec, 0, backProp.prev_convLayer1Kernel13Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel13Depth2_v_vec, 0, backProp.prev_convLayer1Kernel13Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel13Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel13Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel13Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel13Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel13Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel13Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel13Depth3_m_hat_vec[i] = backProp.convLayer1Kernel13Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel13Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel13Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel13Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel13Depth3_m_vec, 0, backProp.prev_convLayer1Kernel13Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel13Depth3_v_vec, 0, backProp.prev_convLayer1Kernel13Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel13Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel13Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel13Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel13Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel13Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel13Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel13Depth4_m_hat_vec[i] = backProp.convLayer1Kernel13Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel13Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel13Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel13Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel13Depth4_m_vec, 0, backProp.prev_convLayer1Kernel13Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel13Depth4_v_vec, 0, backProp.prev_convLayer1Kernel13Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel13Depth4_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel14Depth1"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel14Depth1_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel14Depth1_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel14Depth1_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel14Depth1_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel14Depth1_m_hat_vec[i] = backProp.convLayer1Kernel14Depth1_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel14Depth1_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel14Depth1_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel14Depth1_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel14Depth1_m_vec, 0, backProp.prev_convLayer1Kernel14Depth1_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel14Depth1_v_vec, 0, backProp.prev_convLayer1Kernel14Depth1_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel14Depth1_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel14Depth2"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel14Depth2_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel14Depth2_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel14Depth2_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel14Depth2_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel14Depth2_m_hat_vec[i] = backProp.convLayer1Kernel14Depth2_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel14Depth2_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel14Depth2_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel14Depth2_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel14Depth2_m_vec, 0, backProp.prev_convLayer1Kernel14Depth2_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel14Depth2_v_vec, 0, backProp.prev_convLayer1Kernel14Depth2_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel14Depth2_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel14Depth3"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel14Depth3_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel14Depth3_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel14Depth3_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel14Depth3_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel14Depth3_m_hat_vec[i] = backProp.convLayer1Kernel14Depth3_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel14Depth3_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel14Depth3_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel14Depth3_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel14Depth3_m_vec, 0, backProp.prev_convLayer1Kernel14Depth3_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel14Depth3_v_vec, 0, backProp.prev_convLayer1Kernel14Depth3_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel14Depth3_adapted_rate, 0, 32);
            }
            else if (layer.Equals("convLayer1WeightsKernel14Depth4"))
            {
                double[] temp = new double[32];
                double[] adaptive_learning_rate = new double[32];
                double p_infinity = (2.0 / (1 - backProp.adam_beta_2)) - 1.0;
                double[] rectify_term = new double[32];
                double p_t;
                for (int i = 0; i < 32; i++)
                {
                    backProp.convLayer1Kernel14Depth4_v_vec[i] = backProp.adam_beta_2 * backProp.prev_convLayer1Kernel14Depth4_v_vec[i] + ((1.0 - backProp.adam_beta_2) * Math.Pow(gradient[i], 2));
                    backProp.convLayer1Kernel14Depth4_m_vec[i] = (backProp.adam_beta_1 * backProp.prev_convLayer1Kernel14Depth4_m_vec[i]) + ((1.0 - backProp.adam_beta_1) * gradient[i]);

                    backProp.convLayer1Kernel14Depth4_m_hat_vec[i] = backProp.convLayer1Kernel14Depth4_m_vec[i] / (1.0 - Math.Pow(backProp.adam_beta_1, predictorGui.iterationIdx + 1));
                }

                p_t = p_infinity - (2.0 * (predictorGui.iterationIdx + 1) * Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1));
                if (p_t > 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        adaptive_learning_rate[i] = Math.Sqrt((1.0 - Math.Pow(backProp.adam_beta_2, predictorGui.iterationIdx + 1)) / (backProp.convLayer1Kernel14Depth4_v_vec[i] + backProp.epsilon));
                        rectify_term[i] = Math.Sqrt(((p_t - 4) * (p_t - 2) * p_infinity) / ((p_infinity - 4) * (p_infinity - 2) * p_t + backProp.epsilon));
                        temp[i] = backProp.initial_learning_rate * rectify_term[i] * backProp.convLayer1Kernel14Depth4_m_hat_vec[i] * adaptive_learning_rate[i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        temp[i] = backProp.initial_learning_rate * backProp.convLayer1Kernel14Depth4_m_hat_vec[i];
                    }
                }
                Array.Copy(backProp.convLayer1Kernel14Depth4_m_vec, 0, backProp.prev_convLayer1Kernel14Depth4_m_vec, 0, 32);
                Array.Copy(backProp.convLayer1Kernel14Depth4_v_vec, 0, backProp.prev_convLayer1Kernel14Depth4_v_vec, 0, 32);
                Array.Copy(temp, 0, backProp.convLayer1Kernel14Depth4_adapted_rate, 0, 32);
            }
        }

        public void deconcatenateFilteredValMatrices()
        {
            int deconcatIdx = 0;
            int deconcatMat1Idx = 0;
            int deconcatMat2Idx = 0;
            int deconcatMat3Idx = 0;
            for (int j = 0; j < 1500; j++)
            {
                if (deconcatIdx < 5)
                {
                    Array.Copy(backProp.error_of_concat_filteredVal_output, j, backProp.deconcatMat1, deconcatMat1Idx, 1);
                    deconcatIdx++;
                    deconcatMat1Idx++;
                }
                else if (deconcatIdx >= 5 && deconcatIdx < 10)
                {
                    Array.Copy(backProp.error_of_concat_filteredVal_output, j, backProp.deconcatMat2, deconcatMat2Idx, 1);
                    deconcatIdx++;
                    deconcatMat2Idx++;
                }
                else if (deconcatIdx >= 10 && deconcatIdx < 15)
                {
                    Array.Copy(backProp.error_of_concat_filteredVal_output, j, backProp.deconcatMat3, deconcatMat3Idx, 1);
                    deconcatIdx++;
                    deconcatMat3Idx++;
                }

                if(deconcatIdx == 15)
                {
                    deconcatIdx = 0;
                }
            }
        }
    }
}
