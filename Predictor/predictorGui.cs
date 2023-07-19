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
using System.ComponentModel;
using System.Configuration;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Windows.Forms;
using System.IO;
using System.Drawing.Drawing2D;
using System.Data.SqlClient;
using System.Runtime.InteropServices;
using ManagedCuda;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.Collections;

namespace Predictor
{
    public partial class predictorGui : Form
    {
        public static predictorGui predictorGui1;
        public static Thread guiThread = new Thread(PredictorThreadWork.appOpen);

        public static bool tensorFound = false;
        public static BackgroundWorker dataInputCtrl = new BackgroundWorker();
        public static BackgroundWorker convFilteringCtrl = new BackgroundWorker();
        public static BackgroundWorker transformerCtrl = new BackgroundWorker();
        public static bool priceSizeFlag = true;
        public static int numEvents = 100;
        public static int tensorIdx = 0;
        public static int exampleIdx = 0;
        public static int trainingExNum = 0;
        public static int miniBatchIdx = 0;
        public static int iterationIdx = 0;
        public static int iterationIdx2 = 0;
        public static int epochIdx = 0;

        public static matrixOps matOps = new matrixOps();

        public static int startingExIdx = 0;

        public static bool flatExCaptureFlag = false;
        public static bool flatExIgnoreFlag = false;
        public static int upOrDownIdx = 0;

        public static int numOfLearnableParams = 0;
        public static int numOfUpExamples = 0;
        public static int numOfFlatExamples = 0;
        public static int numOfDownExamples = 0;

        public static double globalMaxPrice = 0;
        public static double globalMinPrice = 0;
        public static double globalMaxSize = 0;
        public static double globalMinSize = 0;

        public static double globalScaledMaxPrice = 0;
        public static double globalScaledMinPrice = 0;
        public static double globalScaledMaxSize = 0;
        public static double globalScaledMinSize = 0;

        public static double liveAvgCrossEntLoss = 0;
        public static double avgCrossEntropyPerBatch = 0;
        public static double currentAvgCrossEntLoss = 0;
        public static double prevAvgCrossEntropyPerBatch = 0;

        public static double liveAvgCrossEntLossEpoch = 0;
        public static double avgCrossEntropyLossPerEpoch = 0;
        public static double currentAvgCrossEntLossPerEpoch = 0;
        public static double prevAvgCrossEntropyLossPerEpoch = 0;

        public static int miniBatchSize = 32;

        public static ExampleArray[] exampleArray = new ExampleArray[miniBatchSize];

        public static Random rand = new Random();

        public static int percentIgnore = 5;

        public static weightsAdjList[] adjustmentList = new weightsAdjList[miniBatchSize];
        public static inputTensor[] tensorIn = new inputTensor[miniBatchSize];
        public static EventArray[] totalEventsArray = new EventArray[miniBatchSize];
        public static inputTensor prevTensorIn = new inputTensor();
        public static convLayer convModule = new convLayer();
        public static Transformer_Implementation transformerModule = new Transformer_Implementation();
        public static backProp backProp = new backProp();
        public static MLP mlp = new MLP();

        public static nnStructsArray[] networkArray = new nnStructsArray[miniBatchSize];

        public static ArrayList entireDaysPrices = new ArrayList();
        public static ArrayList entireDaysSizes = new ArrayList();

        public static int dayNum = 0;

        public static bool prevMidPoint = true;
        public static bool currentMidPoint = false;
        public static bool changeExNum = false;
        public static bool changeIterNumFlag = false;
        public static bool changeEpochNumFlag = false;
        public static bool changePercentIgnoredNumFlag = false;

        public static double[] prevPrediction = new double[3];

        public static bool trainingActivated = false;
        public static bool trainingBackProp = false;
        public static bool trainingGA = false;
        public static bool runOnce = false;
        public static bool runOnce2 = false;

        public static double epochTimeSeconds = 0;

        public static int selectGpu = 0;

        public predictorGui()
        {
            InitializeComponent();
        }

        private void predictorGui_Load(object sender, EventArgs e)
        {
            this.Location = Screen.AllScreens[0].WorkingArea.Location;
            //this.transformer_gif.Image = Image.FromFile("transformers-optimus-prime.gif");

            minibatchExSelect.Text = "0";

            label41.Text += "(" + percentIgnore.ToString() + ")";

            //check for dayNum existence and set dayNum
            if(File.Exists(@"X:\previousDayData\dayNum.txt"))
            {
                string[] arr = File.ReadAllLines(@"X:\previousDayData\dayNum.txt");
                dayNum = Convert.ToInt32(arr[0]);
                dayNum++;
            }

            for(int i = 0; i < 32; i++)
            {
                adjustmentList[i] = new weightsAdjList();
                networkArray[i] = new nnStructsArray();
                exampleArray[i] = new ExampleArray();
                tensorIn[i] = new inputTensor();
                totalEventsArray[i] = new EventArray();

                //initialize events array with passed in events param
                //we will leave first event empty as our causal padding
                for (int k = 0; k < numEvents + 1; k++)
                {
                    totalEventsArray[i].eventsArray[k] = new Event();
                }

                for (int j = 0; j < 32; j++)
                {
                    networkArray[i].convStructs[j] = new nnConvStructs();
                    networkArray[i].transStructs[j] = new nnTransStructs();
                    networkArray[i].mlpStructs[j] = new nnMLPStructs();
                }
            }

            dataInputCtrl.DoWork += dataInputCtrl_DoWork;
            dataInputCtrl.RunWorkerCompleted += dataInputCtrl_RunWorkerCompleted;
            convFilteringCtrl.DoWork += convFilteringCtrl_DoWork;
            convFilteringCtrl.RunWorkerCompleted += convFilteringCtrl_RunWorkerCompleted;
            transformerCtrl.DoWork += transformerCtrl_DoWork;
            transformerCtrl.RunWorkerCompleted += transformerCtrl_RunWorkerCompleted;

            //initialize weights tensor, PReLU params, and biases only once on start up
            //if weights flat files exist the network will initialize using saved weights
            convModule.lecun_normal_init_layer(1);
            convModule.lecun_normal_init_layer(2);
            convModule.lecun_normal_init_layer(3);
            convModule.lecun_normal_init_layer(4);
            convModule.lecun_normal_init_layer(5);
            convModule.convLayerBiases_init();
            convModule.convLayerPReLUParams_init();
            convModule.convLayerNormGammaBetaInit();
            transformerModule.tfixupInit_attention_linearLayer(1);
            transformerModule.tfixup_init_affineMLPLayers(1);
            mlp.xavier_init_weights(1);
            mlp.xavier_init_weights(2);
            mlp.mlpLayerBiases_init1();
            mlp.mlpLayerPReLUParams_init1();
            transformerModule.addAndNormGammaBetaInit();
            transformerModule.transMLPBiases_init();
            transformerModule.transMLPPReLUParams_init();

            //initialize weights for training networks in GA mode
            //Parallel.For(0, 32, (i, state) =>
            for(int i = 0; i < 32; i++)
            {
                convModule.lecun_normal_init_layer_GA(i, 1);
                convModule.lecun_normal_init_layer_GA(i, 2);
                convModule.lecun_normal_init_layer_GA(i, 3);
                convModule.lecun_normal_init_layer_GA(i, 4);
                convModule.lecun_normal_init_layer_GA(i, 5);
                convModule.convLayerBiases_init_GA(i);
                convModule.convLayerPReLUParams_init_GA(i);
                convModule.convLayerNormGammaBetaInit_GA(i);

                transformerModule.tfixupInit_attention_linearLayer_GA(i, 1);
                transformerModule.tfixup_init_affineMLPLayers_GA(i, 1);

                transformerModule.addAndNormGammaBetaInit_GA(i);
                transformerModule.transMLPBiases_init_GA(i);
                transformerModule.transMLPPReLUParams_init_GA(i);

                mlp.xavier_init_weights_GA(i, 1);
                mlp.xavier_init_weights_GA(i, 2);
                mlp.mlpLayerBiases_init1_GA(i);
                mlp.mlpLayerPReLUParams_init1_GA(i);
            }//);

            //display the available devices being used by the predictor
            int cudaDevID = selectGpu;
            CudaContext ctx = new CudaContext(0);
            CudaDeviceProperties props;
            props = ctx.GetDeviceInfo();

            devs_in_use.Text = "GPUs in use:" + "\n       Device ID: " + "0" + "\n       GPU Name: " + ctx.GetDeviceName() +
                "\n\nTotal Learnable Parameters: " + numOfLearnableParams.ToString() + "\nDay Number: " + dayNum.ToString();
            ctx.Dispose();
            dataInputCtrl.RunWorkerAsync();
        }

        public static void dataInputCtrl_DoWork(object sender, DoWorkEventArgs e)
        {
            Thread.Sleep(1000);
            if (predictorGui1.gifActivate.Checked == true)
            {
                predictorGui1.transformer_gif.Image = Image.FromFile("transformers-optimus-prime.gif");
            }
            else
            {
                predictorGui1.transformer_gif.Image = null;
            }
            predictorGui1.label1.Text = "Tensor Index = " + tensorIdx.ToString();
            if (!predictorGui1.exNum.Text.Equals("") && changeExNum == true)
            {
                exampleIdx = Convert.ToInt32(predictorGui1.exNum.Text);
                startingExIdx = exampleIdx;
                changeExNum = false;
                if(backProp.listOfTrainingExamples.Count != 0)
                {
                    backProp.listOfTrainingExamples.Clear();
                }
                backProp.createFileList(exampleIdx);
                if (predictorGui1.usingOverFlat.Checked == true)
                {
                    backProp.reduceFileList();
                }
                if (predictorGui1.usingUnderUp.Checked == true)
                {
                    backProp.increaseFileList();
                }
                predictorGui1.exampleClassCount();
                predictorGui1.numOfUpEx.Text = "Up Ex Count: " + numOfUpExamples.ToString() + " Global Max Price: " + Math.Round(globalMaxPrice, 3).ToString();
                predictorGui1.numOfFlatEx.Text = "Flat Ex Count: " + numOfFlatExamples.ToString() + " Global Min Price: " + Math.Round(globalMinPrice, 3).ToString();
                predictorGui1.numOfDownEx.Text = "Down Ex Count: " + numOfDownExamples.ToString() + " Global Max Size: " + Math.Round(globalMaxSize, 3).ToString() + "\n" +
                                                                                                           "\n                     Global Min Size: " + Math.Round(globalMinSize, 3).ToString();
                predictorGui1.exNum.Text = "";
            }
            if(!predictorGui1.iterNum.Text.Equals("") && changeIterNumFlag == true)
            {
                iterationIdx = Convert.ToInt32(predictorGui1.iterNum.Text);
                changeIterNumFlag = false;
                predictorGui1.iterNum.Text = "";
            }
            if (!predictorGui1.epochNum.Text.Equals("") && changeEpochNumFlag == true)
            {
                epochIdx = Convert.ToInt32(predictorGui1.epochNum.Text);
                changeEpochNumFlag = false;
                predictorGui1.epochNum.Text = "";
            }
            if(!predictorGui1.percentIgnoredText.Text.Equals("") && changePercentIgnoredNumFlag == true)
            {
                percentIgnore = Convert.ToInt32(predictorGui1.percentIgnoredText.Text);
                changePercentIgnoredNumFlag = false;
                predictorGui1.percentIgnoredText.Text = "";
            }
            predictorGui1.exNumLabel.Text = "Current Training Example Num: " + exampleIdx.ToString() + "      Tensor Index = " + tensorIdx.ToString() + "     Iteration Index = " + iterationIdx.ToString() + "     Epoch Index = " + epochIdx.ToString();

            if (trainingActivated == true && !File.Exists(@"X:\parsedTensorInput.txt"))
            {
                Parallel.For(0, 32, (i, state) =>
                {
                    if (File.Exists(@"X:\trainingData\trainingTensor" + backProp.listOfTrainingExamples[i] + ".ex.txt"))
                    {
                        //StreamWriter output = File.AppendText(@"X:\parsedTensorInput" + i + ".txt");
                        exampleArray[i].inputLines = File.ReadAllLines(@"X:\trainingData\trainingTensor" + backProp.listOfTrainingExamples[tensorIdx + i] + ".ex.txt");
                        string[] inputLines;
                        inputLines = File.ReadAllLines(@"X:\trainingData\trainingTensorExample" + backProp.listOfTrainingExamples[tensorIdx + i].ToString() + ".gt.txt");
                        exampleArray[i].actualOutcomes[0] = Convert.ToInt32(inputLines[0]);
                        exampleArray[i].actualOutcomes[1] = Convert.ToInt32(inputLines[1]);
                        exampleArray[i].actualOutcomes[2] = Convert.ToInt32(inputLines[2]);
                        //foreach (string line in exampleArray[i].inputLines)
                        //{
                        //    output.WriteLine(line);
                        //}
                        //output.Close();
                    }
                });
                tensorIdx += 32;
                predictorGui1.label1.Text = "Parsed Training Example Number " + backProp.listOfTrainingExamples[tensorIdx];
                predictorGui1.label3.Text = "Working on mini batch example " + miniBatchIdx.ToString();
                trainingExNum = Convert.ToInt32(backProp.listOfTrainingExamples[tensorIdx]);
                Thread.Sleep(1000);
            }

            if (dayNum == 0 && predictorGui1.buildTrdata.Checked == true)
            {
                predictorGui1.buildTrdata.Checked = false;
                predictorGui1.buildTrdata.Text = "Cannot build training data on day 0";
            }

            if(trainingActivated == true)
            {
                Parallel.For(0, 32, (i, state) =>
                {
                    int lineCount = 0;
                    foreach (string line in exampleArray[i].inputLines)
                    {
                        //write2.WriteLine(line);
                        string[] lineElements;
                        lineElements = exampleArray[i].inputLines[lineCount].Split(' ');
                        tensorIn[i].price[lineCount] = Convert.ToDouble(lineElements[0]);
                        tensorIn[i].size[lineCount] = Convert.ToDouble(lineElements[1]);
                        lineCount++;
                    }
                    predictorGui1.norm_inputs_resp_to_prev_day_mean_and_std(i);
                });

                predictorGui1.label1.Text = "Running convFilteringCtrl.\n";
                convFilteringCtrl.RunWorkerAsync();
            }

            if (File.Exists(@"X:\parsedTensorInput.txt") && trainingActivated == false)
            {
                tensorFound = true;
                exampleArray[0].inputLines = File.ReadAllLines(@"X:\parsedTensorInput.txt");
                //StreamWriter write2 = File.AppendText(@"X:\parsedTensor10" + tensorIdx.ToString() + ".txt");
                int lineCount = 0;
                foreach (string line in exampleArray[0].inputLines)
                {
                    //write2.WriteLine(line);
                    string[] lineElements;
                    lineElements = exampleArray[0].inputLines[lineCount].Split(' ');
                    tensorIn[0].price[lineCount] = Convert.ToDouble(lineElements[0]);
                    if (predictorGui1.accumPricesSizes.Checked == true && predictorGui1.activateTraining.Checked == false)
                    {
                        entireDaysPrices.Add(tensorIn[0].price[lineCount]);
                    }
                    tensorIn[0].size[lineCount] = Convert.ToDouble(lineElements[1]);
                    if(predictorGui1.accumPricesSizes.Checked == true && predictorGui1.activateTraining.Checked == false)
                    {
                        entireDaysSizes.Add(tensorIn[0].size[lineCount]);
                    }
                    lineCount++;
                }
                //write2.Close();

                //we do not want normalization to happen while we are building the training data
                if (predictorGui1.buildTrdata.Checked == false)
                {
                    if ((predictorGui1.accumPricesSizes.Checked == false || dayNum == 0) && predictorGui1.activateTraining.Checked == false)
                    {
                        predictorGui1.norm_inputs_resp_to_input_mean_and_std(0);
                        if (predictorGui1.enableOutputs.Checked == true)
                        {
                            StreamWriter tensorInOut = File.AppendText(@"X:\tensorInNormPerEx.txt");
                            for (int i = 0; i < 3200; i++)
                            {
                                tensorInOut.WriteLine(tensorIn[0].price[i].ToString() + ' ' + tensorIn[0].size[i].ToString());
                            }
                            tensorInOut.Close();
                        }
                    }
                    else if(dayNum != 0 || predictorGui1.activateTraining.Checked == true)
                    {
                        predictorGui1.norm_inputs_resp_to_prev_day_mean_and_std(0);
                        if (predictorGui1.enableOutputs.Checked == true)
                        {
                            StreamWriter tensorInOut = File.AppendText(@"X:\tensorInNormPrevDayData.txt");
                            for (int i = 0; i < 3200; i++)
                            {
                                tensorInOut.WriteLine(tensorIn[0].price[i].ToString() + ' ' + tensorIn[0].size[i].ToString());
                            }
                            tensorInOut.Close();
                        }
                    }
                }

                predictorGui1.label1.Text = "Tensor input file processed\n";
                predictorGui1.label2.Text = "";
                if (trainingActivated != true)
                {
                    predictorGui1.label3.Text = "";
                }
                predictorGui1.transOut.Text = "";
                predictorGui1.mlpOut.Text = "";
                predictorGui1.Update();
                Thread.Sleep(1000);
                lineCount = 0;
                try
                {
                    if (!Directory.Exists(@"X:\trainingData") && predictorGui1.buildTrdata.Checked == true)
                    {
                        Directory.CreateDirectory(@"X:\trainingData");
                    }
                    //calculate midpoint of current tensor for use with cross entropy loss
                    if (prevMidPoint == true && currentMidPoint == false && tensorIdx == 0 && predictorGui1.buildTrdata.Checked == true)
                    {
                        int midPointIdx = 15;
                        for (int i = 0; i < 100; i++)
                        {
                            backProp.midPointCompareSmoothed1 += ((tensorIn[0].price[midPointIdx] + tensorIn[0].price[midPointIdx + 1]) / 2);
                            midPointIdx += 32;
                        }
                        backProp.midPointCompareSmoothed1 /= numEvents;
                        midPointIdx -= 32;
                        backProp.midPointCompare1 = (tensorIn[0].price[midPointIdx] + tensorIn[0].price[midPointIdx + 1]) / 2;
                        predictorGui1.label4.Text = "Current Midpoint: " + backProp.midPointCompare1;

                        //save off first tensor to come in
                        Array.Copy(tensorIn[0].price, 0, prevTensorIn.price, 0, 3200);
                        Array.Copy(tensorIn[0].size, 0, prevTensorIn.size, 0, 3200);

                        prevMidPoint = false;
                        currentMidPoint = true;
                    }
                    else if (prevMidPoint == false && currentMidPoint == true && tensorIdx == 1 && predictorGui1.buildTrdata.Checked == true)
                    {
                        int midPointIdx = 15;
                        for (int i = 0; i < 100; i++)
                        {
                            backProp.midPointCompareSmoothed2 += ((tensorIn[0].price[midPointIdx] + tensorIn[0].price[midPointIdx + 1]) / 2);
                            midPointIdx += 32;
                        }
                        backProp.midPointCompareSmoothed2 /= numEvents;
                        midPointIdx -= 32;
                        backProp.midPointCompare2 = (tensorIn[0].price[midPointIdx] + tensorIn[0].price[midPointIdx + 1]) / 2;

                        //calculate percentage change between second tensor's smoothed mean vs first tensor's smoothed mean
                        backProp.pricePercentChange = (backProp.midPointCompareSmoothed2 - backProp.midPointCompare1) / backProp.midPointCompare1;

                        predictorGui1.label3.Text = "Percentage Change of Price = " + backProp.pricePercentChange.ToString();
                        predictorGui1.label4.Text = "Current Midpoint: " + backProp.midPointCompare2 + "\n" + "Previous Midpoint: " + backProp.midPointCompare1 +
                            "\nSmoothed Midpoint: " + backProp.midPointCompareSmoothed2;

                        //calculate price directionality and create one hot encoding
                        backProp.priceDirectionality(exampleIdx, false);

                        predictorGui1.prevPredOutputs.Text = "Previous Predicted Outputs: UP: " + Math.Round(prevPrediction[0], 3).ToString() + "  FLAT: " +
                            Math.Round(prevPrediction[1], 3).ToString() + "  DOWN: " + Math.Round(prevPrediction[2], 3).ToString();
                        predictorGui1.oneHotOut.Text = "Corresponding One Hot: UP: " + backProp.actualOutcomes[0].ToString() + "  FLAT: " + backProp.actualOutcomes[1].ToString() +
                            "   DOWN: " + backProp.actualOutcomes[2].ToString();

                        int randomVal0To99 = rand.Next(100);
                        if(randomVal0To99 < percentIgnore)
                        {
                            flatExCaptureFlag = true;
                        }

                        if ((predictorGui1.ignoreUpEx.Checked == false && backProp.actualOutcomes[0] == 1) || backProp.actualOutcomes[2] == 1)
                        {
                            backProp.buildTrainingDataExamples(exampleIdx, prevTensorIn);
                            upOrDownIdx++;
                            exampleIdx++;
                            backProp.cross_entropy_loss_per_Ex(exampleIdx);
                            predictorGui1.crossEntLoss.Text = "Cross Entropy Loss for Ex. Num: " + backProp.cross_entropy_loss.ToString();
                        }
                        else if (upOrDownIdx >= 2 && backProp.actualOutcomes[1] == 1 && flatExCaptureFlag == true)
                        {
                            backProp.buildTrainingDataExamples(exampleIdx, prevTensorIn);
                            if (flatExCaptureFlag == false && (upOrDownIdx >= 2 && backProp.actualOutcomes[1] == 1))
                            {
                                upOrDownIdx = 0;
                            }
                            exampleIdx++;
                            backProp.cross_entropy_loss_per_Ex(exampleIdx);
                            predictorGui1.crossEntLoss.Text = "Cross Entropy Loss for Ex. Num: " + backProp.cross_entropy_loss.ToString();
                        }
                        else
                        {
                            predictorGui1.crossEntLoss.Text = "Flat example ignored. RNG Num: " + randomVal0To99.ToString();
                            flatExIgnoreFlag = true;
                        }
                        flatExCaptureFlag = false;
                        //load prevTensor with the current tensor as this will become the prevTensor in the next pass
                        Array.Copy(tensorIn[0].price, 0, prevTensorIn.price, 0, 3200);
                        Array.Copy(tensorIn[0].size, 0, prevTensorIn.size, 0, 3200);

                        prevMidPoint = true;
                        currentMidPoint = false;
                    }
                    else if (prevMidPoint == true && currentMidPoint == false && tensorIdx > 1 && predictorGui1.buildTrdata.Checked == true)
                    {
                        int midPointIdx = 15;
                        backProp.midPointCompareSmoothed1 = backProp.midPointCompareSmoothed2;
                        backProp.midPointCompare1 = backProp.midPointCompare2;
                        backProp.midPointCompareSmoothed2 = 0;
                        for (int i = 0; i < 100; i++)
                        {
                            backProp.midPointCompareSmoothed2 += ((tensorIn[0].price[midPointIdx] + tensorIn[0].price[midPointIdx + 1]) / 2);
                            midPointIdx += 32;
                        }
                        backProp.midPointCompareSmoothed2 /= numEvents;
                        midPointIdx -= 32;
                        backProp.midPointCompare2 = (tensorIn[0].price[midPointIdx] + tensorIn[0].price[midPointIdx + 1]) / 2;

                        //calculate percentage change between second tensor's smoothed mean vs first tensor's smoothed mean
                        backProp.pricePercentChange = (backProp.midPointCompareSmoothed2 - backProp.midPointCompare1) / backProp.midPointCompare1;

                        predictorGui1.label3.Text = "Percentage Change of Price = " + backProp.pricePercentChange.ToString();
                        predictorGui1.label4.Text = "Current Midpoint: " + backProp.midPointCompare2 + "\n" + "Previous Midpoint: " + backProp.midPointCompare1 + 
                            "\nSmoothed Midpoint: " + backProp.midPointCompareSmoothed2;

                        //calculate price directionality and populate array starting at next idx
                        backProp.priceDirectionality(exampleIdx, false);

                        predictorGui1.prevPredOutputs.Text = "Previous Predicted Outputs: UP: " + Math.Round(prevPrediction[0], 3).ToString() + "  FLAT: " +
                            Math.Round(prevPrediction[1], 3).ToString() + "  DOWN: " + Math.Round(prevPrediction[2], 3).ToString();
                        predictorGui1.oneHotOut.Text = "Corresponding One Hot: UP: " + backProp.actualOutcomes[0].ToString() + "  FLAT: " + backProp.actualOutcomes[1].ToString() +
                            "   DOWN: " + backProp.actualOutcomes[2].ToString();

                        int randomVal0To99 = rand.Next(100);
                        if(randomVal0To99 < percentIgnore)
                        {
                            flatExCaptureFlag = true;
                        }

                        if ((predictorGui1.ignoreUpEx.Checked == false && backProp.actualOutcomes[0] == 1) || backProp.actualOutcomes[2] == 1)
                        {
                            backProp.buildTrainingDataExamples(exampleIdx, prevTensorIn);
                            upOrDownIdx++;
                            exampleIdx++;
                            backProp.cross_entropy_loss_per_Ex(exampleIdx);
                            predictorGui1.crossEntLoss.Text = "Cross Entropy Loss for Ex. Num: " + backProp.cross_entropy_loss.ToString();
                        }
                        else if (upOrDownIdx >= 2 && backProp.actualOutcomes[1] == 1 && flatExCaptureFlag == true)
                        {
                            backProp.buildTrainingDataExamples(exampleIdx, prevTensorIn);
                            if (flatExCaptureFlag == false && (upOrDownIdx >= 2 && backProp.actualOutcomes[1] == 1))
                            {
                                upOrDownIdx = 0;
                            }
                            exampleIdx++;
                            backProp.cross_entropy_loss_per_Ex(exampleIdx);
                            predictorGui1.crossEntLoss.Text = "Cross Entropy Loss for Ex. Num: " + backProp.cross_entropy_loss.ToString();
                        }
                        else
                        {
                            predictorGui1.crossEntLoss.Text = "Flat example ignored. RNG Num: " + randomVal0To99.ToString();
                            flatExIgnoreFlag = true;
                        }
                        flatExCaptureFlag = false;

                        //load prevTensor with the current tensor as this will become the prevTensor in the next pass
                        Array.Copy(tensorIn[0].price, 0, prevTensorIn.price, 0, 3200);
                        Array.Copy(tensorIn[0].size, 0, prevTensorIn.size, 0, 3200);
                    }
                    /*
                    StreamWriter write = File.AppendText(@"X:\inputTensor" + tensorIdx.ToString() + ".txt");
                    for (int j = 0; j < (32 * numEvents); j++)
                    {
                        write.WriteLine(tensorIn.price[j].ToString() + " " + tensorIn.size[j].ToString());
                    }
                    write.Close();
                    */
                    File.Delete(@"X:\parsedTensorInput.txt");
                }
                catch
                {
                    predictorGui1.label1.Text += "Could not delete parsedTensorInput.txt";
                }
                predictorGui1.label1.Text = "Deleted input file\n";
                predictorGui1.Update();
                convFilteringCtrl.RunWorkerAsync();
                Thread.Sleep(1000);
                if (predictorGui1.activateTraining.Checked == false)
                {
                    tensorIdx++;
                }
            }
            else
            {
                tensorFound = false;
            }
            predictorGui1.Update();
        }

        private static void dataInputCtrl_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (predictorGui1.activateTraining.Checked == true)
            {
                trainingActivated = true;
                if(runOnce != true)
                {
                    runOnce = true;
                    dataInputCtrl.RunWorkerAsync();
                }
            }
            else if (predictorGui1.buildTrdata.Checked == true)
            {
                if (runOnce2 != true)
                {
                    runOnce2 = true;
                }
                dataInputCtrl.RunWorkerAsync();
            }
            else
            {
                dataInputCtrl.RunWorkerAsync();
            }
        }

        public static void convFilteringCtrl_DoWork(object sender, DoWorkEventArgs e)
        {
            var watch = Stopwatch.StartNew();
            if (trainingActivated == true)
            {
                Parallel.For(0, 32, (i, state) =>
                {
                    convModule.convolution1D(numEvents, i, true);
                });
            }
            else
            {
                convModule.convolution1D(numEvents, 0, false);
            }
            watch.Stop();
            predictorGui1.label2.Text = "Convolutions completed in " + ((double)watch.ElapsedMilliseconds / 1000F).ToString() + " seconds.";
        }

        public static void convFilteringCtrl_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            predictorGui1.label1.Text = "Running transformerCtrl.\n";
            if (predictorGui1.enableConvVisual.Checked == true)
            {
                predictorGui1.convLayer5OutShow();
            }
            transformerCtrl.RunWorkerAsync();
        }

        public static void transformerCtrl_DoWork(object sender, DoWorkEventArgs e)
        {
            var watch = Stopwatch.StartNew();
            if (trainingActivated == true)
            {
                Parallel.For(0, 32, (i, state) =>
                {
                    Parallel.For(0, 32, (k, state2) =>
                    {
                        transformerModule.positionalEncoding(i, k, 1);

                        Parallel.For(1, 4, (j, state3) =>
                        {
                            transformerModule.attentionHeads(i, k, j);
                        });

                        transformerModule.scaleAndSoftmax_with_masking(i, k);

                        Parallel.For(1, 4, (j, state3) =>
                        {
                            transformerModule.matMulFilteredValueMat(i, k, j);
                        });

                        transformerModule.concatFilteredValMats(i, k);
                        transformerModule.finalAttentionBlockLinearLayer(i, k);
                        transformerModule.addAndNormLayer(i, k, 1, 1);
                        transformerModule.affineTransformMLP(i, k, 1);
                        transformerModule.addAndNormLayer(i, k, 2, 1);

                        Array.Copy(networkArray[i].transStructs[k].transformerBlockFinalOutput, 0, networkArray[i].transStructs[k].transformerBlock1Output, 0, 1500);

                        //block 2 start
                        transformerModule.positionalEncoding(i, k, 2);
                        Parallel.For(1, 4, (j, state3) =>
                        {
                            transformerModule.attentionHeads(i, k, j);
                        });

                        transformerModule.scaleAndSoftmax_with_masking(i, k);

                        Parallel.For(1, 4, (j, state3) =>
                        {
                            transformerModule.matMulFilteredValueMat(i, k, j);
                        });

                        transformerModule.concatFilteredValMats(i, k);
                        transformerModule.finalAttentionBlockLinearLayer(i, k);
                        transformerModule.addAndNormLayer(i, k, 1, 2);
                        transformerModule.affineTransformMLP(i, k, 2);
                        transformerModule.addAndNormLayer(i, k, 2, 2);

                        Array.Copy(networkArray[i].transStructs[k].transformerBlockFinalOutput, 0, networkArray[i].transStructs[k].transformerBlock2Output, 0, 1500);

                        //StreamWriter output = File.AppendText(@"X:\debugOutput\transformerBlockFinalOutput" + i + "-" + k + ".txt");
                        //for (int m = 0; m < 1500; m++)
                        //{
                        //    output.WriteLine(predictorGui.networkArray[i].transStructs[k].transformerBlockFinalOutput[m].ToString());
                        //}
                        //output.Close();
                    });
                });
            }
            else
            {
                double[] temp;
                transformerModule.positionalEncoding(0, 0, 1);
                if (predictorGui1.posEncodingVisualEbl.Checked == true)
                {
                    predictorGui1.posEncodingShow();
                }
                Parallel.For(1, 4, (j, state) =>
                {
                    transformerModule.attentionHeads(0, 0, j);
                });
                //transformerModule.attentionHeads(1);
                //transformerModule.attentionHeads(2);
                //transformerModule.attentionHeads(3);
                transformerModule.scaleAndSoftmax_with_masking(0, 0);
                Parallel.For(1, 4, (j, state) =>
                {
                    transformerModule.matMulFilteredValueMat(0, 0, j);
                });
                //transformerModule.matMulFilteredValueMat(1);
                //transformerModule.matMulFilteredValueMat(2);
                //transformerModule.matMulFilteredValueMat(3);
                transformerModule.concatFilteredValMats(0, 0);
                transformerModule.finalAttentionBlockLinearLayer(0, 0);
                transformerModule.addAndNormLayer(0, 0, 1, 1);
                transformerModule.affineTransformMLP(0, 0, 1);
                transformerModule.addAndNormLayer(0, 0, 2, 1);

                if (predictorGui1.attenFiltEbl.Checked == true)
                {
                    predictorGui1.transFilterShow(1);
                    predictorGui1.Update();
                }

                if (trainingBackProp == true)
                {
                    //retranspose key matrices in order to do backpropagation correctly
                    temp = matOps.transposeMat(networkArray[0].transStructs[0].key_head1, 100, 5);
                    Array.Copy(temp, 0, networkArray[0].transStructs[0].key_head1, 0, 500);
                    temp = matOps.transposeMat(networkArray[0].transStructs[0].key_head2, 100, 5);
                    Array.Copy(temp, 0, networkArray[0].transStructs[0].key_head2, 0, 500);
                    temp = matOps.transposeMat(networkArray[0].transStructs[0].key_head3, 100, 5);
                    Array.Copy(temp, 0, networkArray[0].transStructs[0].key_head3, 0, 500);
                

                    //save off the first transformer block output and the entire state of the first transformer block 1
                    Array.Copy(networkArray[0].transStructs[0].transformerBlockFinalOutput, 0, networkArray[0].transStructs[0].transformerBlock1Output, 0, 1500);
                    Array.Copy(networkArray[0].transStructs[0].residualConnectionOutputNorm, 0, networkArray[0].transStructs[0].residualConnectionOutputNormCpy, 0, 1500);
                    backPropFunctions.makeTransformer1InputsCopy();
                }

                transformerModule.positionalEncoding(0, 0, 2);
                Parallel.For(1, 4, (j, state) =>
                {
                    transformerModule.attentionHeads(0, 0, j);
                });
                //transformerModule.attentionHeads(1);
                //transformerModule.attentionHeads(2);
                //transformerModule.attentionHeads(3);
                transformerModule.scaleAndSoftmax_with_masking(0, 0);
                Parallel.For(1, 4, (j, state) =>
                {
                    transformerModule.matMulFilteredValueMat(0, 0, j);
                });
                //transformerModule.matMulFilteredValueMat(1);
                //transformerModule.matMulFilteredValueMat(2);
                //transformerModule.matMulFilteredValueMat(3);
                transformerModule.concatFilteredValMats(0, 0);
                transformerModule.finalAttentionBlockLinearLayer(0, 0);
                transformerModule.addAndNormLayer(0, 0, 1, 2);
                transformerModule.affineTransformMLP(0, 0, 2);
                transformerModule.addAndNormLayer(0, 0, 2, 2);

                if (predictorGui1.attenFiltEbl.Checked == true)
                {
                    predictorGui1.transFilterShow(2);
                    predictorGui1.Update();
                }

                Array.Copy(networkArray[0].transStructs[0].transformerBlockFinalOutput, 0, networkArray[0].transStructs[0].transformerBlock2Output, 0, 1500);

                //retranspose key matrices in order to do backpropagation correctly
                temp = matOps.transposeMat(networkArray[0].transStructs[0].key_head1, 100, 5);
                Array.Copy(temp, 0, networkArray[0].transStructs[0].key_head1, 0, 500);
                temp = matOps.transposeMat(networkArray[0].transStructs[0].key_head2, 100, 5);
                Array.Copy(temp, 0, networkArray[0].transStructs[0].key_head2, 0, 500);
                temp = matOps.transposeMat(networkArray[0].transStructs[0].key_head3, 100, 5);
                Array.Copy(temp, 0, networkArray[0].transStructs[0].key_head3, 0, 500);

                //StreamWriter output = File.AppendText(@"X:\debugOutput\transformerBlockFinalOutput" + "0" + "-" + "0" + ".txt");
                //for (int m = 0; m < 1500; m++)
                //{
                //    output.WriteLine(predictorGui.networkArray[0].transStructs[0].transformerBlockFinalOutput[m].ToString());
                //}
                //output.Close();
            }

            watch.Stop();
            predictorGui1.transOut.Text = "Completed Transformer Blocks 1 and 2 in " + ((double)watch.ElapsedMilliseconds / 1000F).ToString() + " seconds.";
        }

        public static void transformerCtrl_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            var watch = Stopwatch.StartNew();
            predictorGui1.label1.Text = "Running MLP.\n";
            if (trainingActivated == true)
            {
                Parallel.For(0, 32, (i, state) =>
                {
                    Parallel.For(0, 32, (k, state2) =>
                    {
                        mlp.firstLayer(i, k);
                        mlp.secondLayer(i, k);

                        backProp.cross_entropy_loss_per_Ex_GA(i, k);
                    });
                });

                predictorGui1.label1.Text = "Outputting cross entropy losses.\n";
                StreamWriter output = File.AppendText(@"X:\cross_entropy_losses_for_all_networks.txt");
                for (int i = 0; i < 32; i++)
                {
                    for (int j = 0; j < 32; j++)
                    {
                        output.WriteLine(networkArray[i].mlpStructs[j].cross_entropy_loss_per_example.ToString());
                    }
                }
                output.Close();
            }
            else
            {
                mlp.firstLayer(0, 0);
                mlp.secondLayer(0, 0);
                Array.Copy(networkArray[0].mlpStructs[0].secondLayerOut, 0, prevPrediction, 0, 3); //copy prediction into prevPrediction
                                                                                                   //this will be used to calculate the cross entropy loss of the previous prediction
                                                                                                   //when the predictor goes through another tensor
                if (predictorGui1.roundTo4.Checked == true)
                {
                    predictorGui1.upProb.Text = Math.Round(networkArray[0].mlpStructs[0].secondLayerOut[0], 4).ToString();
                    predictorGui1.flatProb.Text = Math.Round(networkArray[0].mlpStructs[0].secondLayerOut[1], 4).ToString();
                    predictorGui1.downProb.Text = Math.Round(networkArray[0].mlpStructs[0].secondLayerOut[2], 4).ToString();
                }
                else if (predictorGui1.roundTo3.Checked == true)
                {
                    predictorGui1.upProb.Text = Math.Round(networkArray[0].mlpStructs[0].secondLayerOut[0], 3).ToString();
                    predictorGui1.flatProb.Text = Math.Round(networkArray[0].mlpStructs[0].secondLayerOut[1], 3).ToString();
                    predictorGui1.downProb.Text = Math.Round(networkArray[0].mlpStructs[0].secondLayerOut[2], 3).ToString();
                }

                if (predictorGui1.buildTrdata.Checked == true && flatExIgnoreFlag == true)
                {
                    if (File.Exists(@"X:\trainingData\trainingTensorExample" + exampleIdx.ToString() + ".gt.txt"))
                    {
                        File.Delete(@"X:\trainingData\trainingTensorExample" + exampleIdx.ToString() + ".gt.txt");
                    }
                    flatExIgnoreFlag = false;
                }
            }
            watch.Stop();
            predictorGui1.mlpOut.Text = "Completed MLP in " + ((double)watch.ElapsedMilliseconds / 1000F).ToString() + " seconds.";
            if (trainingActivated == true && trainingBackProp == true)
            {
                backProp.priceDirectionality(trainingExNum, true);
                backProp.cross_entropy_loss_per_Ex(trainingExNum);
                avgCrossEntropyPerBatch += backProp.cross_entropy_losses_mini_batch[miniBatchIdx];
                liveAvgCrossEntLoss = avgCrossEntropyPerBatch;

                if (predictorGui1.outputLosses.Checked == true)
                {
                    StreamWriter trainingResOut = File.AppendText(@"X:\lossValuesPerExample.txt");
                    trainingResOut.WriteLine(trainingExNum.ToString() + "   " + backProp.cross_entropy_losses_mini_batch[miniBatchIdx]);
                    trainingResOut.Close();

                    StreamWriter output = File.AppendText(@"X:\finalPredictionProbabilities.txt");
                    output.WriteLine("Example Num: " + trainingExNum.ToString());
                    for (int i = 0; i < 3; i++)
                    {
                        output.WriteLine("Prediction Probabilities[" + i.ToString() + "] = " + networkArray[0].mlpStructs[0].secondLayerOut[i].ToString() +
                            "     " + "Expected Output[" + i.ToString() + "] = " + backProp.actualOutcomes[i].ToString());
                    }
                    output.WriteLine();
                    output.Close();
                }

                //run calculations for all weight adjustments here for this training example
                backProp.mlpCalculateAdjustments();
                backProp.affineMLPCalculateAdjustments(2);
                backProp.finalLinearLayerCalculateAdjustments(2);
                backProp.queryKeyValueCalculateAdjustments();

                //save off adjustments for this run
                Array.Copy(backProp.deltaFirstWeights, 0, adjustmentList[miniBatchIdx].mlpFirstLayerWeightsAdj, 0, 96000);
                Array.Copy(backProp.deltaSecondWeights, 0, adjustmentList[miniBatchIdx].mlpSecondLayerWeightsAdj, 0, 192);
                Array.Copy(backProp.deltaThirdWeights, 0, adjustmentList[miniBatchIdx].mlpThirdLayerWeightsAdj, 0, 9);
                Array.Copy(backProp.deltaFirstBias, 0, adjustmentList[miniBatchIdx].mlpLayer1BiasAdj, 0, 64);
                Array.Copy(backProp.deltaFirstPReLU, 0, adjustmentList[miniBatchIdx].mlpLayer1PReLUAdj, 0, 64);
                Array.Copy(backProp.deltaSecondBias, 0, adjustmentList[miniBatchIdx].mlpLayer2BiasAdj, 0, 3);
                Array.Copy(backProp.deltaSecondPReLU, 0, adjustmentList[miniBatchIdx].mlpLayer2PReLUAdj, 0, 3);

                Array.Copy(backProp.deltaAffineWeights2, 0, adjustmentList[miniBatchIdx].affineMLPSecondLayerWeightsPass1Block2Adj, 0, 900);
                Array.Copy(backProp.deltaAffineWeights1, 0, adjustmentList[miniBatchIdx].affineMLPFirstLayerWeightsPass1Block2Adj, 0, 900);
                Array.Copy(backProp.deltaAffineMLPBias1, 0, adjustmentList[miniBatchIdx].affineMLPFirstLayerBiasPass2Block2Adj, 0, 6000);
                Array.Copy(backProp.deltaAffineMLPPreluParam1, 0, adjustmentList[miniBatchIdx].affineMLPFirstLayerPreluPass2Block2Adj, 0, 6000);
                Array.Copy(backProp.deltaAffineMLPBias2, 0, adjustmentList[miniBatchIdx].affineMLPSecondLayerBiasBlock2Adj, 0, 1500);
                Array.Copy(backProp.deltaAddAndNormBeta2, 0, adjustmentList[miniBatchIdx].AddAndNormBeta2Adj, 0, 1500);
                Array.Copy(backProp.deltaAddAndNormGamma2, 0, adjustmentList[miniBatchIdx].AddAndNormGamma2Adj, 0, 1500);
                Array.Copy(backProp.deltaAddAndNormBeta1, 0, adjustmentList[miniBatchIdx].AddAndNormBeta1Adj, 0, 1500);
                Array.Copy(backProp.deltaAddAndNormGamma1, 0, adjustmentList[miniBatchIdx].AddAndNormGamma1Adj, 0, 1500);
                Array.Copy(backProp.deltaFinalLinearLayerWeights, 0, adjustmentList[miniBatchIdx].finalLinearLayerWeightsAdj, 0, 225);

                Array.Copy(backProp.deltaQueryWeightsHead1, 0, adjustmentList[miniBatchIdx].queryHead1LinearLayerWeightsAdj, 0, 75);
                Array.Copy(backProp.deltaKeyWeightsHead1, 0, adjustmentList[miniBatchIdx].keyHead1LinearLayerWeightsAdj, 0, 75);
                Array.Copy(backProp.deltaValueWeightsHead1, 0, adjustmentList[miniBatchIdx].valueHead1LinearLayerWeightsAdj, 0, 75);

                Array.Copy(backProp.deltaQueryWeightsHead2, 0, adjustmentList[miniBatchIdx].queryHead2LinearLayerWeightsAdj, 0, 75);
                Array.Copy(backProp.deltaKeyWeightsHead2, 0, adjustmentList[miniBatchIdx].keyHead2LinearLayerWeightsAdj, 0, 75);
                Array.Copy(backProp.deltaValueWeightsHead2, 0, adjustmentList[miniBatchIdx].valueHead2LinearLayerWeightsAdj, 0, 75);

                Array.Copy(backProp.deltaQueryWeightsHead3, 0, adjustmentList[miniBatchIdx].queryHead3LinearLayerWeightsAdj, 0, 75);
                Array.Copy(backProp.deltaKeyWeightsHead3, 0, adjustmentList[miniBatchIdx].keyHead3LinearLayerWeightsAdj, 0, 75);
                Array.Copy(backProp.deltaValueWeightsHead3, 0, adjustmentList[miniBatchIdx].valueHead3LinearLayerWeightsAdj, 0, 75);

                backProp.calculateErrorMatForTransBlock1();
                //reload inputs for transformer block 1 and rerun calculations
                backPropFunctions.reloadTransformer1Inputs();
                Array.Copy(networkArray[0].transStructs[0].transformerBlock1Output, 0, networkArray[0].transStructs[0].transformerBlock2Output, 0, 1500);
                backProp.affineMLPCalculateAdjustments(1);
                backProp.finalLinearLayerCalculateAdjustments(1);
                backProp.queryKeyValueCalculateAdjustments();

                Array.Copy(backProp.deltaAffineWeights2, 0, adjustmentList[miniBatchIdx].affineMLPSecondLayerWeightsPass1Block1Adj, 0, 900);
                Array.Copy(backProp.deltaAffineWeights1, 0, adjustmentList[miniBatchIdx].affineMLPFirstLayerWeightsPass1Block1Adj, 0, 900);
                Array.Copy(backProp.deltaAffineMLPBias1, 0, adjustmentList[miniBatchIdx].affineMLPFirstLayerBiasPass2Block1Adj, 0, 6000);
                Array.Copy(backProp.deltaAffineMLPPreluParam1, 0, adjustmentList[miniBatchIdx].affineMLPFirstLayerPreluPass2Block1Adj, 0, 6000);
                Array.Copy(backProp.deltaAffineMLPBias2, 0, adjustmentList[miniBatchIdx].affineMLPSecondLayerBiasBlock1Adj, 0, 1500);
                Array.Copy(backProp.deltaAddAndNormBeta2, 0, adjustmentList[miniBatchIdx].AddAndNormBeta2Adj1, 0, 1500);
                Array.Copy(backProp.deltaAddAndNormGamma2, 0, adjustmentList[miniBatchIdx].AddAndNormGamma2Adj1, 0, 1500);
                Array.Copy(backProp.deltaAddAndNormBeta1, 0, adjustmentList[miniBatchIdx].AddAndNormBeta1Adj1, 0, 1500);
                Array.Copy(backProp.deltaAddAndNormGamma1, 0, adjustmentList[miniBatchIdx].AddAndNormGamma1Adj1, 0, 1500);
                Array.Copy(backProp.deltaFinalLinearLayerWeights, 0, adjustmentList[miniBatchIdx].finalLinearLayerWeightsAdj1, 0, 225);

                Array.Copy(backProp.deltaQueryWeightsHead1, 0, adjustmentList[miniBatchIdx].queryHead1LinearLayerWeightsAdj1, 0, 75);
                Array.Copy(backProp.deltaKeyWeightsHead1, 0, adjustmentList[miniBatchIdx].keyHead1LinearLayerWeightsAdj1, 0, 75);
                Array.Copy(backProp.deltaValueWeightsHead1, 0, adjustmentList[miniBatchIdx].valueHead1LinearLayerWeightsAdj1, 0, 75);

                Array.Copy(backProp.deltaQueryWeightsHead2, 0, adjustmentList[miniBatchIdx].queryHead2LinearLayerWeightsAdj1, 0, 75);
                Array.Copy(backProp.deltaKeyWeightsHead2, 0, adjustmentList[miniBatchIdx].keyHead2LinearLayerWeightsAdj1, 0, 75);
                Array.Copy(backProp.deltaValueWeightsHead2, 0, adjustmentList[miniBatchIdx].valueHead2LinearLayerWeightsAdj1, 0, 75);

                Array.Copy(backProp.deltaQueryWeightsHead3, 0, adjustmentList[miniBatchIdx].queryHead3LinearLayerWeightsAdj1, 0, 75);
                Array.Copy(backProp.deltaKeyWeightsHead3, 0, adjustmentList[miniBatchIdx].keyHead3LinearLayerWeightsAdj1, 0, 75);
                Array.Copy(backProp.deltaValueWeightsHead3, 0, adjustmentList[miniBatchIdx].valueHead3LinearLayerWeightsAdj1, 0, 75);

                backProp.convModuleFilterAdjustments();

                Array.Copy(backProp.deltaConvLayer5Biases, 0, adjustmentList[miniBatchIdx].convLayer5BiasesAdj, 0, 1400);
                Array.Copy(backProp.deltaConvLayer5PReLUParams, 0, adjustmentList[miniBatchIdx].convLayer5PReLUParamAdj, 0, 1400);
                Array.Copy(backProp.deltaConvLayer5NormGamma, 0, adjustmentList[miniBatchIdx].convLayer5NormGammaAdj, 0, 1400);
                Array.Copy(backProp.deltaConvLayer5NormBeta, 0, adjustmentList[miniBatchIdx].convLayer5NormBetaAdj, 0, 1400);
                Array.Copy(backProp.deltaConvLayer4Biases, 0, adjustmentList[miniBatchIdx].convLayer4BiasesAdj, 0, 1400);
                Array.Copy(backProp.deltaConvLayer4PReLUParams, 0, adjustmentList[miniBatchIdx].convLayer4PReLUParamAdj, 0, 1400);
                Array.Copy(backProp.deltaConvLayer3Biases, 0, adjustmentList[miniBatchIdx].convLayer3BiasesAdj, 0, 1400);
                Array.Copy(backProp.deltaConvLayer3PReLUParams, 0, adjustmentList[miniBatchIdx].convLayer3PReLUParamAdj, 0, 1400);
                Array.Copy(backProp.deltaConvLayer2Biases, 0, adjustmentList[miniBatchIdx].convLayer2BiasesAdj, 0, 1400);
                Array.Copy(backProp.deltaConvLayer2PReLUParams, 0, adjustmentList[miniBatchIdx].convLayer2PReLUParamAdj, 0, 1400);
                Array.Copy(backProp.deltaConvLayer1Biases, 0, adjustmentList[miniBatchIdx].convLayer1BiasesAdj, 0, 1400);
                Array.Copy(backProp.deltaConvLayer1PReLUParams, 0, adjustmentList[miniBatchIdx].convLayer1PReLUParamAdj, 0, 1400);

                Array.Copy(backProp.deltaConvLayer5Kernel1_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel1Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel1_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel1Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel2_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel2Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel2_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel2Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel3_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel3Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel3_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel3Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel4_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel4Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel4_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel4Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel5_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel5Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel5_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel5Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel6_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel6Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel6_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel6Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel7_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel7Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel7_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel7Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel8_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel8Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel8_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel8Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel9_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel9Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel9_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel9Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel10_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel10Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel10_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel10Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel11_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel11Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel11_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel11Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel12_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel12Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel12_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel12Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel13_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel13Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel13_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel13Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel14_depth1, 0, adjustmentList[miniBatchIdx].convLayer5Kernel14Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer5Kernel14_depth2, 0, adjustmentList[miniBatchIdx].convLayer5Kernel14Depth2Adj, 0, 14);

                Array.Copy(backProp.deltaConvLayer4Kernel1_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel1Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel1_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel1Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel2_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel2Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel2_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel2Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel3_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel3Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel3_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel3Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel4_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel4Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel4_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel4Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel5_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel5Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel5_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel5Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel6_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel6Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel6_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel6Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel7_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel7Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel7_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel7Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel8_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel8Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel8_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel8Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel9_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel9Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel9_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel9Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel10_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel10Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel10_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel10Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel11_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel11Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel11_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel11Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel12_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel12Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel12_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel12Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel13_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel13Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel13_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel13Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel14_depth1, 0, adjustmentList[miniBatchIdx].convLayer4Kernel14Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer4Kernel14_depth2, 0, adjustmentList[miniBatchIdx].convLayer4Kernel14Depth2Adj, 0, 14);

                Array.Copy(backProp.deltaConvLayer3Kernel1_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel1Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel1_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel1Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel2_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel2Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel2_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel2Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel3_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel3Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel3_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel3Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel4_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel4Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel4_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel4Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel5_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel5Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel5_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel5Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel6_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel6Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel6_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel6Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel7_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel7Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel7_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel7Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel8_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel8Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel8_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel8Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel9_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel9Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel9_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel9Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel10_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel10Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel10_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel10Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel11_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel11Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel11_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel11Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel12_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel12Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel12_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel12Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel13_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel13Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel13_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel13Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel14_depth1, 0, adjustmentList[miniBatchIdx].convLayer3Kernel14Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer3Kernel14_depth2, 0, adjustmentList[miniBatchIdx].convLayer3Kernel14Depth2Adj, 0, 14);

                Array.Copy(backProp.deltaConvLayer2Kernel1_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel1Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel1_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel1Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel2_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel2Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel2_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel2Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel3_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel3Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel3_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel3Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel4_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel4Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel4_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel4Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel5_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel5Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel5_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel5Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel6_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel6Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel6_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel6Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel7_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel7Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel7_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel7Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel8_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel8Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel8_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel8Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel9_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel9Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel9_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel9Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel10_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel10Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel10_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel10Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel11_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel11Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel11_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel11Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel12_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel12Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel12_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel12Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel13_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel13Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel13_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel13Depth2Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel14_depth1, 0, adjustmentList[miniBatchIdx].convLayer2Kernel14Depth1Adj, 0, 14);
                Array.Copy(backProp.deltaConvLayer2Kernel14_depth2, 0, adjustmentList[miniBatchIdx].convLayer2Kernel14Depth2Adj, 0, 14);

                Array.Copy(backProp.deltaConvLayer1Kernel1_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel1Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel1_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel1Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel1_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel1Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel1_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel1Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel2_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel2Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel2_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel2Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel2_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel2Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel2_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel2Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel3_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel3Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel3_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel3Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel3_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel3Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel3_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel3Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel4_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel4Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel4_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel4Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel4_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel4Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel4_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel4Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel5_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel5Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel5_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel5Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel5_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel5Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel5_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel5Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel6_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel6Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel6_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel6Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel6_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel6Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel6_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel6Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel7_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel7Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel7_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel7Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel7_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel7Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel7_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel7Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel8_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel8Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel8_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel8Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel8_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel8Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel8_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel8Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel9_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel9Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel9_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel9Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel9_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel9Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel9_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel9Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel10_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel10Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel10_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel10Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel10_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel10Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel10_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel10Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel11_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel11Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel11_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel11Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel11_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel11Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel11_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel11Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel12_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel12Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel12_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel12Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel12_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel12Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel12_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel12Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel13_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel13Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel13_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel13Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel13_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel13Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel13_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel13Depth4Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel14_depth1, 0, adjustmentList[miniBatchIdx].convLayer1Kernel14Depth1Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel14_depth2, 0, adjustmentList[miniBatchIdx].convLayer1Kernel14Depth2Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel14_depth3, 0, adjustmentList[miniBatchIdx].convLayer1Kernel14Depth3Adj, 0, 32);
                Array.Copy(backProp.deltaConvLayer1Kernel14_depth4, 0, adjustmentList[miniBatchIdx].convLayer1Kernel14Depth4Adj, 0, 32);

                predictorGui1.crossEntLoss.Text = "Cross Entropy Loss for Ex. Num: " + trainingExNum + " = " + backProp.cross_entropy_loss.ToString();
                predictorGui1.prevPredOutputs.Text = "Previous Predicted Outputs: UP: " + Math.Round(prevPrediction[0], 3).ToString() + "  FLAT: " +
                        Math.Round(prevPrediction[1], 3).ToString() + "  DOWN: " + Math.Round(prevPrediction[2], 3).ToString();
                predictorGui1.oneHotOut.Text = "Corresponding One Hot: UP: " + backProp.actualOutcomes[0].ToString() + "  FLAT: " + backProp.actualOutcomes[1].ToString() +
                    "   DOWN: " + backProp.actualOutcomes[2].ToString();

                miniBatchIdx++;
                tensorIdx++;
                predictorGui1.label12.Text = "Avg. Loss across mini batch:\n" + "Live:   " + (liveAvgCrossEntLoss / miniBatchIdx).ToString() + "\n";
                
                if (miniBatchIdx == miniBatchSize)
                {
                    backPropFunctions funcs = new backPropFunctions();
                    avgCrossEntropyPerBatch /= miniBatchSize;
                    currentAvgCrossEntLoss = avgCrossEntropyPerBatch;
                    liveAvgCrossEntLossEpoch += avgCrossEntropyPerBatch;
                    iterationIdx++;
                    iterationIdx2++;
                    predictorGui1.label1.Text = "Performing Gradient Descent step.";
                    if (predictorGui1.outputLosses.Checked == true)
                    {
                        StreamWriter batchLossOut = File.AppendText(@"X:\lossValuesPerMinibatch.txt");
                        batchLossOut.WriteLine(iterationIdx.ToString() + "  " + currentAvgCrossEntLoss.ToString());
                        batchLossOut.Close();
                    }
                    if (tensorIdx == backProp.listOfTrainingExamples.Count)
                    {
                        currentAvgCrossEntLossPerEpoch = liveAvgCrossEntLossEpoch / (iterationIdx2);
                        predictorGui1.label14.Text = "Current:   " + currentAvgCrossEntLossPerEpoch.ToString() + "\n"
                            + "Previous: " + prevAvgCrossEntropyLossPerEpoch.ToString();
                        prevAvgCrossEntropyLossPerEpoch = currentAvgCrossEntLossPerEpoch;
                        //backProp.listOfTrainingExamples.Clear();
                        //backProp.createFileList(exampleIdx);
                        if (predictorGui1.usingOverFlat.Checked == true)
                        {
                            //backProp.reduceFileList();
                        }
                        backProp.shuffleFileList();
                        predictorGui1.exampleClassCount();
                        predictorGui1.numOfUpEx.Text = "Up Ex Count: " + numOfUpExamples.ToString();
                        predictorGui1.numOfFlatEx.Text = "Flat Ex Count: " + numOfFlatExamples.ToString();
                        predictorGui1.numOfDownEx.Text = "Down Ex Count: " + numOfDownExamples.ToString();
                        epochIdx++;
                        if (predictorGui1.outputLosses.Checked == true)
                        {
                            StreamWriter epochLossOut = File.AppendText(@"X:\lossValuesPerEpoch.txt");
                            epochLossOut.WriteLine(epochIdx.ToString() + "  " + currentAvgCrossEntLossPerEpoch.ToString());
                            epochLossOut.Close();
                        }
                    }
                    predictorGui1.label5.Text = "Current:   " + currentAvgCrossEntLoss.ToString() + "\n"
                        + "Previous: " + prevAvgCrossEntropyPerBatch.ToString();
                    predictorGui1.label13.Text = "Avg. Loss across epoch:\n" + "Live:   " + (liveAvgCrossEntLossEpoch / iterationIdx2).ToString() + "\n";

                    if(tensorIdx == backProp.listOfTrainingExamples.Count)
                    {
                        tensorIdx = 0;
                        iterationIdx2 = 0;
                        liveAvgCrossEntLossEpoch = 0;
                    }

                    for (int i = 0; i < 96000; i++)
                    {
                        for(int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].mlpFirstLayerWeightsAdj[i] += adjustmentList[k].mlpFirstLayerWeightsAdj[i];
                        }
                        adjustmentList[0].mlpFirstLayerWeightsAdj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].mlpFirstLayerWeightsAdj, "mlpFirstLayer", 0);
                    for (int i = 0; i < 96000; i++)
                    {
                        networkArray[0].mlpStructs[0].firstLayerWeights[i] -= backProp.mlpFirstLayer_adapted_rate[i];
                    }
                    for (int i = 0; i < 192; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].mlpSecondLayerWeightsAdj[i] += adjustmentList[k].mlpSecondLayerWeightsAdj[i];
                        }
                        adjustmentList[0].mlpSecondLayerWeightsAdj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].mlpSecondLayerWeightsAdj, "mlpSecondLayer", 0);
                    for (int i = 0; i < 192; i++)
                    {
                        networkArray[0].mlpStructs[0].secondLayerWeights[i] -= backProp.mlpSecondLayer_adapted_rate[i];
                    }
                    for (int i = 0; i < 9; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].mlpThirdLayerWeightsAdj[i] += adjustmentList[k].mlpThirdLayerWeightsAdj[i];
                        }
                        adjustmentList[0].mlpThirdLayerWeightsAdj[i] /= miniBatchSize;
                    }
                    for (int i = 0; i < 64; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].mlpLayer1BiasAdj[i] += adjustmentList[k].mlpLayer1BiasAdj[i];
                            adjustmentList[0].mlpLayer1PReLUAdj[i] += adjustmentList[k].mlpLayer1PReLUAdj[i];
                        }
                        adjustmentList[0].mlpLayer1BiasAdj[i] /= miniBatchSize;
                        adjustmentList[0].mlpLayer1PReLUAdj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].mlpLayer1BiasAdj, "mlpFirstLayerBiasPrelu", 0);
                    for (int i = 0; i < 64; i++)
                    {
                        networkArray[0].mlpStructs[0].mlpLayer1Bias[i] -= backProp.mlpFirstLayerBias_adapted_rate[i];
                        networkArray[0].mlpStructs[0].mlpLayer1PReLUParam[i] -= backProp.mlpFirstLayerBias_adapted_rate[i];
                    }
                    for (int i = 0; i < 900; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].affineMLPSecondLayerWeightsPass1Block2Adj[i] += adjustmentList[k].affineMLPSecondLayerWeightsPass1Block2Adj[i];
                            adjustmentList[0].affineMLPFirstLayerWeightsPass1Block2Adj[i] += adjustmentList[k].affineMLPFirstLayerWeightsPass1Block2Adj[i];

                            adjustmentList[0].affineMLPSecondLayerWeightsPass1Block1Adj[i] += adjustmentList[k].affineMLPSecondLayerWeightsPass1Block1Adj[i];
                            adjustmentList[0].affineMLPFirstLayerWeightsPass1Block1Adj[i] += adjustmentList[k].affineMLPFirstLayerWeightsPass1Block1Adj[i];
                        }
                        adjustmentList[0].affineMLPSecondLayerWeightsPass1Block2Adj[i] /= miniBatchSize;
                        adjustmentList[0].affineMLPFirstLayerWeightsPass1Block2Adj[i] /= miniBatchSize;

                        adjustmentList[0].affineMLPSecondLayerWeightsPass1Block1Adj[i] /= miniBatchSize;
                        adjustmentList[0].affineMLPFirstLayerWeightsPass1Block1Adj[i] /= miniBatchSize;

                        adjustmentList[0].affineMLPSecondLayerWeightsPass1Block2Adj[i] = (adjustmentList[0].affineMLPSecondLayerWeightsPass1Block2Adj[i] + adjustmentList[0].affineMLPSecondLayerWeightsPass1Block1Adj[i]) / 2;
                        adjustmentList[0].affineMLPFirstLayerWeightsPass1Block2Adj[i] = (adjustmentList[0].affineMLPFirstLayerWeightsPass1Block2Adj[i] + adjustmentList[0].affineMLPFirstLayerWeightsPass1Block1Adj[i]) / 2;
                        adjustmentList[0].affineMLPSecondLayerWeightsPass1Block1Adj[i] = (adjustmentList[0].affineMLPSecondLayerWeightsPass1Block2Adj[i] + adjustmentList[0].affineMLPSecondLayerWeightsPass1Block1Adj[i]) / 2;
                        adjustmentList[0].affineMLPFirstLayerWeightsPass1Block1Adj[i] = (adjustmentList[0].affineMLPFirstLayerWeightsPass1Block2Adj[i] + adjustmentList[0].affineMLPFirstLayerWeightsPass1Block1Adj[i]) / 2;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].affineMLPSecondLayerWeightsPass1Block2Adj, "affineMLPWeights2", 2);
                    funcs.rectified_adam_optimizer(adjustmentList[0].affineMLPFirstLayerWeightsPass1Block2Adj, "affineMLPWeights1", 2);
                    for (int i = 0; i < 900; i++)
                    {
                        networkArray[0].transStructs[0].affineTransWeights2[i] -= backProp.affineMLPWeights2Block2_adapted_rate[i];
                        networkArray[0].transStructs[0].affineTransWeights1[i] -= backProp.affineMLPWeights1Block2_adapted_rate[i];
                    }
                    for(int i = 0; i < 6000; i++)
                    {
                        for(int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].affineMLPFirstLayerBiasPass2Block2Adj[i] += adjustmentList[k].affineMLPFirstLayerBiasPass2Block2Adj[i];
                            adjustmentList[0].affineMLPFirstLayerBiasPass2Block1Adj[i] += adjustmentList[k].affineMLPFirstLayerBiasPass2Block1Adj[i];
                            adjustmentList[0].affineMLPFirstLayerPreluPass2Block2Adj[i] += adjustmentList[k].affineMLPFirstLayerPreluPass2Block2Adj[i];
                            adjustmentList[0].affineMLPFirstLayerPreluPass2Block1Adj[i] += adjustmentList[k].affineMLPFirstLayerPreluPass2Block1Adj[i];
                        }
                        adjustmentList[0].affineMLPFirstLayerBiasPass2Block2Adj[i] /= miniBatchSize;
                        adjustmentList[0].affineMLPFirstLayerBiasPass2Block1Adj[i] /= miniBatchSize;
                        adjustmentList[0].affineMLPFirstLayerPreluPass2Block2Adj[i] /= miniBatchSize;
                        adjustmentList[0].affineMLPFirstLayerPreluPass2Block1Adj[i] /= miniBatchSize;

                        adjustmentList[0].affineMLPFirstLayerBiasPass2Block2Adj[i] = (adjustmentList[0].affineMLPFirstLayerBiasPass2Block2Adj[i] + adjustmentList[0].affineMLPFirstLayerBiasPass2Block1Adj[i]) / 2;
                        adjustmentList[0].affineMLPFirstLayerPreluPass2Block2Adj[i] = (adjustmentList[0].affineMLPFirstLayerPreluPass2Block2Adj[i] + adjustmentList[0].affineMLPFirstLayerPreluPass2Block1Adj[i]) / 2;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].affineMLPFirstLayerBiasPass2Block2Adj, "affineMLPBiasPrelu", 2);
                    for (int i = 0; i < 6000; i++)
                    {
                        networkArray[0].transStructs[0].transPReLUBias[i] -= backProp.affineMLPBias2Block2_adapted_rate[i];
                        networkArray[0].transStructs[0].transPReLUParam[i] -= backProp.affineMLPBias2Block2_adapted_rate[i];
                    }
                    for (int i = 0; i < 1500; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].affineMLPSecondLayerBiasBlock2Adj[i] += adjustmentList[k].affineMLPSecondLayerBiasBlock2Adj[i];
                            adjustmentList[0].affineMLPSecondLayerBiasBlock1Adj[i] += adjustmentList[k].affineMLPSecondLayerBiasBlock1Adj[i];

                            adjustmentList[0].AddAndNormBeta2Adj[i] += adjustmentList[k].AddAndNormBeta2Adj[i];
                            adjustmentList[0].AddAndNormGamma2Adj[i] += adjustmentList[k].AddAndNormGamma2Adj[i];
                            adjustmentList[0].AddAndNormBeta1Adj[i] += adjustmentList[k].AddAndNormBeta1Adj[i];
                            adjustmentList[0].AddAndNormGamma1Adj[i] += adjustmentList[k].AddAndNormGamma1Adj[i];

                            adjustmentList[0].AddAndNormBeta2Adj1[i] += adjustmentList[k].AddAndNormBeta2Adj1[i];
                            adjustmentList[0].AddAndNormGamma2Adj1[i] += adjustmentList[k].AddAndNormGamma2Adj1[i];
                            adjustmentList[0].AddAndNormBeta1Adj1[i] += adjustmentList[k].AddAndNormBeta1Adj1[i];
                            adjustmentList[0].AddAndNormGamma1Adj1[i] += adjustmentList[k].AddAndNormGamma1Adj1[i];
                        }
                        adjustmentList[0].AddAndNormBeta2Adj[i] /= miniBatchSize;
                        adjustmentList[0].AddAndNormGamma2Adj[i] /= miniBatchSize;
                        adjustmentList[0].AddAndNormBeta1Adj[i] /= miniBatchSize;
                        adjustmentList[0].AddAndNormGamma1Adj[i] /= miniBatchSize;
                        adjustmentList[0].AddAndNormBeta2Adj1[i] /= miniBatchSize;
                        adjustmentList[0].AddAndNormGamma2Adj1[i] /= miniBatchSize;
                        adjustmentList[0].AddAndNormBeta1Adj1[i] /= miniBatchSize;
                        adjustmentList[0].AddAndNormGamma1Adj1[i] /= miniBatchSize;
                        adjustmentList[0].AddAndNormBeta2Adj[i] = (adjustmentList[0].AddAndNormBeta2Adj[i] + adjustmentList[0].AddAndNormBeta2Adj1[i]) / 2;
                        adjustmentList[0].AddAndNormGamma2Adj[i] = (adjustmentList[0].AddAndNormGamma2Adj[i] + adjustmentList[0].AddAndNormGamma2Adj1[i]) / 2;
                        adjustmentList[0].AddAndNormBeta1Adj[i] = (adjustmentList[0].AddAndNormBeta1Adj[i] + adjustmentList[0].AddAndNormBeta1Adj1[i]) / 2;
                        adjustmentList[0].AddAndNormGamma1Adj[i] = (adjustmentList[0].AddAndNormGamma1Adj[i] + adjustmentList[0].AddAndNormGamma1Adj1[i]) / 2;

                        adjustmentList[0].affineMLPSecondLayerBiasBlock2Adj[i] /= miniBatchSize;
                        adjustmentList[0].affineMLPSecondLayerBiasBlock1Adj[i] /= miniBatchSize;
                        adjustmentList[0].affineMLPSecondLayerBiasBlock2Adj[i] = (adjustmentList[0].affineMLPSecondLayerBiasBlock2Adj[i] + adjustmentList[0].affineMLPSecondLayerBiasBlock1Adj[i]) / 2;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].AddAndNormBeta2Adj, "affineMLPBeta2", 2);
                    funcs.rectified_adam_optimizer(adjustmentList[0].AddAndNormGamma2Adj, "affineMLPGamma2", 2);
                    funcs.rectified_adam_optimizer(adjustmentList[0].affineMLPSecondLayerBiasBlock2Adj, "affineMLPBias2", 2);
                    funcs.rectified_adam_optimizer(adjustmentList[0].AddAndNormBeta1Adj, "affineMLPBeta1", 2);
                    funcs.rectified_adam_optimizer(adjustmentList[0].AddAndNormGamma1Adj, "affineMLPGamma1", 2);
                    for (int i = 0; i < 1500; i++)
                    {
                        networkArray[0].transStructs[0].addAndNorm2Beta[i] -= backProp.affineMLPBeta2_adapted_rate[i];
                        networkArray[0].transStructs[0].addAndNorm2Gamma[i] -= backProp.affineMLPGamma2_adapted_rate[i];
                        networkArray[0].transStructs[0].addAndNorm1Beta[i] -= backProp.affineMLPBeta1_adapted_rate[i];
                        networkArray[0].transStructs[0].addAndNorm1Gamma[i] -= backProp.affineMLPGamma1_adapted_rate[i];

                        networkArray[0].transStructs[0].transMLPSecondLayerBias[i] -= backProp.affineMLPBias2_adapted_rate[i];
                    }
                    for (int i = 0; i < 225; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].finalLinearLayerWeightsAdj[i] += adjustmentList[k].finalLinearLayerWeightsAdj[i];
                            adjustmentList[0].finalLinearLayerWeightsAdj1[i] += adjustmentList[k].finalLinearLayerWeightsAdj1[i];
                        }
                        adjustmentList[0].finalLinearLayerWeightsAdj[i] /= miniBatchSize;
                        adjustmentList[0].finalLinearLayerWeightsAdj1[i] /= miniBatchSize;
                        adjustmentList[0].finalLinearLayerWeightsAdj[i] = (adjustmentList[0].finalLinearLayerWeightsAdj[i] + adjustmentList[0].finalLinearLayerWeightsAdj1[i]) / 2;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].finalLinearLayerWeightsAdj, "finalLinearLayerWeights", 2);
                    for(int i = 0; i < 225; i++)
                    {
                        networkArray[0].transStructs[0].finalLinearLayerWeights[i] -= backProp.finalLinearLayerBlock2_adapted_rate[i];
                    }
                    for (int i = 0; i < 75; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            //second block adjustments
                            adjustmentList[0].queryHead1LinearLayerWeightsAdj[i] += adjustmentList[k].queryHead1LinearLayerWeightsAdj[i];
                            adjustmentList[0].keyHead1LinearLayerWeightsAdj[i] += adjustmentList[k].keyHead1LinearLayerWeightsAdj[i];
                            adjustmentList[0].valueHead1LinearLayerWeightsAdj[i] += adjustmentList[k].valueHead1LinearLayerWeightsAdj[i];

                            adjustmentList[0].queryHead2LinearLayerWeightsAdj[i] += adjustmentList[k].queryHead2LinearLayerWeightsAdj[i];
                            adjustmentList[0].keyHead2LinearLayerWeightsAdj[i] += adjustmentList[k].keyHead2LinearLayerWeightsAdj[i];
                            adjustmentList[0].valueHead2LinearLayerWeightsAdj[i] += adjustmentList[k].valueHead2LinearLayerWeightsAdj[i];

                            adjustmentList[0].queryHead3LinearLayerWeightsAdj[i] += adjustmentList[k].queryHead3LinearLayerWeightsAdj[i];
                            adjustmentList[0].keyHead3LinearLayerWeightsAdj[i] += adjustmentList[k].keyHead3LinearLayerWeightsAdj[i];
                            adjustmentList[0].valueHead3LinearLayerWeightsAdj[i] += adjustmentList[k].valueHead3LinearLayerWeightsAdj[i];

                            //first block adjustments
                            adjustmentList[0].queryHead1LinearLayerWeightsAdj1[i] += adjustmentList[k].queryHead1LinearLayerWeightsAdj1[i];
                            adjustmentList[0].keyHead1LinearLayerWeightsAdj1[i] += adjustmentList[k].keyHead1LinearLayerWeightsAdj1[i];
                            adjustmentList[0].valueHead1LinearLayerWeightsAdj1[i] += adjustmentList[k].valueHead1LinearLayerWeightsAdj1[i];

                            adjustmentList[0].queryHead2LinearLayerWeightsAdj1[i] += adjustmentList[k].queryHead2LinearLayerWeightsAdj1[i];
                            adjustmentList[0].keyHead2LinearLayerWeightsAdj1[i] += adjustmentList[k].keyHead2LinearLayerWeightsAdj1[i];
                            adjustmentList[0].valueHead2LinearLayerWeightsAdj1[i] += adjustmentList[k].valueHead2LinearLayerWeightsAdj1[i];

                            adjustmentList[0].queryHead3LinearLayerWeightsAdj1[i] += adjustmentList[k].queryHead3LinearLayerWeightsAdj1[i];
                            adjustmentList[0].keyHead3LinearLayerWeightsAdj1[i] += adjustmentList[k].keyHead3LinearLayerWeightsAdj1[i];
                            adjustmentList[0].valueHead3LinearLayerWeightsAdj1[i] += adjustmentList[k].valueHead3LinearLayerWeightsAdj1[i];
                        }
                        adjustmentList[0].queryHead1LinearLayerWeightsAdj[i] /= miniBatchSize;
                        adjustmentList[0].keyHead1LinearLayerWeightsAdj[i] /= miniBatchSize;
                        adjustmentList[0].valueHead1LinearLayerWeightsAdj[i] /= miniBatchSize;

                        adjustmentList[0].queryHead2LinearLayerWeightsAdj[i] /= miniBatchSize;
                        adjustmentList[0].keyHead2LinearLayerWeightsAdj[i] /= miniBatchSize;
                        adjustmentList[0].valueHead2LinearLayerWeightsAdj[i] /= miniBatchSize;

                        adjustmentList[0].queryHead3LinearLayerWeightsAdj[i] /= miniBatchSize;
                        adjustmentList[0].keyHead3LinearLayerWeightsAdj[i] /= miniBatchSize;
                        adjustmentList[0].valueHead3LinearLayerWeightsAdj[i] /= miniBatchSize;

                        adjustmentList[0].queryHead1LinearLayerWeightsAdj1[i] /= miniBatchSize;
                        adjustmentList[0].keyHead1LinearLayerWeightsAdj1[i] /= miniBatchSize;
                        adjustmentList[0].valueHead1LinearLayerWeightsAdj1[i] /= miniBatchSize;

                        adjustmentList[0].queryHead2LinearLayerWeightsAdj1[i] /= miniBatchSize;
                        adjustmentList[0].keyHead2LinearLayerWeightsAdj1[i] /= miniBatchSize;
                        adjustmentList[0].valueHead2LinearLayerWeightsAdj1[i] /= miniBatchSize;

                        adjustmentList[0].queryHead3LinearLayerWeightsAdj1[i] /= miniBatchSize;
                        adjustmentList[0].keyHead3LinearLayerWeightsAdj1[i] /= miniBatchSize;
                        adjustmentList[0].valueHead3LinearLayerWeightsAdj1[i] /= miniBatchSize;

                        adjustmentList[0].queryHead1LinearLayerWeightsAdj[i] = (adjustmentList[0].queryHead1LinearLayerWeightsAdj[i] + adjustmentList[0].queryHead1LinearLayerWeightsAdj1[i]) / 2;
                        adjustmentList[0].keyHead1LinearLayerWeightsAdj[i] = (adjustmentList[0].keyHead1LinearLayerWeightsAdj[i] + adjustmentList[0].keyHead1LinearLayerWeightsAdj1[i]) / 2;
                        adjustmentList[0].valueHead1LinearLayerWeightsAdj[i] = (adjustmentList[0].valueHead1LinearLayerWeightsAdj[i] + adjustmentList[0].valueHead1LinearLayerWeightsAdj1[i]) / 2;

                        adjustmentList[0].queryHead2LinearLayerWeightsAdj[i] = (adjustmentList[0].queryHead2LinearLayerWeightsAdj[i] + adjustmentList[0].queryHead2LinearLayerWeightsAdj1[i]) / 2;
                        adjustmentList[0].keyHead2LinearLayerWeightsAdj[i] = (adjustmentList[0].keyHead2LinearLayerWeightsAdj[i] + adjustmentList[0].keyHead2LinearLayerWeightsAdj1[i]) / 2;
                        adjustmentList[0].valueHead2LinearLayerWeightsAdj[i] = (adjustmentList[0].valueHead2LinearLayerWeightsAdj[i] + adjustmentList[0].valueHead2LinearLayerWeightsAdj1[i]) / 2;

                        adjustmentList[0].queryHead3LinearLayerWeightsAdj[i] = (adjustmentList[0].queryHead3LinearLayerWeightsAdj[i] + adjustmentList[0].queryHead3LinearLayerWeightsAdj1[i]) / 2;
                        adjustmentList[0].keyHead3LinearLayerWeightsAdj[i] = (adjustmentList[0].keyHead3LinearLayerWeightsAdj[i] + adjustmentList[0].keyHead3LinearLayerWeightsAdj1[i]) / 2;
                        adjustmentList[0].valueHead3LinearLayerWeightsAdj[i] = (adjustmentList[0].valueHead3LinearLayerWeightsAdj[i] + adjustmentList[0].valueHead3LinearLayerWeightsAdj1[i]) / 2;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].queryHead1LinearLayerWeightsAdj, "queryHead1Weights", 2);
                    funcs.rectified_adam_optimizer(adjustmentList[0].keyHead1LinearLayerWeightsAdj, "keyHead1Weights", 2);
                    funcs.rectified_adam_optimizer(adjustmentList[0].valueHead1LinearLayerWeightsAdj, "valueHead1Weights", 2);

                    funcs.rectified_adam_optimizer(adjustmentList[0].queryHead2LinearLayerWeightsAdj, "queryHead2Weights", 2);
                    funcs.rectified_adam_optimizer(adjustmentList[0].keyHead2LinearLayerWeightsAdj, "keyHead2Weights", 2);
                    funcs.rectified_adam_optimizer(adjustmentList[0].valueHead2LinearLayerWeightsAdj, "valueHead2Weights", 2);

                    funcs.rectified_adam_optimizer(adjustmentList[0].queryHead3LinearLayerWeightsAdj, "queryHead3Weights", 2);
                    funcs.rectified_adam_optimizer(adjustmentList[0].keyHead3LinearLayerWeightsAdj, "keyHead3Weights", 2);
                    funcs.rectified_adam_optimizer(adjustmentList[0].valueHead3LinearLayerWeightsAdj, "valueHead3Weights", 2);
                    if(predictorGui1.enableOutputs.Checked == true)
                    {
                        StreamWriter output = File.AppendText(@"X:\queryHead1Block2_adapted_rate.txt");
                        for(int i = 0; i < 75; i++)
                        {
                            output.WriteLine(backProp.queryHead1Block2_adapted_rate[i].ToString());
                        }
                        output.Close();
                    }
                    for (int i = 0; i < 75; i++)
                    {
                        networkArray[0].transStructs[0].queryLinearLayerWeights_head1[i] -= backProp.queryHead1Block2_adapted_rate[i];
                        networkArray[0].transStructs[0].keyLinearLayerWeights_head1[i] -= backProp.keyHead1Block2_adapted_rate[i];
                        networkArray[0].transStructs[0].valueLinearLayerWeights_head1[i] -= backProp.valueHead1Block2_adapted_rate[i];

                        networkArray[0].transStructs[0].queryLinearLayerWeights_head2[i] -= backProp.queryHead2Block2_adapted_rate[i];
                        networkArray[0].transStructs[0].keyLinearLayerWeights_head2[i] -= backProp.keyHead2Block2_adapted_rate[i];
                        networkArray[0].transStructs[0].valueLinearLayerWeights_head2[i] -= backProp.valueHead2Block2_adapted_rate[i];

                        networkArray[0].transStructs[0].queryLinearLayerWeights_head3[i] -= backProp.queryHead3Block2_adapted_rate[i];
                        networkArray[0].transStructs[0].keyLinearLayerWeights_head3[i] -= backProp.keyHead3Block2_adapted_rate[i];
                        networkArray[0].transStructs[0].valueLinearLayerWeights_head3[i] -= backProp.valueHead3Block2_adapted_rate[i];
                    }
                    for (int i = 0; i < 1400; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].convLayer5BiasesAdj[i] += adjustmentList[k].convLayer5BiasesAdj[i];
                            adjustmentList[0].convLayer5PReLUParamAdj[i] += adjustmentList[k].convLayer5PReLUParamAdj[i];
                            adjustmentList[0].convLayer5NormGammaAdj[i] += adjustmentList[k].convLayer5NormGammaAdj[i];
                            adjustmentList[0].convLayer5NormBetaAdj[i] += adjustmentList[k].convLayer5NormBetaAdj[i];
                        }
                        adjustmentList[0].convLayer5BiasesAdj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5PReLUParamAdj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5NormGammaAdj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5NormBetaAdj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5BiasesAdj, "convLayer5Bias", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5NormGammaAdj, "convLayer5Gamma", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5NormBetaAdj, "convLayer5Beta", 0);
                    for (int i = 0; i < 1400; i++)
                    {
                        predictorGui.networkArray[0].convStructs[0].convLayer5Bias[i] -= backProp.convLayer5Bias_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5PReLUParam[i] -= backProp.convLayer5Bias_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormGamma[i] -= backProp.convLayer5Gamma_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormBeta[i] -= backProp.convLayer5Beta_adapted_rate[i];
                    }
                    for (int i = 0; i < 14; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].convLayer5Kernel1Depth1Adj[i] += adjustmentList[k].convLayer5Kernel1Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel1Depth2Adj[i] += adjustmentList[k].convLayer5Kernel1Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel2Depth1Adj[i] += adjustmentList[k].convLayer5Kernel2Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel2Depth2Adj[i] += adjustmentList[k].convLayer5Kernel2Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel3Depth1Adj[i] += adjustmentList[k].convLayer5Kernel3Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel3Depth2Adj[i] += adjustmentList[k].convLayer5Kernel3Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel4Depth1Adj[i] += adjustmentList[k].convLayer5Kernel4Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel4Depth2Adj[i] += adjustmentList[k].convLayer5Kernel4Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel5Depth1Adj[i] += adjustmentList[k].convLayer5Kernel5Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel5Depth2Adj[i] += adjustmentList[k].convLayer5Kernel5Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel6Depth1Adj[i] += adjustmentList[k].convLayer5Kernel6Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel6Depth2Adj[i] += adjustmentList[k].convLayer5Kernel6Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel7Depth1Adj[i] += adjustmentList[k].convLayer5Kernel7Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel7Depth2Adj[i] += adjustmentList[k].convLayer5Kernel7Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel8Depth1Adj[i] += adjustmentList[k].convLayer5Kernel8Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel8Depth2Adj[i] += adjustmentList[k].convLayer5Kernel8Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel9Depth1Adj[i] += adjustmentList[k].convLayer5Kernel9Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel9Depth2Adj[i] += adjustmentList[k].convLayer5Kernel9Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel10Depth1Adj[i] += adjustmentList[k].convLayer5Kernel10Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel10Depth2Adj[i] += adjustmentList[k].convLayer5Kernel10Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel11Depth1Adj[i] += adjustmentList[k].convLayer5Kernel11Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel11Depth2Adj[i] += adjustmentList[k].convLayer5Kernel11Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel12Depth1Adj[i] += adjustmentList[k].convLayer5Kernel12Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel12Depth2Adj[i] += adjustmentList[k].convLayer5Kernel12Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel13Depth1Adj[i] += adjustmentList[k].convLayer5Kernel13Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel13Depth2Adj[i] += adjustmentList[k].convLayer5Kernel13Depth2Adj[i];
                            adjustmentList[0].convLayer5Kernel14Depth1Adj[i] += adjustmentList[k].convLayer5Kernel14Depth1Adj[i];
                            adjustmentList[0].convLayer5Kernel14Depth2Adj[i] += adjustmentList[k].convLayer5Kernel14Depth2Adj[i];
                        }
                        adjustmentList[0].convLayer5Kernel1Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel1Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel2Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel2Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel3Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel3Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel4Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel4Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel5Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel5Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel6Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel6Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel7Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel7Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel8Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel8Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel9Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel9Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel10Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel10Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel11Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel11Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel12Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel12Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel13Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel13Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel14Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer5Kernel14Depth2Adj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel1Depth1Adj, "convLayer5WeightsKernel1Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel1Depth2Adj, "convLayer5WeightsKernel1Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel2Depth1Adj, "convLayer5WeightsKernel2Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel2Depth2Adj, "convLayer5WeightsKernel2Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel3Depth1Adj, "convLayer5WeightsKernel3Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel3Depth2Adj, "convLayer5WeightsKernel3Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel4Depth1Adj, "convLayer5WeightsKernel4Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel4Depth2Adj, "convLayer5WeightsKernel4Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel5Depth1Adj, "convLayer5WeightsKernel5Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel5Depth2Adj, "convLayer5WeightsKernel5Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel6Depth1Adj, "convLayer5WeightsKernel6Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel6Depth2Adj, "convLayer5WeightsKernel6Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel7Depth1Adj, "convLayer5WeightsKernel7Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel7Depth2Adj, "convLayer5WeightsKernel7Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel8Depth1Adj, "convLayer5WeightsKernel8Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel8Depth2Adj, "convLayer5WeightsKernel8Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel9Depth1Adj, "convLayer5WeightsKernel9Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel9Depth2Adj, "convLayer5WeightsKernel9Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel10Depth1Adj, "convLayer5WeightsKernel10Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel10Depth2Adj, "convLayer5WeightsKernel10Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel11Depth1Adj, "convLayer5WeightsKernel11Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel11Depth2Adj, "convLayer5WeightsKernel11Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel12Depth1Adj, "convLayer5WeightsKernel12Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel12Depth2Adj, "convLayer5WeightsKernel12Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel13Depth1Adj, "convLayer5WeightsKernel13Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel13Depth2Adj, "convLayer5WeightsKernel13Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel14Depth1Adj, "convLayer5WeightsKernel14Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer5Kernel14Depth2Adj, "convLayer5WeightsKernel14Depth2", 0);
                    for (int i = 0; i < 14; i++)
                    {
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[0].depth1[i] -= backProp.convLayer5Kernel1Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[0].depth2[i] -= backProp.convLayer5Kernel1Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[1].depth1[i] -= backProp.convLayer5Kernel2Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[1].depth2[i] -= backProp.convLayer5Kernel2Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[2].depth1[i] -= backProp.convLayer5Kernel3Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[2].depth2[i] -= backProp.convLayer5Kernel3Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[3].depth1[i] -= backProp.convLayer5Kernel4Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[3].depth2[i] -= backProp.convLayer5Kernel4Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[4].depth1[i] -= backProp.convLayer5Kernel5Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[4].depth2[i] -= backProp.convLayer5Kernel5Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[5].depth1[i] -= backProp.convLayer5Kernel6Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[5].depth2[i] -= backProp.convLayer5Kernel6Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[6].depth1[i] -= backProp.convLayer5Kernel7Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[6].depth2[i] -= backProp.convLayer5Kernel7Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[7].depth1[i] -= backProp.convLayer5Kernel8Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[7].depth2[i] -= backProp.convLayer5Kernel8Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[8].depth1[i] -= backProp.convLayer5Kernel9Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[8].depth2[i] -= backProp.convLayer5Kernel9Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[9].depth1[i] -= backProp.convLayer5Kernel10Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[9].depth2[i] -= backProp.convLayer5Kernel10Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[10].depth1[i] -= backProp.convLayer5Kernel11Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[10].depth2[i] -= backProp.convLayer5Kernel11Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[11].depth1[i] -= backProp.convLayer5Kernel12Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[11].depth2[i] -= backProp.convLayer5Kernel12Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[12].depth1[i] -= backProp.convLayer5Kernel13Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[12].depth2[i] -= backProp.convLayer5Kernel13Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[13].depth1[i] -= backProp.convLayer5Kernel14Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[13].depth2[i] -= backProp.convLayer5Kernel14Depth2_adapted_rate[i];
                    }
                    for (int i = 0; i < 1400; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].convLayer4BiasesAdj[i] += adjustmentList[k].convLayer4BiasesAdj[i];
                            adjustmentList[0].convLayer4PReLUParamAdj[i] += adjustmentList[k].convLayer4PReLUParamAdj[i];
                        }
                        adjustmentList[0].convLayer4BiasesAdj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4PReLUParamAdj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4BiasesAdj, "convLayer4BiasPrelu", 0);
                    for (int i = 0; i < 1400; i++)
                    {
                        predictorGui.networkArray[0].convStructs[0].convLayer4Bias[i] -= backProp.convLayer4Bias_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4PReLUParam[i] -= backProp.convLayer4Bias_adapted_rate[i];
                    }
                    for (int i = 0; i < 14; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].convLayer4Kernel1Depth1Adj[i] += adjustmentList[k].convLayer4Kernel1Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel1Depth2Adj[i] += adjustmentList[k].convLayer4Kernel1Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel2Depth1Adj[i] += adjustmentList[k].convLayer4Kernel2Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel2Depth2Adj[i] += adjustmentList[k].convLayer4Kernel2Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel3Depth1Adj[i] += adjustmentList[k].convLayer4Kernel3Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel3Depth2Adj[i] += adjustmentList[k].convLayer4Kernel3Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel4Depth1Adj[i] += adjustmentList[k].convLayer4Kernel4Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel4Depth2Adj[i] += adjustmentList[k].convLayer4Kernel4Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel5Depth1Adj[i] += adjustmentList[k].convLayer4Kernel5Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel5Depth2Adj[i] += adjustmentList[k].convLayer4Kernel5Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel6Depth1Adj[i] += adjustmentList[k].convLayer4Kernel6Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel6Depth2Adj[i] += adjustmentList[k].convLayer4Kernel6Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel7Depth1Adj[i] += adjustmentList[k].convLayer4Kernel7Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel7Depth2Adj[i] += adjustmentList[k].convLayer4Kernel7Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel8Depth1Adj[i] += adjustmentList[k].convLayer4Kernel8Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel8Depth2Adj[i] += adjustmentList[k].convLayer4Kernel8Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel9Depth1Adj[i] += adjustmentList[k].convLayer4Kernel9Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel9Depth2Adj[i] += adjustmentList[k].convLayer4Kernel9Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel10Depth1Adj[i] += adjustmentList[k].convLayer4Kernel10Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel10Depth2Adj[i] += adjustmentList[k].convLayer4Kernel10Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel11Depth1Adj[i] += adjustmentList[k].convLayer4Kernel11Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel11Depth2Adj[i] += adjustmentList[k].convLayer4Kernel11Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel12Depth1Adj[i] += adjustmentList[k].convLayer4Kernel12Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel12Depth2Adj[i] += adjustmentList[k].convLayer4Kernel12Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel13Depth1Adj[i] += adjustmentList[k].convLayer4Kernel13Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel13Depth2Adj[i] += adjustmentList[k].convLayer4Kernel13Depth2Adj[i];
                            adjustmentList[0].convLayer4Kernel14Depth1Adj[i] += adjustmentList[k].convLayer4Kernel14Depth1Adj[i];
                            adjustmentList[0].convLayer4Kernel14Depth2Adj[i] += adjustmentList[k].convLayer4Kernel14Depth2Adj[i];
                        }
                        adjustmentList[0].convLayer4Kernel1Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel1Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel2Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel2Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel3Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel3Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel4Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel4Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel5Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel5Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel6Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel6Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel7Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel7Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel8Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel8Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel9Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel9Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel10Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel10Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel11Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel11Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel12Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel12Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel13Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel13Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel14Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer4Kernel14Depth2Adj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel1Depth1Adj, "convLayer4WeightsKernel1Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel1Depth2Adj, "convLayer4WeightsKernel1Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel2Depth1Adj, "convLayer4WeightsKernel2Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel2Depth2Adj, "convLayer4WeightsKernel2Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel3Depth1Adj, "convLayer4WeightsKernel3Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel3Depth2Adj, "convLayer4WeightsKernel3Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel4Depth1Adj, "convLayer4WeightsKernel4Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel4Depth2Adj, "convLayer4WeightsKernel4Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel5Depth1Adj, "convLayer4WeightsKernel5Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel5Depth2Adj, "convLayer4WeightsKernel5Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel6Depth1Adj, "convLayer4WeightsKernel6Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel6Depth2Adj, "convLayer4WeightsKernel6Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel7Depth1Adj, "convLayer4WeightsKernel7Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel7Depth2Adj, "convLayer4WeightsKernel7Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel8Depth1Adj, "convLayer4WeightsKernel8Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel8Depth2Adj, "convLayer4WeightsKernel8Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel9Depth1Adj, "convLayer4WeightsKernel9Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel9Depth2Adj, "convLayer4WeightsKernel9Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel10Depth1Adj, "convLayer4WeightsKernel10Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel10Depth2Adj, "convLayer4WeightsKernel10Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel11Depth1Adj, "convLayer4WeightsKernel11Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel11Depth2Adj, "convLayer4WeightsKernel11Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel12Depth1Adj, "convLayer4WeightsKernel12Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel12Depth2Adj, "convLayer4WeightsKernel12Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel13Depth1Adj, "convLayer4WeightsKernel13Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel13Depth2Adj, "convLayer4WeightsKernel13Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel14Depth1Adj, "convLayer4WeightsKernel14Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer4Kernel14Depth2Adj, "convLayer4WeightsKernel14Depth2", 0);
                    for (int i = 0; i < 14; i++)
                    {
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[0].depth1[i] -= backProp.convLayer4Kernel1Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[0].depth2[i] -= backProp.convLayer4Kernel1Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[1].depth1[i] -= backProp.convLayer4Kernel2Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[1].depth2[i] -= backProp.convLayer4Kernel2Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[2].depth1[i] -= backProp.convLayer4Kernel3Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[2].depth2[i] -= backProp.convLayer4Kernel3Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[3].depth1[i] -= backProp.convLayer4Kernel4Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[3].depth2[i] -= backProp.convLayer4Kernel4Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[4].depth1[i] -= backProp.convLayer4Kernel5Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[4].depth2[i] -= backProp.convLayer4Kernel5Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[5].depth1[i] -= backProp.convLayer4Kernel6Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[5].depth2[i] -= backProp.convLayer4Kernel6Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[6].depth1[i] -= backProp.convLayer4Kernel7Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[6].depth2[i] -= backProp.convLayer4Kernel7Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[7].depth1[i] -= backProp.convLayer4Kernel8Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[7].depth2[i] -= backProp.convLayer4Kernel8Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[8].depth1[i] -= backProp.convLayer4Kernel9Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[8].depth2[i] -= backProp.convLayer4Kernel9Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[9].depth1[i] -= backProp.convLayer4Kernel10Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[9].depth2[i] -= backProp.convLayer4Kernel10Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[10].depth1[i] -= backProp.convLayer4Kernel11Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[10].depth2[i] -= backProp.convLayer4Kernel11Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[11].depth1[i] -= backProp.convLayer4Kernel12Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[11].depth2[i] -= backProp.convLayer4Kernel12Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[12].depth1[i] -= backProp.convLayer4Kernel13Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[12].depth2[i] -= backProp.convLayer4Kernel13Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[13].depth1[i] -= backProp.convLayer4Kernel14Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[13].depth2[i] -= backProp.convLayer4Kernel14Depth2_adapted_rate[i];
                    }
                    for (int i = 0; i < 1400; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].convLayer3BiasesAdj[i] += adjustmentList[k].convLayer3BiasesAdj[i];
                            adjustmentList[0].convLayer3PReLUParamAdj[i] += adjustmentList[k].convLayer3PReLUParamAdj[i];
                        }
                        adjustmentList[0].convLayer3BiasesAdj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3PReLUParamAdj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3BiasesAdj, "convLayer3BiasPrelu", 0);
                    for (int i = 0; i < 1400; i++)
                    {
                        predictorGui.networkArray[0].convStructs[0].convLayer3Bias[i] -= backProp.convLayer3Bias_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3PReLUParam[i] -= backProp.convLayer3Bias_adapted_rate[i];
                    }
                    for (int i = 0; i < 14; i++)
                    {
                        for (int k = 1; k < 32; k++)
                        {
                            adjustmentList[0].convLayer3Kernel1Depth1Adj[i] += adjustmentList[k].convLayer3Kernel1Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel1Depth2Adj[i] += adjustmentList[k].convLayer3Kernel1Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel2Depth1Adj[i] += adjustmentList[k].convLayer3Kernel2Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel2Depth2Adj[i] += adjustmentList[k].convLayer3Kernel2Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel3Depth1Adj[i] += adjustmentList[k].convLayer3Kernel3Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel3Depth2Adj[i] += adjustmentList[k].convLayer3Kernel3Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel4Depth1Adj[i] += adjustmentList[k].convLayer3Kernel4Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel4Depth2Adj[i] += adjustmentList[k].convLayer3Kernel4Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel5Depth1Adj[i] += adjustmentList[k].convLayer3Kernel5Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel5Depth2Adj[i] += adjustmentList[k].convLayer3Kernel5Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel6Depth1Adj[i] += adjustmentList[k].convLayer3Kernel6Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel6Depth2Adj[i] += adjustmentList[k].convLayer3Kernel6Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel7Depth1Adj[i] += adjustmentList[k].convLayer3Kernel7Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel7Depth2Adj[i] += adjustmentList[k].convLayer3Kernel7Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel8Depth1Adj[i] += adjustmentList[k].convLayer3Kernel8Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel8Depth2Adj[i] += adjustmentList[k].convLayer3Kernel8Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel9Depth1Adj[i] += adjustmentList[k].convLayer3Kernel9Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel9Depth2Adj[i] += adjustmentList[k].convLayer3Kernel9Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel10Depth1Adj[i] += adjustmentList[k].convLayer3Kernel10Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel10Depth2Adj[i] += adjustmentList[k].convLayer3Kernel10Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel11Depth1Adj[i] += adjustmentList[k].convLayer3Kernel11Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel11Depth2Adj[i] += adjustmentList[k].convLayer3Kernel11Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel12Depth1Adj[i] += adjustmentList[k].convLayer3Kernel12Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel12Depth2Adj[i] += adjustmentList[k].convLayer3Kernel12Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel13Depth1Adj[i] += adjustmentList[k].convLayer3Kernel13Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel13Depth2Adj[i] += adjustmentList[k].convLayer3Kernel13Depth2Adj[i];
                            adjustmentList[0].convLayer3Kernel14Depth1Adj[i] += adjustmentList[k].convLayer3Kernel14Depth1Adj[i];
                            adjustmentList[0].convLayer3Kernel14Depth2Adj[i] += adjustmentList[k].convLayer3Kernel14Depth2Adj[i];
                        }
                        adjustmentList[0].convLayer3Kernel1Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel1Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel2Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel2Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel3Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel3Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel4Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel4Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel5Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel5Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel6Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel6Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel7Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel7Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel8Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel8Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel9Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel9Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel10Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel10Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel11Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel11Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel12Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel12Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel13Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel13Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel14Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer3Kernel14Depth2Adj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel1Depth1Adj, "convLayer3WeightsKernel1Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel1Depth2Adj, "convLayer3WeightsKernel1Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel2Depth1Adj, "convLayer3WeightsKernel2Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel2Depth2Adj, "convLayer3WeightsKernel2Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel3Depth1Adj, "convLayer3WeightsKernel3Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel3Depth2Adj, "convLayer3WeightsKernel3Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel4Depth1Adj, "convLayer3WeightsKernel4Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel4Depth2Adj, "convLayer3WeightsKernel4Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel5Depth1Adj, "convLayer3WeightsKernel5Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel5Depth2Adj, "convLayer3WeightsKernel5Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel6Depth1Adj, "convLayer3WeightsKernel6Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel6Depth2Adj, "convLayer3WeightsKernel6Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel7Depth1Adj, "convLayer3WeightsKernel7Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel7Depth2Adj, "convLayer3WeightsKernel7Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel8Depth1Adj, "convLayer3WeightsKernel8Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel8Depth2Adj, "convLayer3WeightsKernel8Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel9Depth1Adj, "convLayer3WeightsKernel9Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel9Depth2Adj, "convLayer3WeightsKernel9Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel10Depth1Adj, "convLayer3WeightsKernel10Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel10Depth2Adj, "convLayer3WeightsKernel10Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel11Depth1Adj, "convLayer3WeightsKernel11Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel11Depth2Adj, "convLayer3WeightsKernel11Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel12Depth1Adj, "convLayer3WeightsKernel12Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel12Depth2Adj, "convLayer3WeightsKernel12Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel13Depth1Adj, "convLayer3WeightsKernel13Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel13Depth2Adj, "convLayer3WeightsKernel13Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel14Depth1Adj, "convLayer3WeightsKernel14Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer3Kernel14Depth2Adj, "convLayer3WeightsKernel14Depth2", 0);
                    for (int i = 0; i < 14; i++)
                    {
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[0].depth1[i] -= backProp.convLayer3Kernel1Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[0].depth2[i] -= backProp.convLayer3Kernel1Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[1].depth1[i] -= backProp.convLayer3Kernel2Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[1].depth2[i] -= backProp.convLayer3Kernel2Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[2].depth1[i] -= backProp.convLayer3Kernel3Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[2].depth2[i] -= backProp.convLayer3Kernel3Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[3].depth1[i] -= backProp.convLayer3Kernel4Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[3].depth2[i] -= backProp.convLayer3Kernel4Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[4].depth1[i] -= backProp.convLayer3Kernel5Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[4].depth2[i] -= backProp.convLayer3Kernel5Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[5].depth1[i] -= backProp.convLayer3Kernel6Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[5].depth2[i] -= backProp.convLayer3Kernel6Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[6].depth1[i] -= backProp.convLayer3Kernel7Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[6].depth2[i] -= backProp.convLayer3Kernel7Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[7].depth1[i] -= backProp.convLayer3Kernel8Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[7].depth2[i] -= backProp.convLayer3Kernel8Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[8].depth1[i] -= backProp.convLayer3Kernel9Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[8].depth2[i] -= backProp.convLayer3Kernel9Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[9].depth2[i] -= backProp.convLayer3Kernel10Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[10].depth1[i] -= backProp.convLayer3Kernel11Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[10].depth2[i] -= backProp.convLayer3Kernel11Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[11].depth1[i] -= backProp.convLayer3Kernel12Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[11].depth2[i] -= backProp.convLayer3Kernel12Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[12].depth1[i] -= backProp.convLayer3Kernel13Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[12].depth2[i] -= backProp.convLayer3Kernel13Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[13].depth1[i] -= backProp.convLayer3Kernel14Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[13].depth2[i] -= backProp.convLayer3Kernel14Depth2_adapted_rate[i];
                    }
                    for (int i = 0; i < 1400; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].convLayer2BiasesAdj[i] += adjustmentList[k].convLayer2BiasesAdj[i];
                            adjustmentList[0].convLayer2PReLUParamAdj[i] += adjustmentList[k].convLayer2PReLUParamAdj[i];
                        }
                        adjustmentList[0].convLayer2BiasesAdj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2PReLUParamAdj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2BiasesAdj, "convLayer2BiasPrelu", 0);
                    for (int i = 0; i < 1400; i++)
                    {
                        predictorGui.networkArray[0].convStructs[0].convLayer2Bias[i] -= backProp.convLayer2Bias_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2PReLUParam[i] -= backProp.convLayer2Bias_adapted_rate[i];
                    }
                    for (int i = 0; i < 14; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].convLayer2Kernel1Depth1Adj[i] += adjustmentList[k].convLayer2Kernel1Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel1Depth2Adj[i] += adjustmentList[k].convLayer2Kernel1Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel2Depth1Adj[i] += adjustmentList[k].convLayer2Kernel2Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel2Depth2Adj[i] += adjustmentList[k].convLayer2Kernel2Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel3Depth1Adj[i] += adjustmentList[k].convLayer2Kernel3Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel3Depth2Adj[i] += adjustmentList[k].convLayer2Kernel3Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel4Depth1Adj[i] += adjustmentList[k].convLayer2Kernel4Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel4Depth2Adj[i] += adjustmentList[k].convLayer2Kernel4Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel5Depth1Adj[i] += adjustmentList[k].convLayer2Kernel5Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel5Depth2Adj[i] += adjustmentList[k].convLayer2Kernel5Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel6Depth1Adj[i] += adjustmentList[k].convLayer2Kernel6Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel6Depth2Adj[i] += adjustmentList[k].convLayer2Kernel6Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel7Depth1Adj[i] += adjustmentList[k].convLayer2Kernel7Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel7Depth2Adj[i] += adjustmentList[k].convLayer2Kernel7Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel8Depth1Adj[i] += adjustmentList[k].convLayer2Kernel8Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel8Depth2Adj[i] += adjustmentList[k].convLayer2Kernel8Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel9Depth1Adj[i] += adjustmentList[k].convLayer2Kernel9Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel9Depth2Adj[i] += adjustmentList[k].convLayer2Kernel9Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel10Depth1Adj[i] += adjustmentList[k].convLayer2Kernel10Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel10Depth2Adj[i] += adjustmentList[k].convLayer2Kernel10Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel11Depth1Adj[i] += adjustmentList[k].convLayer2Kernel11Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel11Depth2Adj[i] += adjustmentList[k].convLayer2Kernel11Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel12Depth1Adj[i] += adjustmentList[k].convLayer2Kernel12Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel12Depth2Adj[i] += adjustmentList[k].convLayer2Kernel12Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel13Depth1Adj[i] += adjustmentList[k].convLayer2Kernel13Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel13Depth2Adj[i] += adjustmentList[k].convLayer2Kernel13Depth2Adj[i];
                            adjustmentList[0].convLayer2Kernel14Depth1Adj[i] += adjustmentList[k].convLayer2Kernel14Depth1Adj[i];
                            adjustmentList[0].convLayer2Kernel14Depth2Adj[i] += adjustmentList[k].convLayer2Kernel14Depth2Adj[i];
                        }
                        adjustmentList[0].convLayer2Kernel1Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel1Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel2Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel2Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel3Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel3Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel4Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel4Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel5Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel5Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel6Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel6Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel7Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel7Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel8Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel8Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel9Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel9Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel10Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel10Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel11Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel11Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel12Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel12Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel13Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel13Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel14Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer2Kernel14Depth2Adj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel1Depth1Adj, "convLayer2WeightsKernel1Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel1Depth2Adj, "convLayer2WeightsKernel1Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel2Depth1Adj, "convLayer2WeightsKernel2Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel2Depth2Adj, "convLayer2WeightsKernel2Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel3Depth1Adj, "convLayer2WeightsKernel3Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel3Depth2Adj, "convLayer2WeightsKernel3Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel4Depth1Adj, "convLayer2WeightsKernel4Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel4Depth2Adj, "convLayer2WeightsKernel4Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel5Depth1Adj, "convLayer2WeightsKernel5Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel5Depth2Adj, "convLayer2WeightsKernel5Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel6Depth1Adj, "convLayer2WeightsKernel6Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel6Depth2Adj, "convLayer2WeightsKernel6Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel7Depth1Adj, "convLayer2WeightsKernel7Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel7Depth2Adj, "convLayer2WeightsKernel7Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel8Depth1Adj, "convLayer2WeightsKernel8Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel8Depth2Adj, "convLayer2WeightsKernel8Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel9Depth1Adj, "convLayer2WeightsKernel9Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel9Depth2Adj, "convLayer2WeightsKernel9Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel10Depth1Adj, "convLayer2WeightsKernel10Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel10Depth2Adj, "convLayer2WeightsKernel10Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel11Depth1Adj, "convLayer2WeightsKernel11Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel11Depth2Adj, "convLayer2WeightsKernel11Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel12Depth1Adj, "convLayer2WeightsKernel12Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel12Depth2Adj, "convLayer2WeightsKernel12Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel13Depth1Adj, "convLayer2WeightsKernel13Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel13Depth2Adj, "convLayer2WeightsKernel13Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel14Depth1Adj, "convLayer2WeightsKernel14Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer2Kernel14Depth2Adj, "convLayer2WeightsKernel14Depth2", 0);
                    for (int i = 0; i < 14; i++)
                    {
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[0].depth1[i] -= backProp.convLayer2Kernel1Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[0].depth2[i] -= backProp.convLayer2Kernel1Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[1].depth1[i] -= backProp.convLayer2Kernel2Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[1].depth2[i] -= backProp.convLayer2Kernel2Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[2].depth1[i] -= backProp.convLayer2Kernel3Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[2].depth2[i] -= backProp.convLayer2Kernel3Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[3].depth1[i] -= backProp.convLayer2Kernel4Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[3].depth2[i] -= backProp.convLayer2Kernel4Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[4].depth1[i] -= backProp.convLayer2Kernel5Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[4].depth2[i] -= backProp.convLayer2Kernel5Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[5].depth1[i] -= backProp.convLayer2Kernel6Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[5].depth2[i] -= backProp.convLayer2Kernel6Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[6].depth1[i] -= backProp.convLayer2Kernel7Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[6].depth2[i] -= backProp.convLayer2Kernel7Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[7].depth1[i] -= backProp.convLayer2Kernel8Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[7].depth2[i] -= backProp.convLayer2Kernel8Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[8].depth1[i] -= backProp.convLayer2Kernel9Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[8].depth2[i] -= backProp.convLayer2Kernel9Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[9].depth1[i] -= backProp.convLayer2Kernel10Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[9].depth2[i] -= backProp.convLayer2Kernel10Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[10].depth1[i] -= backProp.convLayer2Kernel11Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[10].depth2[i] -= backProp.convLayer2Kernel11Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[11].depth1[i] -= backProp.convLayer2Kernel12Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[11].depth2[i] -= backProp.convLayer2Kernel12Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[12].depth1[i] -= backProp.convLayer2Kernel13Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[12].depth2[i] -= backProp.convLayer2Kernel13Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[13].depth1[i] -= backProp.convLayer2Kernel14Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[13].depth2[i] -= backProp.convLayer2Kernel14Depth2_adapted_rate[i];
                    }
                    for (int i = 0; i < 1400; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].convLayer1BiasesAdj[i] += adjustmentList[k].convLayer1BiasesAdj[i];
                            adjustmentList[0].convLayer1PReLUParamAdj[i] += adjustmentList[k].convLayer1PReLUParamAdj[i];
                        }
                        adjustmentList[0].convLayer1BiasesAdj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1PReLUParamAdj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1BiasesAdj, "convLayer1BiasPrelu", 0);
                    for (int i = 0; i < 1400; i++)
                    {
                        predictorGui.networkArray[0].convStructs[0].convLayer1Bias[i] -= backProp.convLayer1Bias_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1PReLUParam[i] -= backProp.convLayer1Bias_adapted_rate[i];
                    }
                    for (int i = 0; i < 32; i++)
                    {
                        for (int k = 1; k < miniBatchSize; k++)
                        {
                            adjustmentList[0].convLayer1Kernel1Depth1Adj[i] += adjustmentList[k].convLayer1Kernel1Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel1Depth2Adj[i] += adjustmentList[k].convLayer1Kernel1Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel1Depth3Adj[i] += adjustmentList[k].convLayer1Kernel1Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel1Depth4Adj[i] += adjustmentList[k].convLayer1Kernel1Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel2Depth1Adj[i] += adjustmentList[k].convLayer1Kernel2Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel2Depth2Adj[i] += adjustmentList[k].convLayer1Kernel2Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel2Depth3Adj[i] += adjustmentList[k].convLayer1Kernel2Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel2Depth4Adj[i] += adjustmentList[k].convLayer1Kernel2Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel3Depth1Adj[i] += adjustmentList[k].convLayer1Kernel3Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel3Depth2Adj[i] += adjustmentList[k].convLayer1Kernel3Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel3Depth3Adj[i] += adjustmentList[k].convLayer1Kernel3Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel3Depth4Adj[i] += adjustmentList[k].convLayer1Kernel3Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel4Depth1Adj[i] += adjustmentList[k].convLayer1Kernel4Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel4Depth2Adj[i] += adjustmentList[k].convLayer1Kernel4Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel4Depth3Adj[i] += adjustmentList[k].convLayer1Kernel4Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel4Depth4Adj[i] += adjustmentList[k].convLayer1Kernel4Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel5Depth1Adj[i] += adjustmentList[k].convLayer1Kernel5Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel5Depth2Adj[i] += adjustmentList[k].convLayer1Kernel5Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel5Depth3Adj[i] += adjustmentList[k].convLayer1Kernel5Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel5Depth4Adj[i] += adjustmentList[k].convLayer1Kernel5Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel6Depth1Adj[i] += adjustmentList[k].convLayer1Kernel6Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel6Depth2Adj[i] += adjustmentList[k].convLayer1Kernel6Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel6Depth3Adj[i] += adjustmentList[k].convLayer1Kernel6Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel6Depth4Adj[i] += adjustmentList[k].convLayer1Kernel6Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel7Depth1Adj[i] += adjustmentList[k].convLayer1Kernel7Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel7Depth2Adj[i] += adjustmentList[k].convLayer1Kernel7Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel7Depth3Adj[i] += adjustmentList[k].convLayer1Kernel7Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel7Depth4Adj[i] += adjustmentList[k].convLayer1Kernel7Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel8Depth1Adj[i] += adjustmentList[k].convLayer1Kernel8Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel8Depth2Adj[i] += adjustmentList[k].convLayer1Kernel8Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel8Depth3Adj[i] += adjustmentList[k].convLayer1Kernel8Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel8Depth4Adj[i] += adjustmentList[k].convLayer1Kernel8Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel9Depth1Adj[i] += adjustmentList[k].convLayer1Kernel9Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel9Depth2Adj[i] += adjustmentList[k].convLayer1Kernel9Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel9Depth3Adj[i] += adjustmentList[k].convLayer1Kernel9Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel9Depth4Adj[i] += adjustmentList[k].convLayer1Kernel9Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel10Depth1Adj[i] += adjustmentList[k].convLayer1Kernel10Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel10Depth2Adj[i] += adjustmentList[k].convLayer1Kernel10Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel10Depth3Adj[i] += adjustmentList[k].convLayer1Kernel10Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel10Depth4Adj[i] += adjustmentList[k].convLayer1Kernel10Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel11Depth1Adj[i] += adjustmentList[k].convLayer1Kernel11Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel11Depth2Adj[i] += adjustmentList[k].convLayer1Kernel11Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel11Depth3Adj[i] += adjustmentList[k].convLayer1Kernel11Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel11Depth4Adj[i] += adjustmentList[k].convLayer1Kernel11Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel12Depth1Adj[i] += adjustmentList[k].convLayer1Kernel12Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel12Depth2Adj[i] += adjustmentList[k].convLayer1Kernel12Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel12Depth3Adj[i] += adjustmentList[k].convLayer1Kernel12Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel12Depth4Adj[i] += adjustmentList[k].convLayer1Kernel12Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel13Depth1Adj[i] += adjustmentList[k].convLayer1Kernel13Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel13Depth2Adj[i] += adjustmentList[k].convLayer1Kernel13Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel13Depth3Adj[i] += adjustmentList[k].convLayer1Kernel13Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel13Depth4Adj[i] += adjustmentList[k].convLayer1Kernel13Depth4Adj[i];
                            adjustmentList[0].convLayer1Kernel14Depth1Adj[i] += adjustmentList[k].convLayer1Kernel14Depth1Adj[i];
                            adjustmentList[0].convLayer1Kernel14Depth2Adj[i] += adjustmentList[k].convLayer1Kernel14Depth2Adj[i];
                            adjustmentList[0].convLayer1Kernel14Depth3Adj[i] += adjustmentList[k].convLayer1Kernel14Depth3Adj[i];
                            adjustmentList[0].convLayer1Kernel14Depth4Adj[i] += adjustmentList[k].convLayer1Kernel14Depth4Adj[i];
                        }
                        adjustmentList[0].convLayer1Kernel1Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel1Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel1Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel1Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel2Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel2Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel2Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel2Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel3Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel3Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel3Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel3Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel4Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel4Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel4Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel4Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel5Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel5Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel5Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel5Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel6Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel6Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel6Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel6Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel7Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel7Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel7Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel7Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel8Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel8Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel8Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel8Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel9Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel9Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel9Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel9Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel10Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel10Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel10Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel10Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel11Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel11Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel11Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel11Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel12Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel12Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel12Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel12Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel13Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel13Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel13Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel13Depth4Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel14Depth1Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel14Depth2Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel14Depth3Adj[i] /= miniBatchSize;
                        adjustmentList[0].convLayer1Kernel14Depth4Adj[i] /= miniBatchSize;
                    }
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel1Depth1Adj, "convLayer1WeightsKernel1Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel1Depth2Adj, "convLayer1WeightsKernel1Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel1Depth3Adj, "convLayer1WeightsKernel1Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel1Depth4Adj, "convLayer1WeightsKernel1Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel2Depth1Adj, "convLayer1WeightsKernel2Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel2Depth2Adj, "convLayer1WeightsKernel2Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel2Depth3Adj, "convLayer1WeightsKernel2Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel2Depth4Adj, "convLayer1WeightsKernel2Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel3Depth1Adj, "convLayer1WeightsKernel3Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel3Depth2Adj, "convLayer1WeightsKernel3Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel3Depth3Adj, "convLayer1WeightsKernel3Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel3Depth4Adj, "convLayer1WeightsKernel3Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel4Depth1Adj, "convLayer1WeightsKernel4Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel4Depth2Adj, "convLayer1WeightsKernel4Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel4Depth3Adj, "convLayer1WeightsKernel4Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel4Depth4Adj, "convLayer1WeightsKernel4Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel5Depth1Adj, "convLayer1WeightsKernel5Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel5Depth2Adj, "convLayer1WeightsKernel5Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel5Depth3Adj, "convLayer1WeightsKernel5Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel5Depth4Adj, "convLayer1WeightsKernel5Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel6Depth1Adj, "convLayer1WeightsKernel6Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel6Depth2Adj, "convLayer1WeightsKernel6Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel6Depth3Adj, "convLayer1WeightsKernel6Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel6Depth4Adj, "convLayer1WeightsKernel6Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel7Depth1Adj, "convLayer1WeightsKernel7Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel7Depth2Adj, "convLayer1WeightsKernel7Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel7Depth3Adj, "convLayer1WeightsKernel7Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel7Depth4Adj, "convLayer1WeightsKernel7Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel8Depth1Adj, "convLayer1WeightsKernel8Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel8Depth2Adj, "convLayer1WeightsKernel8Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel8Depth3Adj, "convLayer1WeightsKernel8Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel8Depth4Adj, "convLayer1WeightsKernel8Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel9Depth1Adj, "convLayer1WeightsKernel9Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel9Depth2Adj, "convLayer1WeightsKernel9Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel9Depth3Adj, "convLayer1WeightsKernel9Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel9Depth4Adj, "convLayer1WeightsKernel9Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel10Depth1Adj, "convLayer1WeightsKernel10Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel10Depth2Adj, "convLayer1WeightsKernel10Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel10Depth3Adj, "convLayer1WeightsKernel10Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel10Depth4Adj, "convLayer1WeightsKernel10Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel11Depth1Adj, "convLayer1WeightsKernel11Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel11Depth2Adj, "convLayer1WeightsKernel11Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel11Depth3Adj, "convLayer1WeightsKernel11Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel11Depth4Adj, "convLayer1WeightsKernel11Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel12Depth1Adj, "convLayer1WeightsKernel12Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel12Depth2Adj, "convLayer1WeightsKernel12Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel12Depth3Adj, "convLayer1WeightsKernel12Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel12Depth4Adj, "convLayer1WeightsKernel12Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel13Depth1Adj, "convLayer1WeightsKernel13Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel13Depth2Adj, "convLayer1WeightsKernel13Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel13Depth3Adj, "convLayer1WeightsKernel13Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel13Depth4Adj, "convLayer1WeightsKernel13Depth4", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel14Depth1Adj, "convLayer1WeightsKernel14Depth1", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel14Depth2Adj, "convLayer1WeightsKernel14Depth2", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel14Depth3Adj, "convLayer1WeightsKernel14Depth3", 0);
                    funcs.rectified_adam_optimizer(adjustmentList[0].convLayer1Kernel14Depth4Adj, "convLayer1WeightsKernel14Depth4", 0);
                    for (int i = 0; i < 32; i++)
                    {
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[0].depth1[i] -= backProp.convLayer1Kernel1Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[0].depth2[i] -= backProp.convLayer1Kernel1Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[0].depth3[i] -= backProp.convLayer1Kernel1Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[0].depth4[i] -= backProp.convLayer1Kernel1Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[1].depth1[i] -= backProp.convLayer1Kernel2Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[1].depth2[i] -= backProp.convLayer1Kernel2Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[1].depth3[i] -= backProp.convLayer1Kernel2Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[1].depth4[i] -= backProp.convLayer1Kernel2Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[2].depth1[i] -= backProp.convLayer1Kernel3Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[2].depth2[i] -= backProp.convLayer1Kernel3Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[2].depth3[i] -= backProp.convLayer1Kernel3Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[2].depth4[i] -= backProp.convLayer1Kernel3Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[3].depth1[i] -= backProp.convLayer1Kernel4Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[3].depth2[i] -= backProp.convLayer1Kernel4Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[3].depth3[i] -= backProp.convLayer1Kernel4Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[3].depth4[i] -= backProp.convLayer1Kernel4Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[4].depth1[i] -= backProp.convLayer1Kernel5Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[4].depth2[i] -= backProp.convLayer1Kernel5Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[4].depth3[i] -= backProp.convLayer1Kernel5Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[4].depth4[i] -= backProp.convLayer1Kernel5Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[5].depth1[i] -= backProp.convLayer1Kernel6Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[5].depth2[i] -= backProp.convLayer1Kernel6Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[5].depth3[i] -= backProp.convLayer1Kernel6Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[5].depth4[i] -= backProp.convLayer1Kernel6Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[6].depth1[i] -= backProp.convLayer1Kernel7Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[6].depth2[i] -= backProp.convLayer1Kernel7Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[6].depth3[i] -= backProp.convLayer1Kernel7Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[6].depth4[i] -= backProp.convLayer1Kernel7Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[7].depth1[i] -= backProp.convLayer1Kernel8Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[7].depth2[i] -= backProp.convLayer1Kernel8Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[7].depth3[i] -= backProp.convLayer1Kernel8Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[7].depth4[i] -= backProp.convLayer1Kernel8Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[8].depth1[i] -= backProp.convLayer1Kernel9Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[8].depth2[i] -= backProp.convLayer1Kernel9Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[8].depth3[i] -= backProp.convLayer1Kernel9Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[8].depth4[i] -= backProp.convLayer1Kernel9Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[9].depth1[i] -= backProp.convLayer1Kernel10Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[9].depth2[i] -= backProp.convLayer1Kernel10Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[9].depth3[i] -= backProp.convLayer1Kernel10Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[9].depth4[i] -= backProp.convLayer1Kernel10Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[10].depth1[i] -= backProp.convLayer1Kernel11Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[10].depth2[i] -= backProp.convLayer1Kernel11Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[10].depth3[i] -= backProp.convLayer1Kernel11Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[10].depth4[i] -= backProp.convLayer1Kernel11Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[11].depth1[i] -= backProp.convLayer1Kernel12Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[11].depth2[i] -= backProp.convLayer1Kernel12Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[11].depth3[i] -= backProp.convLayer1Kernel12Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[11].depth4[i] -= backProp.convLayer1Kernel12Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[12].depth1[i] -= backProp.convLayer1Kernel13Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[12].depth2[i] -= backProp.convLayer1Kernel13Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[12].depth3[i] -= backProp.convLayer1Kernel13Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[12].depth4[i] -= backProp.convLayer1Kernel13Depth4_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[13].depth1[i] -= backProp.convLayer1Kernel14Depth1_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[13].depth2[i] -= backProp.convLayer1Kernel14Depth2_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[13].depth3[i] -= backProp.convLayer1Kernel14Depth3_adapted_rate[i];
                        predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[13].depth4[i] -= backProp.convLayer1Kernel14Depth4_adapted_rate[i];
                    }
                    miniBatchIdx = 0;
                    prevAvgCrossEntropyPerBatch = avgCrossEntropyPerBatch;
                    avgCrossEntropyPerBatch = 0;
                }
            }
            if(trainingActivated == true)
            {
                dataInputCtrl.RunWorkerAsync();
            }
        }

        public class PredictorThreadWork
        {
            public static void appOpen()
            {
                Application.Run(predictorGui1);
            }
        }

        private void exitBtn_Click(object sender, EventArgs e)
        {
            if(File.Exists(@"X:\mlpSecondLayerWeightsFlatFile.txt"))
            {
                File.Delete(@"X:\mlpSecondLayerWeightsFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\mlpSecondLayerWeightsFlatFile.txt");
                for (int i = 0; i < 192; i++)
                {
                    output.WriteLine(networkArray[0].mlpStructs[0].secondLayerWeights[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\mlpFirstLayerWeightsFlatFile.txt"))
            {
                File.Delete(@"X:\mlpFirstLayerWeightsFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\mlpFirstLayerWeightsFlatFile.txt");
                for (int i = 0; i < 96000; i++)
                {
                    output.WriteLine(networkArray[0].mlpStructs[0].firstLayerWeights[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\mlpFirstLayerPReLUParamsFlatFile.txt"))
            {
                File.Delete(@"X:\mlpFirstLayerPReLUParamsFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\mlpFirstLayerPReLUParamsFlatFile.txt");
                for (int i = 0; i < 64; i++)
                {
                    output.WriteLine(networkArray[0].mlpStructs[0].mlpLayer1PReLUParam[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\mlpFirstLayerBiasFlatFile.txt"))
            {
                File.Delete(@"X:\mlpFirstLayerBiasFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\mlpFirstLayerBiasFlatFile.txt");
                for (int i = 0; i < 64; i++)
                {
                    output.WriteLine(networkArray[0].mlpStructs[0].mlpLayer1Bias[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\affineMLPTransformerWeightsFlatFile1.txt"))
            {
                File.Delete(@"X:\affineMLPTransformerWeightsFlatFile1.txt");
                StreamWriter output = File.AppendText(@"X:\affineMLPTransformerWeightsFlatFile1.txt");
                File.Delete(@"X:\affineMLPTransformerWeightsFlatFile2.txt");
                StreamWriter output2 = File.AppendText(@"X:\affineMLPTransformerWeightsFlatFile2.txt");

                for (int i = 0; i < 900; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].affineTransWeights1[i].ToString());
                    output2.WriteLine(networkArray[0].transStructs[0].affineTransWeights2[i].ToString());
                }
                output.Close();
                output2.Close();
            }
            if(File.Exists(@"X:\queryLinearLayerWeightsHead1FlatFile.txt"))
            {
                File.Delete(@"X:\queryLinearLayerWeightsHead1FlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\queryLinearLayerWeightsHead1FlatFile.txt");
                File.Delete(@"X:\keyLinearLayerWeightsHead1FlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\keyLinearLayerWeightsHead1FlatFile.txt");
                File.Delete(@"X:\valueLinearLayerWeightsHead1FlatFile.txt");
                StreamWriter output3 = File.AppendText(@"X:\valueLinearLayerWeightsHead1FlatFile.txt");

                File.Delete(@"X:\queryLinearLayerWeightsHead2FlatFile.txt");
                StreamWriter output4 = File.AppendText(@"X:\queryLinearLayerWeightsHead2FlatFile.txt");
                File.Delete(@"X:\keyLinearLayerWeightsHead2FlatFile.txt");
                StreamWriter output5 = File.AppendText(@"X:\keyLinearLayerWeightsHead2FlatFile.txt");
                File.Delete(@"X:\valueLinearLayerWeightsHead2FlatFile.txt");
                StreamWriter output6 = File.AppendText(@"X:\valueLinearLayerWeightsHead2FlatFile.txt");

                File.Delete(@"X:\queryLinearLayerWeightsHead3FlatFile.txt");
                StreamWriter output7 = File.AppendText(@"X:\queryLinearLayerWeightsHead3FlatFile.txt");
                File.Delete(@"X:\keyLinearLayerWeightsHead3FlatFile.txt");
                StreamWriter output8 = File.AppendText(@"X:\keyLinearLayerWeightsHead3FlatFile.txt");
                File.Delete(@"X:\valueLinearLayerWeightsHead3FlatFile.txt");
                StreamWriter output9 = File.AppendText(@"X:\valueLinearLayerWeightsHead3FlatFile.txt");

                for(int i = 0; i < 75; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].queryLinearLayerWeights_head1[i].ToString());
                    output2.WriteLine(networkArray[0].transStructs[0].keyLinearLayerWeights_head1[i].ToString());
                    output3.WriteLine(networkArray[0].transStructs[0].valueLinearLayerWeights_head1[i].ToString());

                    output4.WriteLine(networkArray[0].transStructs[0].queryLinearLayerWeights_head2[i].ToString());
                    output5.WriteLine(networkArray[0].transStructs[0].keyLinearLayerWeights_head2[i].ToString());
                    output6.WriteLine(networkArray[0].transStructs[0].valueLinearLayerWeights_head2[i].ToString());

                    output7.WriteLine(networkArray[0].transStructs[0].queryLinearLayerWeights_head3[i].ToString());
                    output8.WriteLine(networkArray[0].transStructs[0].keyLinearLayerWeights_head3[i].ToString());
                    output9.WriteLine(networkArray[0].transStructs[0].valueLinearLayerWeights_head3[i].ToString());
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
            if (File.Exists(@"X:\affineMLPBiasesFlatFile.txt"))
            {
                File.Delete(@"X:\affineMLPBiasesFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\affineMLPBiasesFlatFile.txt");
                for (int i = 0; i < 6000; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].transPReLUBias[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\affineMLPPreluParamFlatFile.txt"))
            {
                File.Delete(@"X:\affineMLPPreluParamFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\affineMLPPreluParamFlatFile.txt");
                for (int i = 0; i < 6000; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].transPReLUParam[i].ToString());
                }
                output.Close();
            }
            if(File.Exists(@"X:\affineMLPSecondLayerBiasesFlatFile.txt"))
            {
                File.Delete(@"X:\affineMLPSecondLayerBiasesFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\affineMLPSecondLayerBiasesFlatFile.txt");
                for(int i = 0; i < 1500; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].transMLPSecondLayerBias[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\addAndNorm1GammaFlatFile.txt"))
            {
                File.Delete(@"X:\addAndNorm1GammaFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\addAndNorm1GammaFlatFile.txt");
                File.Delete(@"X:\addAndNorm1BetaFlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\addAndNorm1BetaFlatFile.txt");
                File.Delete(@"X:\addAndNorm2GammaFlatFile.txt");
                StreamWriter output3 = File.AppendText(@"X:\addAndNorm2GammaFlatFile.txt");
                File.Delete(@"X:\addAndNorm2BetaFlatFile.txt");
                StreamWriter output4 = File.AppendText(@"X:\addAndNorm2BetaFlatFile.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].addAndNorm1Gamma[i].ToString());
                    output2.WriteLine(networkArray[0].transStructs[0].addAndNorm1Beta[i].ToString());
                    output3.WriteLine(networkArray[0].transStructs[0].addAndNorm2Gamma[i].ToString());
                    output4.WriteLine(networkArray[0].transStructs[0].addAndNorm2Beta[i].ToString());
                }
                output.Close();
                output2.Close();
                output3.Close();
                output4.Close();
            }
            if(File.Exists(@"X:\finalLinearLayerWeightsFlatFile.txt"))
            {
                File.Delete(@"X:\finalLinearLayerWeightsFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\finalLinearLayerWeightsFlatFile.txt");
                for(int i = 0; i < 225; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].finalLinearLayerWeights[i].ToString());
                }
                output.Close();
            }
            if(File.Exists(@"X:\convLayerBias1FlatFile.txt"))
            {
                File.Delete(@"X:\convLayerBias1FlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\convLayerBias1FlatFile.txt");
                File.Delete(@"X:\convLayerBias2FlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\convLayerBias2FlatFile.txt");
                File.Delete(@"X:\convLayerBias3FlatFile.txt");
                StreamWriter output3 = File.AppendText(@"X:\convLayerBias3FlatFile.txt");
                File.Delete(@"X:\convLayerBias4FlatFile.txt");
                StreamWriter output4 = File.AppendText(@"X:\convLayerBias4FlatFile.txt");
                File.Delete(@"X:\convLayerBias5FlatFile.txt");
                StreamWriter output5 = File.AppendText(@"X:\convLayerBias5FlatFile.txt");
                for(int i = 0; i < 1400; i++)
                {
                    output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Bias[i].ToString());
                    output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Bias[i].ToString());
                    output3.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Bias[i].ToString());
                    output4.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Bias[i].ToString());
                    output5.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Bias[i].ToString());
                }
                output.Close();
                output2.Close();
                output3.Close();
                output4.Close();
                output5.Close();
            }
            if (File.Exists(@"X:\convLayerPReLUParams1FlatFile.txt"))
            {
                File.Delete(@"X:\convLayerPReLUParams1FlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\convLayerPReLUParams1FlatFile.txt");
                File.Delete(@"X:\convLayerPReLUParams2FlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\convLayerPReLUParams2FlatFile.txt");
                File.Delete(@"X:\convLayerPReLUParams3FlatFile.txt");
                StreamWriter output3 = File.AppendText(@"X:\convLayerPReLUParams3FlatFile.txt");
                File.Delete(@"X:\convLayerPReLUParams4FlatFile.txt");
                StreamWriter output4 = File.AppendText(@"X:\convLayerPReLUParams4FlatFile.txt");
                File.Delete(@"X:\convLayerPReLUParams5FlatFile.txt");
                StreamWriter output5 = File.AppendText(@"X:\convLayerPReLUParams5FlatFile.txt");
                File.Delete(@"X:\convLayerNormGammaFlatFile.txt");
                StreamWriter output6 = File.AppendText(@"X:\convLayerNormGammaFlatFile.txt");
                File.Delete(@"X:\convLayerNormBetaFlatFile.txt");
                StreamWriter output7 = File.AppendText(@"X:\convLayerNormBetaFlatFile.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1PReLUParam[i].ToString());
                    output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2PReLUParam[i].ToString());
                    output3.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3PReLUParam[i].ToString());
                    output4.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4PReLUParam[i].ToString());
                    output5.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5PReLUParam[i].ToString());
                    output6.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormGamma[i].ToString());
                    output7.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormBeta[i].ToString());
                }
                output.Close();
                output2.Close();
                output3.Close();
                output4.Close();
                output5.Close();
                output6.Close();
                output7.Close();
            }
            //loop to handle all the different convolutional kernels for convolutional layer 1
            for (int i = 0; i < 14; i++)
            {
                if (File.Exists(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                {
                    File.Delete(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    StreamWriter output = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    File.Delete(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    File.Delete(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth3FlatFile.txt");
                    StreamWriter output3 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth3FlatFile.txt");
                    File.Delete(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth4FlatFile.txt");
                    StreamWriter output4 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth4FlatFile.txt");
                    for (int j = 0; j < 32; j++)
                    {
                        output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth1[j].ToString());
                        output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth2[j].ToString());
                        output3.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth3[j].ToString());
                        output4.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth4[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                    output3.Close();
                    output4.Close();
                }
                if (File.Exists(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                {
                    File.Delete(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    StreamWriter output = File.AppendText(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    File.Delete(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    for (int j = 0; j < 14; j++)
                    {
                        output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth1[j].ToString());
                        output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth2[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                }
                if (File.Exists(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                {
                    File.Delete(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    StreamWriter output = File.AppendText(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    File.Delete(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    for (int j = 0; j < 14; j++)
                    {
                        output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth1[j].ToString());
                        output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth2[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                }
                if (File.Exists(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                {
                    File.Delete(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    StreamWriter output = File.AppendText(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    File.Delete(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    for (int j = 0; j < 14; j++)
                    {
                        output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth1[j].ToString());
                        output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth2[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                }
                if (File.Exists(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                {
                    File.Delete(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    StreamWriter output = File.AppendText(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    File.Delete(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    for (int j = 0; j < 14; j++)
                    {
                        output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth1[j].ToString());
                        output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth2[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                }
            }

            if(!Directory.Exists(@"X:\previousDayData"))
            {
                Directory.CreateDirectory(@"X:\previousDayData");
            }
            if (File.Exists(@"X:\previousDayData\dayNum.txt"))
            {
                File.Delete(@"X:\previousDayData\dayNum.txt");
            }
            StreamWriter dayNumOut = File.AppendText(@"X:\previousDayData\dayNum.txt");
            dayNumOut.WriteLine(dayNum.ToString());
            dayNumOut.Close();

            if (predictorGui1.accumPricesSizes.Checked == true)
            {
                if (entireDaysPrices.Count != 0)
                {
                    StreamWriter entireDayPricesOut = File.AppendText(@"X:\previousDayData\entireDayPrices" + dayNum.ToString() + ".txt");
                    StreamWriter entireDaySizesOut = File.AppendText(@"X:\previousDayData\entireDaySizes" + dayNum.ToString() + ".txt");
                    for (int i = 0; i < entireDaysPrices.Count; i++)
                    {
                        entireDayPricesOut.WriteLine(entireDaysPrices[i].ToString());
                        entireDaySizesOut.WriteLine(entireDaysSizes[i].ToString());
                    }
                    entireDayPricesOut.Close();
                    entireDaySizesOut.Close();
                }
            }

            if(predictorGui1.buildTrdata.Checked == true)
            {
                for (int i = startingExIdx; i < exampleIdx; i++)
                {
                    StreamWriter output = File.AppendText(@"X:\trainingData\entireDayPrices.forEx" + i.ToString() + ".txt");
                    output.WriteLine("entireDayPrices" + (dayNum - 1).ToString() + ".txt");
                    output.Close();
                    StreamWriter output2 = File.AppendText(@"X:\trainingData\entireDaySizes.forEx" + i.ToString() + ".txt");
                    output2.WriteLine("entireDaySizes" + (dayNum - 1).ToString() + ".txt");
                    output2.Close();
                }
                File.Copy(@"X:\previousDayData\entireDayPrices" + (dayNum - 1).ToString() + ".txt", @"X:\trainingData\entireDayPrices" + (dayNum - 1).ToString() + ".txt");
                File.Copy(@"X:\previousDayData\entireDaySizes" + (dayNum - 1).ToString() + ".txt", @"X:\trainingData\entireDaySizes" + (dayNum - 1).ToString() + ".txt");
            }

            predictorGui1.Close();
            Environment.Exit(1);
        }

        private void changeNum_Click(object sender, EventArgs e)
        {
            changeExNum = true;
        }

        public void transFilterShow(int block)
        {
            Bitmap filter1Graphic = new Bitmap(100, 100);
            Bitmap filter2Graphic = new Bitmap(100, 100);
            Bitmap filter3Graphic = new Bitmap(100, 100);
            int bitmapRowIdx = 0;
            int temp;
            int temp2;
            int temp3;
            int bitmapColIdx = 0;
            if (block == 2)
            {
                for (int i = 0; i < 10000; i++)
                {
                    if (i % 100 == 0 && i != 0)
                    {
                        bitmapRowIdx++;
                        bitmapColIdx = 0;
                    }
                    if (networkArray[0].transStructs[0].attention_filter_head1[i] != 0.0)
                    {
                        temp = Convert.ToInt32(networkArray[0].transStructs[0].attention_filter_head1[i] * 255.0);
                    }
                    else
                    {
                        temp = 0;
                    }
                    filter1Graphic.SetPixel(bitmapColIdx, bitmapRowIdx, Color.FromArgb(temp, temp, temp));
                    if (networkArray[0].transStructs[0].attention_filter_head2[i] != 0.0)
                    {
                        temp2 = Convert.ToInt32(networkArray[0].transStructs[0].attention_filter_head2[i] * 255.0);
                    }
                    else
                    {
                        temp2 = 0;
                    }
                    filter2Graphic.SetPixel(bitmapColIdx, bitmapRowIdx, Color.FromArgb(temp2, temp2, temp2));
                    if (networkArray[0].transStructs[0].attention_filter_head3[i] != 0.0)
                    {
                        temp3 = Convert.ToInt32(networkArray[0].transStructs[0].attention_filter_head3[i] * 255.0);
                    }
                    else
                    {
                        temp3 = 0;
                    }
                    filter3Graphic.SetPixel(bitmapColIdx, bitmapRowIdx, Color.FromArgb(temp3, temp3, temp3));
                    bitmapColIdx++;
                }
            }
            if (block == 1)
            {
                for (int i = 0; i < 10000; i++)
                {
                    if (i % 100 == 0 && i != 0)
                    {
                        bitmapRowIdx++;
                        bitmapColIdx = 0;
                    }
                    if (networkArray[0].transStructs[0].attention_filter_head1[i] != 0.0)
                    {
                        temp = Convert.ToInt32(networkArray[0].transStructs[0].attention_filter_head1[i] * 255.0);
                    }
                    else
                    {
                        temp = 0;
                    }
                    filter1Graphic.SetPixel(bitmapColIdx, bitmapRowIdx, Color.FromArgb(temp, temp, temp));
                    if (networkArray[0].transStructs[0].attention_filter_head2[i] != 0.0)
                    {
                        temp2 = Convert.ToInt32(networkArray[0].transStructs[0].attention_filter_head2[i] * 255.0);
                    }
                    else
                    {
                        temp2 = 0;
                    }
                    filter2Graphic.SetPixel(bitmapColIdx, bitmapRowIdx, Color.FromArgb(temp2, temp2, temp2));
                    if (networkArray[0].transStructs[0].attention_filter_head3[i] != 0.0)
                    {
                        temp3 = Convert.ToInt32(networkArray[0].transStructs[0].attention_filter_head3[i] * 255.0);
                    }
                    else
                    {
                        temp3 = 0;
                    }
                    filter3Graphic.SetPixel(bitmapColIdx, bitmapRowIdx, Color.FromArgb(temp3, temp3, temp3));
                    bitmapColIdx++;
                }
            }

            filter1Graphic = ResizeImage(filter1Graphic, attentionFilter1.Width, attentionFilter1.Height);
            filter2Graphic = ResizeImage(filter2Graphic, attentionFilter2.Width, attentionFilter2.Height);
            filter3Graphic = ResizeImage(filter3Graphic, attentionFilter3.Width, attentionFilter3.Height);

            if (block == 1)
            {
                attentionFilter1.Image = filter1Graphic;
                attentionFilter2.Image = filter2Graphic;
                attentionFilter3.Image = filter3Graphic;
            }
            else
            {
                attentionFilter1Block2.Image = filter1Graphic;
                attentionFilter2Block2.Image = filter2Graphic;
                attentionFilter3Block2.Image = filter3Graphic;
            }
        }

        public void convLayer5OutShow()
        {
            double[] transformerInputMat;
            matrixOps matOps = new matrixOps();
            double[] feature1 = new double[100];
            double[] feature2 = new double[100];
            double[] feature3 = new double[100];
            double[] feature4 = new double[100];
            double[] feature5 = new double[100];
            double[] feature6 = new double[100];
            double[] feature7 = new double[100];
            double[] feature8 = new double[100];
            double[] feature9 = new double[100];
            double[] feature10 = new double[100];
            double[] feature11 = new double[100];
            double[] feature12 = new double[100];
            double[] feature13 = new double[100];
            double[] feature14 = new double[100];
            double[] feature15 = new double[100];
            int temp;
            int rowIdx = 0;

            Bitmap convLayer5Feature1Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature2Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature3Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature4Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature5Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature6Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature7Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature8Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature9Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature10Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature11Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature12Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature13Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature14Graphic = new Bitmap(100, 1);
            Bitmap convLayer5Feature15Graphic = new Bitmap(100, 1);

            transformerInputMat = matOps.transposeMat(networkArray[0].transStructs[0].transformerInput, 15, 100);

            for(int i = 0; i < 100; i++)
            {
                feature1[i] = transformerInputMat[rowIdx];
                if(feature1[i] < 0)
                {
                    feature1[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature1[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature1[i] * 255.0);
                }
                if(temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature1Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature2[i] = transformerInputMat[rowIdx];
                if (feature2[i] < 0)
                {
                    feature2[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature2[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature2[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature2Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature3[i] = transformerInputMat[rowIdx];
                if (feature3[i] < 0)
                {
                    feature3[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature3[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature3[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature3Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature4[i] = transformerInputMat[rowIdx];
                if (feature4[i] < 0)
                {
                    feature4[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature4[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature4[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature4Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature5[i] = transformerInputMat[rowIdx];
                if (feature5[i] < 0)
                {
                    feature5[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature5[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature5[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature5Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature6[i] = transformerInputMat[rowIdx];
                if (feature6[i] < 0)
                {
                    feature6[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature6[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature6[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature6Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature7[i] = transformerInputMat[rowIdx];
                if (feature7[i] < 0)
                {
                    feature7[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature7[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature7[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature7Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature8[i] = transformerInputMat[rowIdx];
                if (feature8[i] < 0)
                {
                    feature8[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature8[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature8[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature8Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature9[i] = transformerInputMat[rowIdx];
                if (feature9[i] < 0)
                {
                    feature9[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature9[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature9[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature9Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature10[i] = transformerInputMat[rowIdx];
                if (feature10[i] < 0)
                {
                    feature10[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature10[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature10[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature10Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature11[i] = transformerInputMat[rowIdx];
                if (feature11[i] < 0)
                {
                    feature11[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature11[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature11[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature11Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature12[i] = transformerInputMat[rowIdx];
                if (feature12[i] < 0)
                {
                    feature12[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature12[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature12[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature12Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature13[i] = transformerInputMat[rowIdx];
                if (feature13[i] < 0)
                {
                    feature13[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature13[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature13[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature13Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature14[i] = transformerInputMat[rowIdx];
                if (feature14[i] < 0)
                {
                    feature14[i] = 0;
                }
                if (predictorGui1.preluSelect.Checked == true || predictorGui1.mishSelect.Checked == true)
                {
                    temp = Convert.ToInt32(feature14[i] * 255.0);
                }
                else
                {
                    temp = Convert.ToInt32(feature14[i] * 255.0);
                }
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature14Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            for (int i = 0; i < 100; i++)
            {
                feature15[i] = transformerInputMat[rowIdx];
                if (feature15[i] < 0)
                {
                    feature15[i] = 0;
                }
                temp = Convert.ToInt32(feature15[i] * 255.0);
                if (temp > 255)
                {
                    temp = 255;
                }
                convLayer5Feature15Graphic.SetPixel(i, 0, Color.FromArgb(temp, temp, temp));
                rowIdx++;
            }
            convLayer5Feature1Graphic = ResizeImage(convLayer5Feature1Graphic, convLayer5Feature1.Width, convLayer5Feature1.Height);
            convLayer5Feature1.Image = convLayer5Feature1Graphic;
            convLayer5Feature2Graphic = ResizeImage(convLayer5Feature2Graphic, convLayer5Feature2.Width, convLayer5Feature2.Height);
            convLayer5Feature2.Image = convLayer5Feature2Graphic;
            convLayer5Feature3Graphic = ResizeImage(convLayer5Feature3Graphic, convLayer5Feature3.Width, convLayer5Feature3.Height);
            convLayer5Feature3.Image = convLayer5Feature3Graphic;
            convLayer5Feature4Graphic = ResizeImage(convLayer5Feature4Graphic, convLayer5Feature4.Width, convLayer5Feature4.Height);
            convLayer5Feature4.Image = convLayer5Feature4Graphic;
            convLayer5Feature5Graphic = ResizeImage(convLayer5Feature5Graphic, convLayer5Feature5.Width, convLayer5Feature5.Height);
            convLayer5Feature5.Image = convLayer5Feature5Graphic;
            convLayer5Feature6Graphic = ResizeImage(convLayer5Feature6Graphic, convLayer5Feature6.Width, convLayer5Feature6.Height);
            convLayer5Feature6.Image = convLayer5Feature6Graphic;
            convLayer5Feature7Graphic = ResizeImage(convLayer5Feature7Graphic, convLayer5Feature7.Width, convLayer5Feature7.Height);
            convLayer5Feature7.Image = convLayer5Feature7Graphic;
            convLayer5Feature8Graphic = ResizeImage(convLayer5Feature8Graphic, convLayer5Feature8.Width, convLayer5Feature8.Height);
            convLayer5Feature8.Image = convLayer5Feature8Graphic;
            convLayer5Feature9Graphic = ResizeImage(convLayer5Feature9Graphic, convLayer5Feature9.Width, convLayer5Feature9.Height);
            convLayer5Feature9.Image = convLayer5Feature9Graphic;
            convLayer5Feature10Graphic = ResizeImage(convLayer5Feature10Graphic, convLayer5Feature10.Width, convLayer5Feature10.Height);
            convLayer5Feature10.Image = convLayer5Feature10Graphic;
            convLayer5Feature11Graphic = ResizeImage(convLayer5Feature11Graphic, convLayer5Feature11.Width, convLayer5Feature11.Height);
            convLayer5Feature11.Image = convLayer5Feature11Graphic;
            convLayer5Feature12Graphic = ResizeImage(convLayer5Feature12Graphic, convLayer5Feature12.Width, convLayer5Feature12.Height);
            convLayer5Feature12.Image = convLayer5Feature12Graphic;
            convLayer5Feature13Graphic = ResizeImage(convLayer5Feature13Graphic, convLayer5Feature13.Width, convLayer5Feature13.Height);
            convLayer5Feature13.Image = convLayer5Feature13Graphic;
            convLayer5Feature14Graphic = ResizeImage(convLayer5Feature14Graphic, convLayer5Feature14.Width, convLayer5Feature14.Height);
            convLayer5Feature14.Image = convLayer5Feature14Graphic;
            convLayer5Feature15Graphic = ResizeImage(convLayer5Feature15Graphic, convLayer5Feature15.Width, convLayer5Feature15.Height);
            convLayer5Feature15.Image = convLayer5Feature15Graphic;
        }

        public void posEncodingShow()
        {
            int bitmapRowIdx = 0;
            int bitmapColIdx = 0;
            double tempDecimal;
            int temp;
            Bitmap posEncodingImage = new Bitmap(100, 15);

            for (int i = 0; i < 1500; i++)
            {
                if (i % 100 == 0 && i != 0)
                {
                    bitmapRowIdx++;
                    bitmapColIdx = 0;
                }
                if(networkArray[0].transStructs[0].positionalEncodingArray[i] < 0)
                {
                    tempDecimal = 0;
                }
                else
                {
                    tempDecimal = networkArray[0].transStructs[0].positionalEncodingArray[i];
                }
                temp = Convert.ToInt32(tempDecimal * 255.0);
                posEncodingImage.SetPixel(bitmapColIdx, bitmapRowIdx, Color.FromArgb(temp, temp, temp));
                bitmapColIdx++;
            }
            posEncodingImage = ResizeImage(posEncodingImage, predictorGui1.posEncodingGraphic.Width, predictorGui1.posEncodingGraphic.Height);
            predictorGui1.posEncodingGraphic.Image = posEncodingImage;
        }

        public Bitmap ResizeImage(System.Drawing.Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(300, 300);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        public void exampleClassCount()
        {
            numOfUpExamples = 0;
            numOfFlatExamples = 0;
            numOfDownExamples = 0;
            for (int i = 0; i < backProp.listOfTrainingExamples.Count; i++)
            {
                string[] inputLines;
                inputLines = File.ReadAllLines(@"X:\trainingData\trainingTensorExample" + backProp.listOfTrainingExamples[i].ToString() + ".gt.txt");
                backProp.actualOutcomes[0] = Convert.ToInt32(inputLines[0]);
                backProp.actualOutcomes[1] = Convert.ToInt32(inputLines[1]);
                backProp.actualOutcomes[2] = Convert.ToInt32(inputLines[2]);
                if(backProp.actualOutcomes[0] == 1)
                {
                    numOfUpExamples++;
                }
                if(backProp.actualOutcomes[1] == 1)
                {
                    numOfFlatExamples++;
                }
                if(backProp.actualOutcomes[2] == 1)
                {
                    numOfDownExamples++;
                }
            }
            backProp.actualOutcomes[0] = 0;
            backProp.actualOutcomes[1] = 0;
            backProp.actualOutcomes[2] = 0;
        }

        private void reloadWeights_Click(object sender, EventArgs e)
        {
            convModule.lecun_normal_init_layer(1);
            convModule.lecun_normal_init_layer(2);
            convModule.lecun_normal_init_layer(3);
            convModule.lecun_normal_init_layer(4);
            convModule.lecun_normal_init_layer(5);
            convModule.convLayerBiases_init();
            convModule.convLayerPReLUParams_init();
            convModule.convLayerNormGammaBetaInit();
            transformerModule.tfixupInit_attention_linearLayer(1);
            transformerModule.xavier_init_affineMLPLayers(1);
            mlp.xavier_init_weights(1);
            mlp.xavier_init_weights(2);
            mlp.mlpLayerBiases_init1();
            mlp.mlpLayerPReLUParams_init1();
            transformerModule.addAndNormGammaBetaInit();
            transformerModule.transMLPBiases_init();
            transformerModule.transMLPPReLUParams_init();
        }

        private void saveWeights_Click(object sender, EventArgs e)
        {
            if (File.Exists(@"X:\mlpSecondLayerWeightsFlatFile.txt"))
            {
                File.Delete(@"X:\mlpSecondLayerWeightsFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\mlpSecondLayerWeightsFlatFile.txt");
                for (int i = 0; i < 192; i++)
                {
                    output.WriteLine(networkArray[0].mlpStructs[0].secondLayerWeights[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\mlpFirstLayerWeightsFlatFile.txt"))
            {
                File.Delete(@"X:\mlpFirstLayerWeightsFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\mlpFirstLayerWeightsFlatFile.txt");
                for (int i = 0; i < 96000; i++)
                {
                    output.WriteLine(networkArray[0].mlpStructs[0].firstLayerWeights[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\mlpFirstLayerPReLUParamsFlatFile.txt"))
            {
                File.Delete(@"X:\mlpFirstLayerPReLUParamsFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\mlpFirstLayerPReLUParamsFlatFile.txt");
                for (int i = 0; i < 64; i++)
                {
                    output.WriteLine(networkArray[0].mlpStructs[0].mlpLayer1PReLUParam[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\mlpFirstLayerBiasFlatFile.txt"))
            {
                File.Delete(@"X:\mlpFirstLayerBiasFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\mlpFirstLayerBiasFlatFile.txt");
                for (int i = 0; i < 64; i++)
                {
                    output.WriteLine(networkArray[0].mlpStructs[0].mlpLayer1Bias[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\affineMLPTransformerWeightsFlatFile1.txt"))
            {
                File.Delete(@"X:\affineMLPTransformerWeightsFlatFile1.txt");
                StreamWriter output = File.AppendText(@"X:\affineMLPTransformerWeightsFlatFile1.txt");
                File.Delete(@"X:\affineMLPTransformerWeightsFlatFile2.txt");
                StreamWriter output2 = File.AppendText(@"X:\affineMLPTransformerWeightsFlatFile2.txt");
                File.Delete(@"X:\affineMLPTransformerWeightsFlatFile3.txt");

                for (int i = 0; i < 900; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].affineTransWeights1[i].ToString());
                    output2.WriteLine(networkArray[0].transStructs[0].affineTransWeights2[i].ToString());
                }
                output.Close();
                output2.Close();
            }
            if (File.Exists(@"X:\queryLinearLayerWeightsHead1FlatFile.txt"))
            {
                File.Delete(@"X:\queryLinearLayerWeightsHead1FlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\queryLinearLayerWeightsHead1FlatFile.txt");
                File.Delete(@"X:\keyLinearLayerWeightsHead1FlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\keyLinearLayerWeightsHead1FlatFile.txt");
                File.Delete(@"X:\valueLinearLayerWeightsHead1FlatFile.txt");
                StreamWriter output3 = File.AppendText(@"X:\valueLinearLayerWeightsHead1FlatFile.txt");

                File.Delete(@"X:\queryLinearLayerWeightsHead2FlatFile.txt");
                StreamWriter output4 = File.AppendText(@"X:\queryLinearLayerWeightsHead2FlatFile.txt");
                File.Delete(@"X:\keyLinearLayerWeightsHead2FlatFile.txt");
                StreamWriter output5 = File.AppendText(@"X:\keyLinearLayerWeightsHead2FlatFile.txt");
                File.Delete(@"X:\valueLinearLayerWeightsHead2FlatFile.txt");
                StreamWriter output6 = File.AppendText(@"X:\valueLinearLayerWeightsHead2FlatFile.txt");

                File.Delete(@"X:\queryLinearLayerWeightsHead3FlatFile.txt");
                StreamWriter output7 = File.AppendText(@"X:\queryLinearLayerWeightsHead3FlatFile.txt");
                File.Delete(@"X:\keyLinearLayerWeightsHead3FlatFile.txt");
                StreamWriter output8 = File.AppendText(@"X:\keyLinearLayerWeightsHead3FlatFile.txt");
                File.Delete(@"X:\valueLinearLayerWeightsHead3FlatFile.txt");
                StreamWriter output9 = File.AppendText(@"X:\valueLinearLayerWeightsHead3FlatFile.txt");

                for (int i = 0; i < 75; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].queryLinearLayerWeights_head1[i].ToString());
                    output2.WriteLine(networkArray[0].transStructs[0].keyLinearLayerWeights_head1[i].ToString());
                    output3.WriteLine(networkArray[0].transStructs[0].valueLinearLayerWeights_head1[i].ToString());

                    output4.WriteLine(networkArray[0].transStructs[0].queryLinearLayerWeights_head2[i].ToString());
                    output5.WriteLine(networkArray[0].transStructs[0].keyLinearLayerWeights_head2[i].ToString());
                    output6.WriteLine(networkArray[0].transStructs[0].valueLinearLayerWeights_head2[i].ToString());

                    output7.WriteLine(networkArray[0].transStructs[0].queryLinearLayerWeights_head3[i].ToString());
                    output8.WriteLine(networkArray[0].transStructs[0].keyLinearLayerWeights_head3[i].ToString());
                    output9.WriteLine(networkArray[0].transStructs[0].valueLinearLayerWeights_head3[i].ToString());
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
            if (File.Exists(@"X:\affineMLPBiasesFlatFile.txt"))
            {
                File.Delete(@"X:\affineMLPBiasesFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\affineMLPBiasesFlatFile.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].transPReLUBias[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\affineMLPPreluParamFlatFile.txt"))
            {
                File.Delete(@"X:\affineMLPPreluParamFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\affineMLPPreluParamFlatFile.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].transPReLUParam[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\affineMLPSecondLayerBiasesFlatFile.txt"))
            {
                File.Delete(@"X:\affineMLPSecondLayerBiasesFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\affineMLPSecondLayerBiasesFlatFile.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].transMLPSecondLayerBias[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\addAndNorm1GammaFlatFile.txt"))
            {
                File.Delete(@"X:\addAndNorm1GammaFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\addAndNorm1GammaFlatFile.txt");
                File.Delete(@"X:\addAndNorm1BetaFlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\addAndNorm1BetaFlatFile.txt");
                File.Delete(@"X:\addAndNorm2GammaFlatFile.txt");
                StreamWriter output3 = File.AppendText(@"X:\addAndNorm2GammaFlatFile.txt");
                File.Delete(@"X:\addAndNorm2BetaFlatFile.txt");
                StreamWriter output4 = File.AppendText(@"X:\addAndNorm2BetaFlatFile.txt");
                for (int i = 0; i < 1500; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].addAndNorm1Gamma[i].ToString());
                    output2.WriteLine(networkArray[0].transStructs[0].addAndNorm1Beta[i].ToString());
                    output3.WriteLine(networkArray[0].transStructs[0].addAndNorm2Gamma[i].ToString());
                    output4.WriteLine(networkArray[0].transStructs[0].addAndNorm2Beta[i].ToString());
                }
                output.Close();
                output2.Close();
                output3.Close();
                output4.Close();
            }
            if (File.Exists(@"X:\finalLinearLayerWeightsFlatFile.txt"))
            {
                File.Delete(@"X:\finalLinearLayerWeightsFlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\finalLinearLayerWeightsFlatFile.txt");
                for (int i = 0; i < 225; i++)
                {
                    output.WriteLine(networkArray[0].transStructs[0].finalLinearLayerWeights[i].ToString());
                }
                output.Close();
            }
            if (File.Exists(@"X:\convLayerBias1FlatFile.txt"))
            {
                File.Delete(@"X:\convLayerBias1FlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\convLayerBias1FlatFile.txt");
                File.Delete(@"X:\convLayerBias2FlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\convLayerBias2FlatFile.txt");
                File.Delete(@"X:\convLayerBias3FlatFile.txt");
                StreamWriter output3 = File.AppendText(@"X:\convLayerBias3FlatFile.txt");
                File.Delete(@"X:\convLayerBias4FlatFile.txt");
                StreamWriter output4 = File.AppendText(@"X:\convLayerBias4FlatFile.txt");
                File.Delete(@"X:\convLayerBias5FlatFile.txt");
                StreamWriter output5 = File.AppendText(@"X:\convLayerBias5FlatFile.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Bias[i].ToString());
                    output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Bias[i].ToString());
                    output3.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Bias[i].ToString());
                    output4.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Bias[i].ToString());
                    output5.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Bias[i].ToString());
                }
                output.Close();
                output2.Close();
                output3.Close();
                output4.Close();
                output5.Close();
            }
            if (File.Exists(@"X:\convLayerPReLUParams1FlatFile.txt"))
            {
                File.Delete(@"X:\convLayerPReLUParams1FlatFile.txt");
                StreamWriter output = File.AppendText(@"X:\convLayerPReLUParams1FlatFile.txt");
                File.Delete(@"X:\convLayerPReLUParams2FlatFile.txt");
                StreamWriter output2 = File.AppendText(@"X:\convLayerPReLUParams2FlatFile.txt");
                File.Delete(@"X:\convLayerPReLUParams3FlatFile.txt");
                StreamWriter output3 = File.AppendText(@"X:\convLayerPReLUParams3FlatFile.txt");
                File.Delete(@"X:\convLayerPReLUParams4FlatFile.txt");
                StreamWriter output4 = File.AppendText(@"X:\convLayerPReLUParams4FlatFile.txt");
                File.Delete(@"X:\convLayerPReLUParams5FlatFile.txt");
                StreamWriter output5 = File.AppendText(@"X:\convLayerPReLUParams5FlatFile.txt");
                File.Delete(@"X:\convLayerNormGammaFlatFile.txt");
                StreamWriter output6 = File.AppendText(@"X:\convLayerNormGammaFlatFile.txt");
                File.Delete(@"X:\convLayerNormBetaFlatFile.txt");
                StreamWriter output7 = File.AppendText(@"X:\convLayerNormBetaFlatFile.txt");
                for (int i = 0; i < 1400; i++)
                {
                    output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1PReLUParam[i].ToString());
                    output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2PReLUParam[i].ToString());
                    output3.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3PReLUParam[i].ToString());
                    output4.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4PReLUParam[i].ToString());
                    output5.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5PReLUParam[i].ToString());
                    output6.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormGamma[i].ToString());
                    output7.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5OutputNormBeta[i].ToString());
                }
                output.Close();
                output2.Close();
                output3.Close();
                output4.Close();
                output5.Close();
                output6.Close();
                output7.Close();
            }
            //loop to handle all the different convolutional kernels for convolutional layer 1
            for (int i = 0; i < 14; i++)
            {
                if (File.Exists(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                {
                    File.Delete(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    StreamWriter output = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    File.Delete(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    File.Delete(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth3FlatFile.txt");
                    StreamWriter output3 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth3FlatFile.txt");
                    File.Delete(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth4FlatFile.txt");
                    StreamWriter output4 = File.AppendText(@"X:\convLayer1Kernel" + (i + 1).ToString() + "_depth4FlatFile.txt");
                    for (int j = 0; j < 32; j++)
                    {
                        output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth1[j].ToString());
                        output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth2[j].ToString());
                        output3.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth3[j].ToString());
                        output4.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer1Kernel1[i].depth4[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                    output3.Close();
                    output4.Close();
                }
                if (File.Exists(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                {
                    File.Delete(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    StreamWriter output = File.AppendText(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    File.Delete(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\convLayer2Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    for (int j = 0; j < 14; j++)
                    {
                        output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth1[j].ToString());
                        output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer2Kernel2[i].depth2[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                }
                if (File.Exists(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                {
                    File.Delete(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    StreamWriter output = File.AppendText(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    File.Delete(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\convLayer3Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    for (int j = 0; j < 14; j++)
                    {
                        output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth1[j].ToString());
                        output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer3Kernel3[i].depth2[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                }
                if (File.Exists(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                {
                    File.Delete(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    StreamWriter output = File.AppendText(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    File.Delete(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\convLayer4Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    for (int j = 0; j < 14; j++)
                    {
                        output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth1[j].ToString());
                        output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer4Kernel4[i].depth2[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                }
                if (File.Exists(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt"))
                {
                    File.Delete(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    StreamWriter output = File.AppendText(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth1FlatFile.txt");
                    File.Delete(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    StreamWriter output2 = File.AppendText(@"X:\convLayer5Kernel" + (i + 1).ToString() + "_depth2FlatFile.txt");
                    for (int j = 0; j < 14; j++)
                    {
                        output.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth1[j].ToString());
                        output2.WriteLine(predictorGui.networkArray[0].convStructs[0].convLayer5Kernel5[i].depth2[j].ToString());
                    }
                    output.Close();
                    output2.Close();
                }
            }
        }

        public void norm_inputs_resp_to_input_mean_and_std(int idx)
        {
            double meanOfPrices = 0;
            double meanOfSizes = 0;
            double stdPrices = 0;
            double stdSizes = 0;

            for (int i = 0; i < 3200; i++)
            {
                meanOfPrices += tensorIn[idx].price[i];
                meanOfSizes += tensorIn[idx].size[i];
            }
            meanOfPrices /= 3200;
            meanOfSizes /= 3200;

            for(int i = 0; i < 3200; i++)
            {
                stdPrices += Math.Pow(tensorIn[idx].price[i] - meanOfPrices, 2);
                stdSizes += Math.Pow(tensorIn[idx].size[i] - meanOfSizes, 2);
            }
            stdPrices /= 3200;
            stdSizes /= 3200;

            stdPrices = Math.Sqrt(stdPrices);
            stdSizes = Math.Sqrt(stdSizes);

            for(int i = 0; i < 3200; i++)
            {
                tensorIn[idx].price[i] = (tensorIn[idx].price[i] - meanOfPrices) / stdPrices;
                tensorIn[idx].size[i] = (tensorIn[idx].size[i] - meanOfSizes) / stdSizes;
            }
        }

        public void norm_inputs_resp_to_prev_day_mean_and_std(int idx)
        {
            double meanOfPrices = 0;
            double meanOfSizes = 0;
            double stdPrices = 0;
            double stdSizes = 0;
            string[] pricesFileName;
            string[] sizesFileName;
            int exIdx;
            string[] arr;
            string[] arr2;

            if(trainingActivated == true)
            {
                exIdx = trainingExNum;
                pricesFileName = File.ReadAllLines(@"X:\trainingData\entireDayPrices.forEx" + exIdx.ToString() + ".txt");
                sizesFileName = File.ReadAllLines(@"X:\trainingData\entireDaySizes.forEx" + exIdx.ToString() + ".txt");
                arr = File.ReadAllLines(@"X:\trainingData\" + pricesFileName[0]);
                arr2 = File.ReadAllLines(@"X:\trainingData\" + sizesFileName[0]);
            }
            else
            {
                arr = File.ReadAllLines(@"X:\previousDayData\entireDayPrices" + (dayNum - 1).ToString() + ".txt");
                arr2 = File.ReadAllLines(@"X:\previousDayData\entireDaySizes" + (dayNum - 1).ToString() + ".txt");
            }
            for (int i = 0; i < arr.Length; i++)
            {
                meanOfPrices += Convert.ToDouble(arr[i]) / arr.Length;
                meanOfSizes += Convert.ToDouble(arr2[i]) / arr2.Length;
            }

            for (int i = 0; i < arr.Length; i++)
            {
                stdPrices += Math.Pow(Convert.ToDouble(arr[i]) - meanOfPrices, 2) / arr.Length;
                stdSizes += Math.Pow(Convert.ToDouble(arr2[i]) - meanOfSizes, 2) / arr2.Length;
            }

            stdPrices = Math.Sqrt(stdPrices);
            stdSizes = Math.Sqrt(stdSizes);

            for (int i = 0; i < 3200; i++)
            {
                tensorIn[idx].price[i] = (tensorIn[idx].price[i] - meanOfPrices) / stdPrices;
                tensorIn[idx].size[i] = (tensorIn[idx].size[i] - meanOfSizes) / stdSizes;
            }

            //further normalization to bring down scales of widely varying examples
            if (globalScaledMaxPrice == 0 && buildTrdata.Checked == false)
            {
                string[] scalingVals = File.ReadAllLines(@"X:\min_max_scaling_values.txt");
                globalScaledMaxPrice = Convert.ToDouble(scalingVals[0]);
                globalScaledMinPrice = Convert.ToDouble(scalingVals[1]);
                globalScaledMaxSize = Convert.ToDouble(scalingVals[2]);
                globalScaledMinSize = Convert.ToDouble(scalingVals[3]);
            }
            double minPrice = globalScaledMinPrice;
            double maxPrice = globalScaledMaxPrice;
            double minSize = globalScaledMinSize;
            double maxSize = globalScaledMaxSize;

            for (int i = 0; i < 3200; i++)
            {
                tensorIn[idx].price[i] = (tensorIn[idx].price[i] - minPrice) / (maxPrice - minPrice);
                tensorIn[idx].size[i] = (tensorIn[idx].size[i] - minSize) / (maxSize - minSize);
            }
        }

        private void changeIterNum_Click(object sender, EventArgs e)
        {
            changeIterNumFlag = true;
        }

        private void changeEpochNum_Click(object sender, EventArgs e)
        {
            changeEpochNumFlag = true;
        }

        private void changePercentIgnored_Click(object sender, EventArgs e)
        {
            changePercentIgnoredNumFlag = true;
        }

        private void selectedLayerWeights_SelectedIndexChanged(object sender, EventArgs e)
        {
            weightStepsOutput.Text = "";
            if (selectedLayerWeights.Text == "convLayer1Kernel1Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel1Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel1Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel1Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel1Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel1Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel1Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel1Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel2Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel2Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel2Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel2Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel2Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel2Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel2Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel2Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel3Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel3Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel3Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel3Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel3Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel3Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel3Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel3Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel4Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel4Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel4Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel4Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel4Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel4Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel4Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel4Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel5Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel5Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel5Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel5Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel5Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel5Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel5Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel5Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel6Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel6Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel6Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel6Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel6Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel6Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel6Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel6Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel7Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel7Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel7Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel7Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel7Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel7Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel7Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel7Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel8Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel8Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel8Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel8Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel8Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel8Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel8Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel8Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel9Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel9Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel9Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel9Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel9Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel9Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel9Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel9Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel10Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel10Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel10Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel10Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel10Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel10Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel10Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel10Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel11Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel11Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel11Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel11Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel11Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel11Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel11Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel11Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel12Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel12Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel12Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel12Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel12Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel12Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel12Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel12Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel13Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel13Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel13Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel13Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel13Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel13Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel13Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel13Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel14Depth1")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel14Depth1_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel14Depth2")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel14Depth2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel14Depth3")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel14Depth3_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Kernel14Depth4")
            {
                for (int i = 0; i < 32; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Kernel14Depth4_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "convLayer1Biases")
            {
                for (int i = 0; i < 1400; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.convLayer1Bias_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "queryHead1")
            {
                for (int i = 0; i < 75; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.queryHead1Block2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "queryHead2")
            {
                for (int i = 0; i < 75; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.queryHead2Block2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
            else if (selectedLayerWeights.Text == "queryHead3")
            {
                for (int i = 0; i < 75; i++)
                {
                    weightStepsOutput.Text += (-1 * backProp.queryHead3Block2_adapted_rate[i]).ToString() + "\r\n";
                }
            }
        }

        private void selectedLayer_SelectedIndexChanged(object sender, EventArgs e)
        {
            layerActivationsOutput.Text = "";
            int minibatchExSelectVal = Convert.ToInt32(minibatchExSelect.Text);
            if (selectedLayer.Text == "inputToConvModule")
            {
                for (int i = 0; i < 3200; i++)
                {
                    layerActivationsOutput.Text += tensorIn[minibatchExSelectVal].price[i].ToString() + ' ' + tensorIn[minibatchExSelectVal].size[i].ToString() + "\r\n";
                }
            }
            else if (selectedLayer.Text == "mlpLayer1")
            {
                for(int i = 0; i < 64; i++)
                {
                    layerActivationsOutput.Text += networkArray[0].mlpStructs[0].firstLayerOut[i].ToString() + "\r\n";
                }
            }
            else if (selectedLayer.Text == "mlpLayer2")
            {
                for (int i = 0; i < 3; i++)
                {
                    layerActivationsOutput.Text += networkArray[0].mlpStructs[0].secondLayerOutRaw[i].ToString() + "\r\n";
                }
            }
            else if (selectedLayer.Text == "TransformerBlock1Output")
            {
                for (int i = 0; i < 1500; i++)
                {
                    layerActivationsOutput.Text += networkArray[0].transStructs[minibatchExSelectVal].transformerBlock1Output[i].ToString() + "\r\n";
                }
            }
            else if (selectedLayer.Text == "TransformerBlock2Output")
            {
                for (int i = 0; i < 1500; i++)
                {
                    layerActivationsOutput.Text += networkArray[0].transStructs[minibatchExSelectVal].transformerBlock2Output[i].ToString() + "\r\n";
                }
            }
            else if (selectedLayer.Text == "convolutional layer 1")
            {
                for (int i = 0; i < 1400; i++)
                {
                    layerActivationsOutput.Text += networkArray[0].convStructs[minibatchExSelectVal].convLayer1Output[i].ToString() + "\r\n";
                }
            }
            else if (selectedLayer.Text == "convolutional layer 2")
            {
                for (int i = 0; i < 1400; i++)
                {
                    layerActivationsOutput.Text += networkArray[0].convStructs[minibatchExSelectVal].convLayer2Output[i].ToString() + "\r\n";
                }
            }
            else if (selectedLayer.Text == "convolutional layer 3")
            {
                for (int i = 0; i < 1400; i++)
                {
                    layerActivationsOutput.Text += networkArray[0].convStructs[minibatchExSelectVal].convLayer3Output[i].ToString() + "\r\n";
                }
            }
            else if (selectedLayer.Text == "convolutional layer 4")
            {
                for (int i = 0; i < 1400; i++)
                {
                    layerActivationsOutput.Text += networkArray[0].convStructs[minibatchExSelectVal].convLayer4Output[i].ToString() + "\r\n";
                }
            }
            else if (selectedLayer.Text == "convolutional layer 5")
            {
                for (int i = 0; i < 1400; i++)
                {
                    layerActivationsOutput.Text += networkArray[0].convStructs[minibatchExSelectVal].convLayer5Output[i].ToString() + "\r\n";
                }
            }
            else if (selectedLayer.Text == "Temporally Encoded Feature Map")
            {
                for (int i = 0; i < 1500; i++)
                {
                    layerActivationsOutput.Text += networkArray[0].convStructs[minibatchExSelectVal].temporalEncodedNormOutput[i].ToString() + "\r\n";
                }
            }
        }
    }
}
