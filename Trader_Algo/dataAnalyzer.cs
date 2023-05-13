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
using System.IO;
using System.Threading.Tasks;
using System.Threading;
using System.ComponentModel;
using System.Collections;

namespace Data_Scraper
{
    public class averages_array_obj
    {
        public float[] averages_array = new float[9];
    }

    public class dataAnalyzer
    {
        public averages_array_obj[] averages = new averages_array_obj[32];
        public void tensorBuild(bool bidsOrAsks)
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            if (bidsOrAsks)
            {
                int tensorIdx = 15;
                int priceTierAssign = 0;
                for (int i = 0; i < 16; i++)
                {
                    if (i + 1 != 16)
                    {
                        if (algoGui.bids.bidsArray[i] == algoGui.bids.bidsArray[i + 1])
                        {
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPriceTier = (price_tier)priceTierAssign;
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPrice = algoGui.bids.bidsArray[i];
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorSize = algoGui.bids.bidsSizeArray[i];
                            tensorIdx--;
                        }
                        else if (algoGui.bids.bidsArray[i] > algoGui.bids.bidsArray[i + 1] && priceTierAssign != 4)
                        {
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPriceTier = (price_tier)priceTierAssign;
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPrice = algoGui.bids.bidsArray[i];
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorSize = algoGui.bids.bidsSizeArray[i];
                            priceTierAssign++;
                            tensorIdx--;
                        }
                        else if (priceTierAssign == 4)
                        {
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPriceTier = (price_tier)priceTierAssign;
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPrice = algoGui.bids.bidsArray[i];
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorSize = algoGui.bids.bidsSizeArray[i];
                            tensorIdx--;
                        }
                    }
                    else
                    {
                        algoGui.tempTensor.vecTensor[tensorIdx].vectorPriceTier = (price_tier)priceTierAssign;
                        algoGui.tempTensor.vecTensor[tensorIdx].vectorPrice = algoGui.bids.bidsArray[i];
                        algoGui.tempTensor.vecTensor[tensorIdx].vectorSize = algoGui.bids.bidsSizeArray[i];
                    }
                }
                algoGui.bidsCompletedEvent.Set();
            }
            else
            {
                int tensorIdx = 16;
                int priceTierAssign = 0;
                for(int i = 0; i < 16; i++)
                {
                    if (i + 1 != 16)
                    {
                        if (algoGui.asks.asksArray[i] == algoGui.asks.asksArray[i + 1])
                        {
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPriceTier = (price_tier)priceTierAssign;
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPrice = algoGui.asks.asksArray[i];
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorSize = algoGui.asks.asksSizeArray[i];
                            tensorIdx++;
                        }
                        else if (algoGui.asks.asksArray[i] < algoGui.asks.asksArray[i + 1] && priceTierAssign != 4)
                        {
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPriceTier = (price_tier)priceTierAssign;
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPrice = algoGui.asks.asksArray[i];
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorSize = algoGui.asks.asksSizeArray[i];
                            priceTierAssign++;
                            tensorIdx++;
                        }
                        else if (priceTierAssign == 4)
                        {
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPriceTier = (price_tier)priceTierAssign;
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorPrice = algoGui.asks.asksArray[i];
                            algoGui.tempTensor.vecTensor[tensorIdx].vectorSize = algoGui.asks.asksSizeArray[i];
                            tensorIdx++;
                        }
                    }
                    else
                    {
                        algoGui.tempTensor.vecTensor[tensorIdx].vectorPriceTier = (price_tier)priceTierAssign;
                        algoGui.tempTensor.vecTensor[tensorIdx].vectorPrice = algoGui.asks.asksArray[i];
                        algoGui.tempTensor.vecTensor[tensorIdx].vectorSize = algoGui.asks.asksSizeArray[i];
                    }
                }
                algoGui.asksCompletedEvent.Set();
            }
        }

        public float greensAvg()
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            float bidsPrice = 0.0f;
            int bidsTotalSize = 0;
            float asksPrice = 0.0f;
            int asksTotalSize = 0;

            float average = 0.0f;

            for(int i = 0; i < 16; i++)
            {
                if(algoGui.tempTensor.vecTensor[i].vectorPriceTier == price_tier.GREEN)
                {
                    bidsPrice = algoGui.tempTensor.vecTensor[i].vectorPrice;
                    bidsTotalSize += algoGui.tempTensor.vecTensor[i].vectorSize;
                }
            }

            for(int i = 16; i < 32; i++)
            {
                if (algoGui.tempTensor.vecTensor[i].vectorPriceTier == price_tier.GREEN)
                {
                    asksPrice = algoGui.tempTensor.vecTensor[i].vectorPrice;
                    asksTotalSize += algoGui.tempTensor.vecTensor[i].vectorSize;
                }
            }

            average = ((bidsPrice * bidsTotalSize) + (asksPrice * asksTotalSize)) / (bidsTotalSize + asksTotalSize);

            algoGui.algoGui_array[5].hrsIndicator.Text = average.ToString();
            algoGui.bidsCompletedEvent.Set();
            return average;
        }

        public float redsAvg()
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            float bidsPrice = 0.0f;
            int bidsTotalSize = 0;
            float asksPrice = 0.0f;
            int asksTotalSize = 0;

            float average = 0.0f;

            for (int i = 0; i < 16; i++)
            {
                if (algoGui.tempTensor.vecTensor[i].vectorPriceTier == price_tier.RED)
                {
                    bidsPrice = algoGui.tempTensor.vecTensor[i].vectorPrice;
                    bidsTotalSize += algoGui.tempTensor.vecTensor[i].vectorSize;
                }
            }

            for (int i = 16; i < 32; i++)
            {
                if (algoGui.tempTensor.vecTensor[i].vectorPriceTier == price_tier.RED)
                {
                    asksPrice = algoGui.tempTensor.vecTensor[i].vectorPrice;
                    asksTotalSize += algoGui.tempTensor.vecTensor[i].vectorSize;
                }
            }

            average = ((bidsPrice * bidsTotalSize) + (asksPrice * asksTotalSize)) / (bidsTotalSize + asksTotalSize);

            algoGui.algoGui_array[5].hrsIndicator.Text += "    " + average.ToString();
            algoGui.bidsCompletedEvent.Set();
            return average;
        }

        public float yellowsAvg()
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            float bidsPrice = 0.0f;
            int bidsTotalSize = 0;
            float asksPrice = 0.0f;
            int asksTotalSize = 0;

            float average = 0.0f;

            for (int i = 0; i < 16; i++)
            {
                if (algoGui.tempTensor.vecTensor[i].vectorPriceTier == price_tier.YELLOW)
                {
                    bidsPrice = algoGui.tempTensor.vecTensor[i].vectorPrice;
                    bidsTotalSize += algoGui.tempTensor.vecTensor[i].vectorSize;
                }
            }

            for (int i = 16; i < 32; i++)
            {
                if (algoGui.tempTensor.vecTensor[i].vectorPriceTier == price_tier.YELLOW)
                {
                    asksPrice = algoGui.tempTensor.vecTensor[i].vectorPrice;
                    asksTotalSize += algoGui.tempTensor.vecTensor[i].vectorSize;
                }
            }

            average = ((bidsPrice * bidsTotalSize) + (asksPrice * asksTotalSize)) / (bidsTotalSize + asksTotalSize);

            algoGui.algoGui_array[5].hrsIndicator.Text += "    " + average.ToString();
            algoGui.bidsCompletedEvent.Set();
            return average;
        }

        public void midpoint()
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            float midpoint;
            midpoint = ((algoGui.tempTensor.vecTensor[15].vectorPrice + algoGui.tempTensor.vecTensor[16].vectorPrice) / 2);

            algoGui.algoGui_array[5].hrsIndicator.Text = midpoint.ToString() + "     " + algoGui.tensor_count.ToString();
        }

        public void bidsAsksErrorCheckerVoter()
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;

            Parallel.For(0, 32, (i, state) =>
            {
                ArrayList votes = new ArrayList();
            });
        }

        public void bidsAsksErrorCheckerAverager()
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            float[] averages_array_rounded = new float[9];
            float[] temp = new float[32];
            Parallel.For(0, 32, (i, state) =>
            {
                averages[i] = new averages_array_obj();
                if (algoGui.tempTensor.vecTensor[i].vectorSize == 0)
                {
                    algoGui.tempTensor.vecTensor[i].vectorSize = 500;
                }
                if (i == 0)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i + 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i + 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i + 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i + 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i + 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i + 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i + 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i + 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 9].vectorPrice) / 2;
                }
                else if (i == 1)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 9].vectorPrice) / 2;
                }
                else if (i == 2)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 9].vectorPrice) / 2;
                }
                else if (i == 3)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 9].vectorPrice) / 2;
                }
                else if (i == 4)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 9].vectorPrice) / 2;
                }
                else if (i == 5)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 9].vectorPrice) / 2;
                }
                else if (i == 6)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 9].vectorPrice) / 2;
                }
                else if (i == 7)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 9].vectorPrice) / 2;
                }
                else if (i == 8)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 9].vectorPrice) / 2;
                }
                else if (i == 31)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i - 1].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i - 2].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i - 3].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i - 4].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i - 5].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i - 6].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i - 7].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 9].vectorPrice + algoGui.tempTensor.vecTensor[i - 8].vectorPrice) / 2;
                }
                else if (i == 30)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 9].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                }
                else if (i == 29)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 9].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                }
                else if (i == 28)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 9].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                }
                else if (i == 27)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 9].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                }
                else if (i == 26)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 9].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                }
                else if (i == 25)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 9].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                }
                else if (i == 24)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 9].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                }
                else if (i == 23)
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 9].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                }
                else
                {
                    averages[i].averages_array[0] = (algoGui.tempTensor.vecTensor[i - 1].vectorPrice + algoGui.tempTensor.vecTensor[i + 1].vectorPrice) / 2;
                    averages[i].averages_array[1] = (algoGui.tempTensor.vecTensor[i - 2].vectorPrice + algoGui.tempTensor.vecTensor[i + 2].vectorPrice) / 2;
                    averages[i].averages_array[2] = (algoGui.tempTensor.vecTensor[i - 3].vectorPrice + algoGui.tempTensor.vecTensor[i + 3].vectorPrice) / 2;
                    averages[i].averages_array[3] = (algoGui.tempTensor.vecTensor[i - 4].vectorPrice + algoGui.tempTensor.vecTensor[i + 4].vectorPrice) / 2;
                    averages[i].averages_array[4] = (algoGui.tempTensor.vecTensor[i - 5].vectorPrice + algoGui.tempTensor.vecTensor[i + 5].vectorPrice) / 2;
                    averages[i].averages_array[5] = (algoGui.tempTensor.vecTensor[i - 6].vectorPrice + algoGui.tempTensor.vecTensor[i + 6].vectorPrice) / 2;
                    averages[i].averages_array[6] = (algoGui.tempTensor.vecTensor[i - 7].vectorPrice + algoGui.tempTensor.vecTensor[i + 7].vectorPrice) / 2;
                    averages[i].averages_array[7] = (algoGui.tempTensor.vecTensor[i - 8].vectorPrice + algoGui.tempTensor.vecTensor[i + 8].vectorPrice) / 2;
                    averages[i].averages_array[8] = (algoGui.tempTensor.vecTensor[i - 9].vectorPrice + algoGui.tempTensor.vecTensor[i + 9].vectorPrice) / 2;
                }

                for (int j = 0; j < 9; j++)
                {
                    averages_array_rounded[j] = (float)Math.Round(averages[i].averages_array[j], 0);
                }

                temp[i] = mostFrequent(averages_array_rounded, 9);
            });
            Parallel.For(0, 32, (i, state) =>
            {
                if (Math.Abs(algoGui.tempTensor.vecTensor[i].vectorPrice - temp[i]) > 3)
                {
                    algoGui.tempTensor.vecTensor[i].vectorPrice = temp[i];
                }
            });
            if(Math.Abs(algoGui.tempTensor.vecTensor[15].vectorPrice - algoGui.tempTensor.vecTensor[16].vectorPrice) > 5)
            {
                if(algoGui.tempTensor.vecTensor[15].vectorPrice > algoGui.tempTensor.vecTensor[16].vectorPrice)
                {
                    algoGui.tempTensor.vecTensor[16].vectorPrice = algoGui.tempTensor.vecTensor[15].vectorPrice;
                }
                else
                {
                    algoGui.tempTensor.vecTensor[15].vectorPrice = algoGui.tempTensor.vecTensor[16].vectorPrice;
                }
            }
            algoGui.bidsCompletedEvent.Set();
        }

        float mostFrequent(float[] arr, int n)
        {

            // Sort the array
            Array.Sort(arr);

            // find the max frequency using
            // linear traversal
            int max_count = 1;
            float res = arr[0];
            int curr_count = 1;

            for (int i = 1; i < n; i++)
            {
                if (arr[i] == arr[i - 1])
                    curr_count++;
                else
                {
                    if (curr_count > max_count)
                    {
                        max_count = curr_count;
                        res = arr[i - 1];
                    }
                    curr_count = 1;
                }
            }

            // If last element is most frequent
            if (curr_count > max_count)
            {
                max_count = curr_count;
                res = arr[n - 1];
            }

            return res;
        }
    }
}
