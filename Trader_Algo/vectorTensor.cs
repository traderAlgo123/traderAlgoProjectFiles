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

namespace Data_Scraper
{
    public class vectorTensor
    {
        public Vector[] vecTensor = new Vector[32];

        public vectorTensor()
        {
            for(int i = 0; i < 32; i++)
            {
                vecTensor[i] = new Vector();
            }
        }

        public void copyTensor(vectorTensor srcTensor, vectorTensor dstTensor)
        {
            for(int i = 0; i < 32; i++)
            {
                dstTensor.vecTensor[i].vectorPrice = srcTensor.vecTensor[i].vectorPrice;
                dstTensor.vecTensor[i].vectorSize = srcTensor.vecTensor[i].vectorSize;
                dstTensor.vecTensor[i].vectorPriceTier = srcTensor.vecTensor[i].vectorPriceTier;
            }
        }

        public int cmpTensor(vectorTensor prevTensor, vectorTensor currentTensor)
        {
            for(int i = 0; i < 32; i++)
            {
                if(currentTensor.vecTensor[i].vectorPrice != prevTensor.vecTensor[i].vectorPrice ||
                   currentTensor.vecTensor[i].vectorSize != prevTensor.vecTensor[i].vectorSize)
                {
                    if(algoGui.critErr == true)
                    {
                        algoGui.critErr = false;
                        return 0;
                    }
                    return 1;
                }
            }
            return 0;
        }
    }
}
