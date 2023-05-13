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

    }

    public class nnMLPStructs
    {

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
