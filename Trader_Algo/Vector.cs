﻿/*
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
    public enum price_tier : int
    {
        GREEN = 0,
        RED,
        YELLOW,
        BLUE,
        OUTLIER
    };

    public class Vector
    {
        public float vectorPrice = new float();
        public int vectorSize = new int();
        public price_tier vectorPriceTier = new price_tier();
    }
}
