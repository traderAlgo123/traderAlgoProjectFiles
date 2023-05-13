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
    public class Event
    {
        public double[] prices = new double[32];
        public double[] sizes = new double[32];
    }
}
