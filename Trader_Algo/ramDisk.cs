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
using System.Diagnostics;
using System.IO;
using System.Security;

namespace Data_Scraper
{
    public class RamDisk
    {
        public const string MountPoint = "X:";

        public void createRamDisk()
        {

            try
            {
                string initializeDisk = "imdisk -a ";
                string imdiskSize = "-s 8192M ";
                string mountPoint = "-m " + MountPoint + " ";


                ProcessStartInfo procStartInfo = new ProcessStartInfo();
                procStartInfo.UseShellExecute = false;
                procStartInfo.CreateNoWindow = true;
                procStartInfo.FileName = "cmd";
                procStartInfo.Arguments = "/C " + initializeDisk + imdiskSize + mountPoint;
                Process.Start(procStartInfo);

                formatRAMDisk();

            }
            catch (Exception objException)
            {
                Console.WriteLine("There was an Error, while trying to create a ramdisk! Do you have imdisk installed?");
                Console.WriteLine(objException);
            }

        }

        /**
         * since the format option with imdisk doesn't seem to work
         * use the fomat X: command via cmd
         * 
         * as I would say in german:
         * "Von hinten durch die Brust ins Auge"
         * **/
        private void formatRAMDisk()
        {

            string cmdFormatHDD = "format " + MountPoint + "/Q /FS:NTFS";

            SecureString password = new SecureString();
            password.AppendChar('0');
            password.AppendChar('8');
            password.AppendChar('1');
            password.AppendChar('5');

            ProcessStartInfo formatRAMDiskProcess = new ProcessStartInfo();
            formatRAMDiskProcess.UseShellExecute = false;
            formatRAMDiskProcess.CreateNoWindow = true;
            formatRAMDiskProcess.RedirectStandardInput = true;
            formatRAMDiskProcess.FileName = "cmd";
            formatRAMDiskProcess.Verb = "runas";
            formatRAMDiskProcess.UserName = "Administrator";
            formatRAMDiskProcess.Password = password;
            formatRAMDiskProcess.Arguments = "/C " + cmdFormatHDD;
            Process process = Process.Start(formatRAMDiskProcess);

            sendCMDInput(process);
        }

        private void sendCMDInput(Process process)
        {
            StreamWriter inputWriter = process.StandardInput;
            inputWriter.WriteLine("J");
            inputWriter.Flush();
            inputWriter.WriteLine("RAMDisk for valueable data");
            inputWriter.Flush();
        }

        public string getMountPoint()
        {
            return MountPoint;
        }
    }
}
