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

namespace Data_Scraper
{
    /// <summary>
    /// Service to read texts from images through OCR Tesseract engine.
    /// </summary>
    public class TesseractService
    {
        private readonly string _tesseractExePath;
        private readonly string _language;

        /// <summary>
        /// Initializes a new instance of the <see cref="TesseractService"/> class.
        /// </summary>
        /// <param name="tesseractDir">The path for the Tesseract4 installation folder (C:\Program Files\Tesseract-OCR).</param>
        /// <param name="language">The language used to extract text from images (eng, por, etc)</param>
        /// <param name="dataDir">The data with the trained models (tessdata). Download the models from https://github.com/tesseract-ocr/tessdata_fast</param>
        public TesseractService(string tesseractDir, string language, string dataDir = null)
        {
            // Tesseract configs.
            _tesseractExePath = Path.Combine(tesseractDir, "tesseract.exe");
            _language = language;

            if (String.IsNullOrEmpty(dataDir))
                 dataDir = Path.Combine(tesseractDir, "tessdata");

            Environment.SetEnvironmentVariable("TESSDATA_PREFIX", dataDir);
        }

        /// <summary>
        /// Read text from the images streams.
        /// </summary>
        /// <param name="images">The images streams.</param>
        /// <returns>The images text.</returns>
        public string GetText(bool bidsOrAsksFlag, params Stream[] images)
        {
            var output = string.Empty;

            if (images.Any())
            {
                try
                {
                    if (bidsOrAsksFlag == true)
                    {
                        var info = new ProcessStartInfo
                        {
                            FileName = _tesseractExePath,
                            Arguments = @"X:\blk_white_test.jpg X:\test -l eng",
                            RedirectStandardError = true,
                            RedirectStandardOutput = true,
                            CreateNoWindow = true,
                            UseShellExecute = false
                        };

                        using (var ps = Process.Start(info))
                        {
                            ps.WaitForExit();

                            var exitCode = ps.ExitCode;

                            if (exitCode == 0)
                            {
                                output = File.ReadAllText(@"X:\test.txt");
                            }
                            else
                            {
                                var stderr = ps.StandardError.ReadToEnd();
                                throw new InvalidOperationException(stderr);
                            }
                        }
                    }
                    else
                    {
                        var info = new ProcessStartInfo
                        {
                            FileName = _tesseractExePath,
                            Arguments = @"X:\blk_white_test2.jpg X:\test2 -l eng digits",
                            RedirectStandardError = true,
                            RedirectStandardOutput = true,
                            CreateNoWindow = true,
                            UseShellExecute = false
                        };

                        using (var ps = Process.Start(info))
                        {
                            ps.WaitForExit();

                            var exitCode = ps.ExitCode;

                            if (exitCode == 0)
                            {
                                output = File.ReadAllText(@"X:\test2.txt");
                            }
                            else
                            {
                                var stderr = ps.StandardError.ReadToEnd();
                                throw new InvalidOperationException(stderr);
                            }
                        }
                    }
                }
                finally
                {
                    
                }
            }

            return output;
        }

        public string GetTextLastTrades(int index, params Stream[] images)
        {
            var output = string.Empty;

            if (images.Any())
            {
                try
                {
                    var info = new ProcessStartInfo
                    {
                        FileName = _tesseractExePath,
                        Arguments = @"X:\lastTradeLine" + index + @"blk_white.jpg X:\lastTradeLine" + index + " -l eng+deu+fra --psm 7 digits",
                        RedirectStandardError = true,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true,
                        UseShellExecute = false
                    };

                    using (var ps = Process.Start(info))
                    {
                        ps.WaitForExit();

                        var exitCode = ps.ExitCode;

                        if (exitCode == 0)
                        {
                            output = File.ReadAllText(@"X:\lastTradeLine" + index + ".txt");
                        }
                        else
                        {
                            var stderr = ps.StandardError.ReadToEnd();
                            throw new InvalidOperationException(stderr);
                        }
                    }
                }
                finally
                {

                }
            }

            return output;
        }

        public string GetTextBids(int index, params Stream[] images)
        {
            var output = string.Empty;

            if (images.Any())
            {
                try
                {
                    var info = new ProcessStartInfo
                    {
                        FileName = _tesseractExePath,
                        Arguments = @"X:\bidsLine" + index + @".jpg  X:\bidsLine" + index + " -l eng+deu+fra digits --psm 7",//eng+deu+fra
                        RedirectStandardError = true,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true,
                        UseShellExecute = false
                    };

                    using (var ps = Process.Start(info))
                    {
                        ps.WaitForExit();

                        var exitCode = ps.ExitCode;

                        if (exitCode == 0)
                        {
                            output = File.ReadAllText(@"X:\bidsLine" + index + ".txt");
                        }
                        else
                        {
                            var stderr = ps.StandardError.ReadToEnd();
                            throw new InvalidOperationException(stderr);
                        }
                    }
                }
                finally
                {

                }
            }

            return output;
        }

        public string GetTextAsks(int index, params Stream[] images)
        {
            var output = string.Empty;

            if (images.Any())
            {
                try
                {
                    var info = new ProcessStartInfo
                    {
                        FileName = _tesseractExePath,
                        Arguments = @"X:\asksLine" + index + @".jpg  X:\asksLine" + index + " -l eng+deu+fra digits --psm 7",
                        RedirectStandardError = true,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true,
                        UseShellExecute = false
                    };

                    using (var ps = Process.Start(info))
                    {
                        ps.WaitForExit();

                        var exitCode = ps.ExitCode;

                        if (exitCode == 0)
                        {
                            output = File.ReadAllText(@"X:\asksLine" + index + ".txt");
                        }
                        else
                        {
                            var stderr = ps.StandardError.ReadToEnd();
                            throw new InvalidOperationException(stderr);
                        }
                    }
                }
                finally
                {

                }
            }

            return output;
        }

        public string GetTextMstClk(params Stream[] images)
        {
            var output = string.Empty;

            if (images.Any())
            {
                try
                {
                    var info = new ProcessStartInfo
                    {
                        FileName = _tesseractExePath,
                        Arguments = @"X:\masterClkTest_blk_white.jpg X:\masterClkTest -l eng+deu+fra --psm 7",
                        RedirectStandardError = true,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true,
                        UseShellExecute = false
                    };

                    using (var ps = Process.Start(info))
                    {
                        ps.WaitForExit();

                        var exitCode = ps.ExitCode;

                        if (exitCode == 0)
                        {
                            output = File.ReadAllText(@"X:\masterClkTest.txt");
                        }
                        else
                        {
                            var stderr = ps.StandardError.ReadToEnd();
                            throw new InvalidOperationException(stderr);
                        }
                    }
                }
                finally
                {

                }
            }

            return output;
        }

        public string GetTextDow(params Stream[] images)
        {
            var output = string.Empty;

            if (images.Any())
            {
                try
                {
                    var info = new ProcessStartInfo
                    {
                        FileName = _tesseractExePath,
                        Arguments = @"X:\dowTest_blk_white.jpg X:\dowTest -l eng+deu+fra  --psm 7",
                        RedirectStandardError = true,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true,
                        UseShellExecute = false
                    };

                    using (var ps = Process.Start(info))
                    {
                        ps.WaitForExit();

                        var exitCode = ps.ExitCode;

                        if (exitCode == 0)
                        {
                            output = File.ReadAllText(@"X:\dowTest.txt");
                        }
                        else
                        {
                            var stderr = ps.StandardError.ReadToEnd();
                            throw new InvalidOperationException(stderr);
                        }
                    }
                }
                finally
                {

                }
            }
            return output;
        }

        private static void WriteInputFiles(Stream[] inputStreams, string tempPath, string tempInputFile)
        {
            // If is only one image file, than use the image file as input file.
            using (var tempStream = File.OpenWrite(tempInputFile))
            {
                CopyStream(inputStreams.First(), tempStream);
            }
        }

        private static void CopyStream(Stream input, Stream output)
        {
            if (input.CanSeek)
                input.Seek(0, SeekOrigin.Begin);
            input.CopyTo(output);
            input.Close();
        }

        private static string NewTempFileName(string tempPath)
        {
            return Path.Combine(tempPath, Guid.NewGuid().ToString());
        }
    }
}
