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
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing.Imaging;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;
using System.Threading;
using System.ComponentModel;
using System.Globalization;
using OpenCvSharp;
using System.Net;
using System.Net.Http;
using System.Net.NetworkInformation;

namespace Data_Scraper
{
    public class Program
    {
        public static RamDisk ramDisk = new RamDisk();
        public static readonly HttpClient client = new HttpClient();

        public static int half_instances = 16;
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            Data_Scraper.algoGui.algoGui_array[0] = new algoGui();
            Data_Scraper.algoGui.algoGui_array[1] = new algoGui();
            Data_Scraper.algoGui.algoGui_array[2] = new algoGui();
            Data_Scraper.algoGui.algoGui_array[3] = new algoGui();
            Data_Scraper.algoGui.algoGui_array[4] = new algoGui();
            Data_Scraper.algoGui.algoGui_array[5] = new algoGui();

            Data_Scraper.algoGui.bidsThread.Start();
            Data_Scraper.algoGui.asksThread.Start();
            Data_Scraper.algoGui.lastsThread.Start();
            Data_Scraper.algoGui.dowThread.Start();
            Data_Scraper.algoGui.etradeClkThread.Start();
            Data_Scraper.algoGui.mainWinThread.Start();

            ramDisk.createRamDisk();
        }

        public enum enmScreenCaptureMode
        {
            Screen,
            Window
        }

        public class Level2DataAnalyzer
        {
            public Bitmap img;
            public Bitmap img2;
            public Bitmap grayscale_img;
            public Bitmap grayscale_img2;
            public Bitmap lastTradeImg;
            public Bitmap dowImg;
            public Bitmap masterClkImg;
            Bitmap[] lastTradesSplitArray = new Bitmap[37];
            Bitmap[] bidsSplitArray = new Bitmap[16];
            Bitmap[] asksSplitArray = new Bitmap[16];
            public TesseractService dowTessService = new TesseractService(@"C:\Program Files\Tesseract-OCR", "eng", @"C:\Program Files\Tesseract-OCR\tessdata");
            public TesseractService clkTessService = new TesseractService(@"C:\Program Files\Tesseract-OCR", "eng", @"C:\Program Files\Tesseract-OCR\tessdata");
            public TesseractService bids_asksTessService = new TesseractService(@"C:\Program Files\Tesseract-OCR", "eng", @"C:\Program Files\Tesseract-OCR\tessdata");
            public TesseractService lastTradesTessService = new TesseractService(@"C:\Program Files\Tesseract-OCR", "eng", @"C:\Program Files\Tesseract-OCR\tessdata");

            public Bitmap blackWhiteFilter(Bitmap image, bool dow)
            {
                using (image)
                {
                    Bitmap filteredImage = new Bitmap(image.Width, image.Height);
                    BitmapData srcData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
                    BitmapData dstData = filteredImage.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);

                    unsafe
                    {
                        byte* dstPointer = (byte*)dstData.Scan0;
                        byte* srcPointer = (byte*)srcData.Scan0;

                        if (dow == true)
                        {
                            for (Int32 y = 0; y < image.Height; y++)
                            {
                                for (Int32 x = 0; x < image.Width; x++)
                                {
                                    Color c = Color.FromArgb(srcPointer[2], srcPointer[1], srcPointer[0]);

                                    if ((c.R >= 50 && c.R <= 255) || (c.G >= 100 && c.G <= 255))
                                    {
                                        if (x < 40)
                                        {
                                            //filteredImage.SetPixel(x, y, Color.White);
                                            dstPointer[0] = 255;
                                            dstPointer[1] = 255;
                                            dstPointer[2] = 255;
                                            dstPointer[3] = 255;

                                            srcPointer += 4;
                                            dstPointer += 4;
                                            continue;
                                        }
                                        //filteredImage.SetPixel(x, y, Color.Black);
                                        dstPointer[0] = 0;
                                        dstPointer[1] = 0;
                                        dstPointer[2] = 0;
                                        dstPointer[3] = 0;
                                    }
                                    else
                                    {
                                        //filteredImage.SetPixel(x, y, Color.White);
                                        dstPointer[0] = 255;
                                        dstPointer[1] = 255;
                                        dstPointer[2] = 255;
                                        dstPointer[3] = 255;
                                    }
                                    srcPointer += 4;
                                    dstPointer += 4;
                                }
                            }
                            return filteredImage;
                        }
                        else
                        {
                            for (Int32 y = 0; y < image.Height; y++)
                            {
                                for (Int32 x = 0; x < image.Width; x++)
                                {
                                    Color c = Color.FromArgb(srcPointer[2], srcPointer[1], srcPointer[0]);

                                    if (c.R <= 205 && c.G <= 160 && c.B <= 205)
                                    {
                                        if (x < image.Width)
                                        {
                                            if (x > 950 || x < 20)
                                            {
                                                //filteredImage.SetPixel(x, y, Color.White);
                                                dstPointer[0] = 255;
                                                dstPointer[1] = 255;
                                                dstPointer[2] = 255;
                                                dstPointer[3] = 255;
                                            }
                                            if (y < 15)
                                            {
                                                //filteredImage.SetPixel(x, y, Color.White);
                                                dstPointer[0] = 255;
                                                dstPointer[1] = 255;
                                                dstPointer[2] = 255;
                                                dstPointer[3] = 255;
                                            }
                                            srcPointer += 4;
                                            dstPointer += 4;
                                            continue;
                                        }
                                        //filteredImage.SetPixel(x, y, Color.Black);
                                        dstPointer[0] = 0;
                                        dstPointer[1] = 0;
                                        dstPointer[2] = 0;
                                        dstPointer[3] = 0;

                                        srcPointer += 4;
                                        dstPointer += 4;
                                    }
                                    else
                                    {
                                        //filteredImage.SetPixel(x, y, Color.White);
                                        dstPointer[0] = 255;
                                        dstPointer[1] = 255;
                                        dstPointer[2] = 255;
                                        dstPointer[3] = 255;

                                        srcPointer += 4;
                                        dstPointer += 4;
                                    }
                                }
                            }
                            filteredImage.UnlockBits(dstData);
                            image.UnlockBits(srcData);
                            return filteredImage;
                        }
                    }
                }
            }

            public Bitmap blackWhiteFilterLastTrades(Bitmap image)
            {
                Bitmap filteredLine = new Bitmap(image.Width + 400, image.Height);
                using (Graphics gfx = Graphics.FromImage(filteredLine))
                using (SolidBrush brush = new SolidBrush(Color.White))
                {
                    gfx.FillRectangle(brush, 400, 0, 440, image.Height);
                }

                for (Int32 y = 0; y < image.Height; y++)
                    for (Int32 x = 0; x < image.Width; x++)
                    {
                        int temp = x;
                        Color c = image.GetPixel(x, y);

                        if (x > 400)
                        {
                            temp = x + 400;
                        }
                        if ((c.R >= 50 && c.R <= 255) || (c.G >= 100 && c.G <= 255))
                        {
                            if (x < image.Width)
                            {
                                if (x > filteredLine.Width - 40 || x < 30)
                                {
                                    filteredLine.SetPixel(temp, y, Color.White);
                                }
                                continue;
                            }
                            filteredLine.SetPixel(temp, y, Color.Black);
                        }
                        else
                        {
                            filteredLine.SetPixel(temp, y, Color.White);
                        }
                    }
                return filteredLine;
            }

            public Bitmap blackWhiteFilterMstClk(Bitmap image)
            {
                Bitmap filteredLine = new Bitmap(image.Width, image.Height);
                for (Int32 y = 0; y < filteredLine.Height; y++)
                    for (Int32 x = 0; x < filteredLine.Width; x++)
                    {
                        Color c = image.GetPixel(x, y);

                        if (c.A == 255 && (c.R >= 120 && c.R <= 255) && (c.G >= 120 && c.G <= 255))
                        {
                            if (x > filteredLine.Width - 18 || x < 10 || y < 15)
                            {
                                filteredLine.SetPixel(x, y, Color.White);
                                continue;
                            }
                            filteredLine.SetPixel(x, y, Color.Black);
                        }
                        else
                        {
                            filteredLine.SetPixel(x, y, Color.White);
                        }
                    }
                return filteredLine;
            }

            public void masterClkRetriever()
            {
                enmScreenCaptureMode scNum = enmScreenCaptureMode.Window;
                ScreenCapturer etradeWincap = new ScreenCapturer();

                masterClkImg = etradeWincap.CaptureMstClk(scNum);
                masterClkImg = etradeWincap.ResizeImage(masterClkImg, masterClkImg.Width * 6, masterClkImg.Height * 6);
                //masterClkImg = etradeWincap.Sharpen(masterClkImg);
                masterClkImg.Save("X://masterClkTest.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
                masterClkImg = blackWhiteFilterMstClk(masterClkImg);
                masterClkImg.Save("X://masterClkTest_blk_white.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
                Mat gs_img = Cv2.ImRead("X://masterClkTest_blk_white.jpg", ImreadModes.Grayscale);
                Mat filtered = new Mat();
                OpenCvSharp.Size size = new OpenCvSharp.Size();
                size.Height = 3; //9
                size.Width = 3; //9
                Cv2.MedianBlur(gs_img, filtered, 7); //last param 9
                Cv2.GaussianBlur(filtered, filtered, size, 0, 0, BorderTypes.Default);
                Cv2.Erode(filtered, filtered, new Mat(), null, 1, BorderTypes.Constant, null);
                using (var ms = filtered.ToMemoryStream())
                {
                    masterClkImg = (Bitmap)etradeWincap.FixedSize(Image.FromStream(ms), masterClkImg.Width, masterClkImg.Height);
                    masterClkImg.Save(("X://masterClkTest_blk_white.jpg"), System.Drawing.Imaging.ImageFormat.Jpeg);
                }
            }

            public void masterClkExtractor(etradeClkObj etradeClk)
            {
                Thread.CurrentThread.Priority = ThreadPriority.Highest;
                String dataStr = "";
                etradeClk.etradeMstClkHour = 0;
                etradeClk.etradeMstClkMinute = 0;
                etradeClk.etradeMstClkSecond = 0;
                etradeClk.etradeMstClkAMPM = "";
                string text = "";
                string ipaddress = "";

                if (algoGui.algoGui_array[5].eblCluster.Checked == true)
                {
                    ipaddress = algoGui.ipaddress3;
                    try
                    {
                        var request = new HttpRequestMessage(HttpMethod.Post, "http://" + ipaddress + "/uploader");
                        var content = new MultipartFormDataContent();

                        byte[] byteArray = File.ReadAllBytes(@"X:\masterClkTest_blk_white.jpg");
                        content.Add(new ByteArrayContent(byteArray), "file", "masterClkTest_blk_white.jpg");
                        request.Content = content;

                        var task1 = Task.Run(() => client.SendAsync(request));
                        task1.Wait();
                        var response = task1.Result;
                        response.EnsureSuccessStatusCode();

                        var execRequest = new HttpRequestMessage(HttpMethod.Get, "http://" + ipaddress + "/execute?program=Tesseract&imagefile=masterClkTest_blk_white"
                            + ".jpg&textfileout=masterClkTest " + "&languageparam=-l&languages=eng%2Bdeu%2Bfra");

                        var task2 = Task.Run(() => client.SendAsync(execRequest));
                        task2.Wait();
                        var execResponse = task2.Result;
                        execResponse.EnsureSuccessStatusCode();

                        var task3 = Task.Run(() => execResponse.Content.ReadAsStringAsync());
                        task3.Wait();
                        text = task3.Result;
                        text = getBetween(text, "{\"execute\":\"", "\\n\"");
                        text += "\n";
                    }
                    catch (Exception ex)
                    {
                    }
                }
                else
                {
                    var stream = File.OpenRead(@"X:\masterClkTest_blk_white.jpg");
                    text = clkTessService.GetTextMstClk(stream);
                }

                int hour_minute_second = 0;

                foreach (Char data in text)
                {
                    if (!Char.IsWhiteSpace(data))
                    {
                        if (Char.Equals(':', data) && hour_minute_second == 0)
                        {
                            etradeClk.etradeMstClkHour = Convert.ToInt32(dataStr, CultureInfo.InvariantCulture.NumberFormat);
                            if (etradeClk.etradeMstClkHour == 40 || etradeClk.etradeMstClkHour == 41 || etradeClk.etradeMstClkHour == 42)
                            {
                                etradeClk.etradeMstClkHour -= 30;
                            }
                            hour_minute_second++;
                            dataStr = "";
                        }
                        else if (Char.Equals(':', data) && hour_minute_second == 1)
                        {
                            etradeClk.etradeMstClkMinute = Convert.ToInt32(dataStr, CultureInfo.InvariantCulture.NumberFormat);
                            hour_minute_second++;
                            dataStr = "";
                        }
                        else if (Char.Equals('|', data) || Char.Equals('E', data))
                        {
                            /* Invalid character, do nothing. */
                        }
                        else
                        {
                            if (data.Equals('N'))
                            {
                                dataStr = dataStr + 'M';
                            }
                            if (data.Equals('R'))
                            {
                                dataStr = dataStr + 'M';
                            }
                            else if (data.Equals('W'))
                            {
                                //do nothing
                            }
                            else
                            {
                                dataStr = dataStr + data;
                            }
                        }
                    }
                    else if (Char.IsWhiteSpace(data) && hour_minute_second == 2)
                    {
                        etradeClk.etradeMstClkSecond = Convert.ToInt32(dataStr, CultureInfo.InvariantCulture.NumberFormat);
                        hour_minute_second++;
                        dataStr = "";
                    }
                    else if (Char.IsWhiteSpace(data) && dataStr != "" && hour_minute_second == 3)
                    {
                        if (dataStr.Equals("AMN"))
                        {
                            dataStr = "AM";
                        }
                        else if (dataStr.Equals("A"))
                        {
                            dataStr = "AM";
                        }
                        else if (dataStr.Equals("AME"))
                        {
                            dataStr = "PM";
                        }
                        else if (dataStr.Equals("PME"))
                        {
                            dataStr = "PM";
                        }
                        etradeClk.etradeMstClkAMPM = dataStr;
                    }
                }
            }

            public void dowDataRetriever()
            {
                enmScreenCaptureMode scNum = enmScreenCaptureMode.Window;
                ScreenCapturer etradeWincap = new ScreenCapturer();

                dowImg = etradeWincap.CaptureDow(scNum);
                dowDataSplitter(dowImg);
                dowImg = etradeWincap.ResizeImage(dowImg, dowImg.Width * 8, dowImg.Height * 8);
                dowImg = etradeWincap.Sharpen(dowImg);
                dowImg.Save("X://dowTest.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
                dowImg = blackWhiteFilter(dowImg, true);
                dowImg.Save("X://dowTest_blk_white.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
                Mat gs_img = Cv2.ImRead("X://dowTest_blk_white.jpg", ImreadModes.Grayscale);
                Mat filtered = new Mat();
                Cv2.MedianBlur(gs_img, filtered, 9);
                using (var ms = filtered.ToMemoryStream())
                {
                    dowImg = (Bitmap)etradeWincap.FixedSize(Image.FromStream(ms), dowImg.Width, dowImg.Height + 400);
                    dowImg.Save(("X://dowTest_blk_white.jpg"), System.Drawing.Imaging.ImageFormat.Jpeg);
                }
            }

            public void dowDataExtractor(dowObj dow)
            {
                Thread.CurrentThread.Priority = ThreadPriority.Highest;
                String dataStr = "";
                dow.dowAvg = 0;
                dow.dowAvgSize = 0;
                string text = "";
                /*
                string ipaddress = "172.28.254.148:8080";

                if (algoGui.algoGui_array[5].eblCluster.Checked == true)
                {
                    var request = new HttpRequestMessage(HttpMethod.Post, "http://" + ipaddress + "/uploader");
                    var content = new MultipartFormDataContent();

                    byte[] byteArray = File.ReadAllBytes(@"X:\dowTest_blk_white.jpg");
                    content.Add(new ByteArrayContent(byteArray), "file", "dowTest_blk_white.jpg");
                    request.Content = content;

                    var task1 = Task.Run(() => client.SendAsync(request));
                    task1.Wait();
                    var response = task1.Result;
                    response.EnsureSuccessStatusCode();

                    var execRequest = new HttpRequestMessage(HttpMethod.Get, "http://" + ipaddress + "/execute?program=tesseract&imagefile=dowTest_blk_white.jpg&textfileout=" +
                        "dowTest&languageparam=-l&languages=eng%2Bdeu%2Bfra");

                    var task2 = Task.Run(() => client.SendAsync(execRequest));
                    task2.Wait();
                    var execResponse = task2.Result;
                    execResponse.EnsureSuccessStatusCode();

                    var task3 = Task.Run(() => execResponse.Content.ReadAsStringAsync());
                    task3.Wait();
                    text = task3.Result;
                    text = getBetween(text, "{\"execute\":\"", "\\n000\"");
                    text += "\n";
                }
                else
                {
                */
                var stream = File.OpenRead(@"X:\dowTest_blk_white.jpg");
                text = dowTessService.GetTextDow(stream);
                //}

                bool sizePriceFlag = false;
                foreach (Char data in text)
                {
                    if (!Char.IsWhiteSpace(data))
                    {
                        dataStr = dataStr + data;
                    }
                    else if (Char.IsWhiteSpace(data) && dataStr != "")
                    {
                        if (sizePriceFlag == true)
                        {
                            dow.dowAvgSize = (int)Convert.ToInt32(dataStr, CultureInfo.InvariantCulture.NumberFormat);
                            sizePriceFlag = false;
                        }

                        else
                        {
                            sizePriceFlag = true;
                            dow.dowAvg = Convert.ToDouble(dataStr, CultureInfo.InvariantCulture.NumberFormat);
                            if (dow.dowAvg > Convert.ToDouble(1000000))
                            {
                                dow.dowAvg = dow.dowAvg / 100;
                            }
                        }
                        dataStr = "";
                    }
                }
            }

            public void lastTradesDataRetriever()
            {
                enmScreenCaptureMode scNum = enmScreenCaptureMode.Window;
                ScreenCapturer etradeWincap = new ScreenCapturer();

                lastTradeImg = etradeWincap.CaptureLast(scNum);
                lastTradeImg.Save("X://lastTradesTest.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
                lastTradesDataSplitter(lastTradeImg);

                Parallel.For(0, 37, (i, state) =>
                {
                    lastTradesSplitArray[i] = etradeWincap.ResizeImage(lastTradesSplitArray[i], lastTradesSplitArray[i].Width * 4, lastTradesSplitArray[i].Height * 4);
                    //lastTradesSplitArray[i] = etradeWincap.Sharpen(lastTradesSplitArray[i]);
                    lastTradesSplitArray[i].Save(("X://lastTradeLine" + i + ".jpg"), System.Drawing.Imaging.ImageFormat.Jpeg);
                    lastTradesSplitArray[i] = blackWhiteFilterLastTrades(lastTradesSplitArray[i]);
                    lastTradesSplitArray[i].Save(("X://lastTradeLine" + i + "blk_white.jpg"), System.Drawing.Imaging.ImageFormat.Jpeg);
                    Mat gs_img = Cv2.ImRead("X://lastTradeLine" + i + "blk_white.jpg", ImreadModes.Grayscale);
                    Mat filtered = new Mat();
                    OpenCvSharp.Size size = new OpenCvSharp.Size();
                    size.Height = 3; //9
                    size.Width = 3; //9
                    Cv2.MedianBlur(gs_img, filtered, 3);
                    Cv2.GaussianBlur(filtered, filtered, size, 0, 0, BorderTypes.Default);
                    using (var ms = filtered.ToMemoryStream())
                    {
                        lastTradesSplitArray[i] = (Bitmap)etradeWincap.FixedSize(Image.FromStream(ms), lastTradesSplitArray[i].Width, lastTradesSplitArray[i].Height + 200);
                        lastTradesSplitArray[i].Save(("X://lastTradeLine" + i + "blk_white.jpg"), System.Drawing.Imaging.ImageFormat.Jpeg);
                    }
                });
            }

            public void lastTradesDataExtractor(lastTradesObj lastTrades)
            {
                Thread.CurrentThread.Priority = ThreadPriority.Highest;

                Parallel.For(0, 37, new ParallelOptions { MaxDegreeOfParallelism = 37 }, (i, state) =>
                {
                    String dataStr = "";
                    lastTrades.lastTradeSizeArray[i] = 0;
                    lastTrades.lastTradePriceArray[i] = 0;
                    var stream = File.OpenRead(@"X:\lastTradeLine" + i + "blk_white.jpg");
                    string text = "";
                    text = lastTradesTessService.GetTextLastTrades(i, stream);

                    bool sizePriceFlag = false;
                    foreach (Char data in text)
                    {
                        if (Char.Equals(':', data))
                        {
                            dataStr = "";
                            break;
                        }
                        if (!Char.IsWhiteSpace(data))
                        {
                            dataStr = dataStr + data;
                        }
                        else if (Char.IsWhiteSpace(data) && dataStr != "")
                        {
                            if (sizePriceFlag == true)
                            {
                                lastTrades.lastTradeSizeArray[i] = (int)Convert.ToInt32(dataStr, CultureInfo.InvariantCulture.NumberFormat);
                                sizePriceFlag = false;
                            }
                            else
                            {
                                float temp = 0.0f;
                                sizePriceFlag = true;
                                temp = (float)Convert.ToDouble(dataStr, CultureInfo.InvariantCulture.NumberFormat);
                                lastTrades.lastTradePriceArray[i] = temp;
                            }
                            dataStr = "";
                        }
                        else
                        {
                            dataStr = "";
                        }
                    }
                });
            }

            public void bidsDataSplitter(Bitmap image)
            {
                var splitInto = 16;
                using (var originalImage = new Bitmap(image))
                {
                    for (int i = 0; i < splitInto; i++)
                    {
                        var rect = new Rectangle(0, originalImage.Height / splitInto * i, originalImage.Width, originalImage.Height / splitInto);
                        using (var clonedImage = originalImage.Clone(rect, originalImage.PixelFormat))
                        {
                            bidsSplitArray[i] = originalImage.Clone(rect, originalImage.PixelFormat);
                        }
                    }
                }
            }

            public void asksDataSplitter(Bitmap image)
            {
                var splitInto = 16;
                using (var originalImage = new Bitmap(image))
                {
                    for (int i = 0; i < splitInto; i++)
                    {
                        var rect = new Rectangle(0, originalImage.Height / splitInto * i, originalImage.Width, originalImage.Height / splitInto);
                        using (var clonedImage = originalImage.Clone(rect, originalImage.PixelFormat))
                        {
                            asksSplitArray[i] = originalImage.Clone(rect, originalImage.PixelFormat);
                        }
                    }
                }
            }

            public void lastTradesDataSplitter(Bitmap image)
            {
                var splitInto = 37;
                using (var originalImage = new Bitmap(image))
                {
                    for (int i = 0; i < splitInto; i++)
                    {
                        var rect = new Rectangle(0, originalImage.Height / splitInto * i, originalImage.Width, originalImage.Height / splitInto);
                        using (var clonedImage = originalImage.Clone(rect, originalImage.PixelFormat))
                        {
                            lastTradesSplitArray[i] = originalImage.Clone(rect, originalImage.PixelFormat);
                        }
                    }
                }
            }

            public void dowDataSplitter(Bitmap image)
            {
                var splitInto = 3;
                using (var originalImage = new Bitmap(image))
                {
                    for (int i = 0; i < splitInto; i++)
                    {
                        var rect = new Rectangle(0, originalImage.Height / splitInto * i, originalImage.Width, originalImage.Height / splitInto);
                        using (var clonedImage = originalImage.Clone(rect, originalImage.PixelFormat))
                        {
                            if (i == 0)
                            {
                                dowImg = originalImage.Clone(rect, originalImage.PixelFormat);
                            }
                        }
                    }
                }
            }

            public void levelTwoDataRetriever(bool bidsOrAsks)
            {
                if (bidsOrAsks == true)
                {
                    enmScreenCaptureMode scNum = enmScreenCaptureMode.Window;
                    ScreenCapturer etradeWinCap = new ScreenCapturer();
                    img = etradeWinCap.Capture(scNum, bidsOrAsks);
                    img = etradeWinCap.ResizeImage(img, 480, 880); //(520, 920)
                    //img = etradeWinCap.Dilate(img);
                    img.Save("X://test.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
                    grayscale_img = blackWhiteFilter(img, false);
                    grayscale_img.Save("X://blk_white_test.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
                    bidsDataSplitter(grayscale_img);
                    Parallel.For(0, half_instances, new ParallelOptions { MaxDegreeOfParallelism = half_instances }, (i, state) =>
                    {
                        //bidsSplitArray[i] = etradeWinCap.ResizeImage(bidsSplitArray[i], bidsSplitArray[i].Width * 2, bidsSplitArray[i].Height * 2);
                        //bidsSplitArray[i] = etradeWinCap.Sharpen(bidsSplitArray[i]);
                        bidsSplitArray[i].Save(("X://bidsLine" + i + ".jpg"), System.Drawing.Imaging.ImageFormat.Jpeg);
                        Mat gs_img = Cv2.ImRead("X://bidsLine" + i + ".jpg", ImreadModes.Grayscale);
                        Mat filtered = new Mat();
                        OpenCvSharp.Size size = new OpenCvSharp.Size();
                        size.Height = 3; //9
                        size.Width = 3; //9
                        Cv2.MedianBlur(gs_img, filtered, 3); //last param 9
                        Cv2.GaussianBlur(filtered, filtered, size, 0, 0, BorderTypes.Default);
                        using (var ms = filtered.ToMemoryStream())
                        {
                            bidsSplitArray[i] = (Bitmap)etradeWinCap.FixedSize(Image.FromStream(ms), bidsSplitArray[i].Width, bidsSplitArray[i].Height + 50);
                            //bidsSplitArray[i] = etradeWinCap.Sharpen(bidsSplitArray[i], bidsSplitArray[i].Width, bidsSplitArray[i].Height);
                            bidsSplitArray[i].Save(("X://bidsLine" + i + ".jpg"), System.Drawing.Imaging.ImageFormat.Jpeg);
                        }
                    });
                }
                else
                {
                    enmScreenCaptureMode scNum2 = enmScreenCaptureMode.Window;
                    ScreenCapturer etradeWinCap2 = new ScreenCapturer();
                    img2 = etradeWinCap2.Capture(scNum2, bidsOrAsks);
                    img2 = etradeWinCap2.ResizeImage(img2, 480, 880); //(520, 920)
                    //img2 = etradeWinCap2.Sharpen(img2);
                    img2.Save("X://test2.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
                    grayscale_img2 = blackWhiteFilter(img2, false);
                    grayscale_img2.Save("X://blk_white_test2.jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
                    asksDataSplitter(grayscale_img2);
                    Parallel.For(0, half_instances, new ParallelOptions { MaxDegreeOfParallelism = half_instances }, (i, state) =>
                    {
                        //asksSplitArray[i] = etradeWinCap2.ResizeImage(asksSplitArray[i], asksSplitArray[i].Width * 2, asksSplitArray[i].Height * 2);
                        //asksSplitArray[i] = etradeWinCap2.Sharpen(asksSplitArray[i]);
                        asksSplitArray[i].Save(("X://asksLine" + i + ".jpg"), System.Drawing.Imaging.ImageFormat.Jpeg);
                        Mat gs_img = Cv2.ImRead("X://asksLine" + i + ".jpg", ImreadModes.Grayscale);
                        Mat filtered = new Mat();
                        OpenCvSharp.Size size = new OpenCvSharp.Size();
                        size.Height = 3; //9
                        size.Width = 3; //9
                        Cv2.MedianBlur(gs_img, filtered, 3); //last param 9
                        Cv2.GaussianBlur(filtered, filtered, size, 0, 0, BorderTypes.Default);
                        using (var ms = filtered.ToMemoryStream())
                        {
                            asksSplitArray[i] = (Bitmap)etradeWinCap2.FixedSize(Image.FromStream(ms), asksSplitArray[i].Width, asksSplitArray[i].Height + 50);
                            //asksSplitArray[i] = etradeWinCap2.Sharpen(asksSplitArray[i], asksSplitArray[i].Width, asksSplitArray[i].Height);
                            asksSplitArray[i].Save(("X://asksLine" + i + ".jpg"), System.Drawing.Imaging.ImageFormat.Jpeg);
                        }
                    });
                }
            }

            public void levelTwoDataExtractor(bidsObj bids, asksObj asks, bool bidsOrAsks)
            {
                Thread.CurrentThread.Priority = ThreadPriority.Highest;
                if (bidsOrAsks == true)
                {
                    Parallel.For(0, half_instances, new ParallelOptions { MaxDegreeOfParallelism = half_instances }, (j, state) =>
                    {
                        algoGui.errorLineFlagRunOnce[j] = false;
                        string ipaddress = "";
                        int j_limit1 = 0;
                        int j_limit2 = 0;
                        int j_limit3 = 0;
                        bool supClusterActive = false;
                        bids.bidsArray[j] = 0;
                        bids.bidsSizeArray[j] = 0;
                        string text = "";
                        string dataStr = "";

                        if (algoGui.node1Online == true && algoGui.node2Online == true && algoGui.node3Online == true)
                        {
                            j_limit1 = 6;
                            j_limit2 = 12;
                            j_limit3 = 16;
                        }
                        else if (algoGui.node1Online == true || algoGui.node2Online == true)
                        {
                            if (algoGui.node1Online == true)
                            {
                                j_limit1 = 8;
                                j_limit2 = 8;
                            }
                            else if (algoGui.node2Online == true)
                            {
                                j_limit1 = 0;
                                j_limit2 = half_instances;
                            }
                        }
                        if (j >= 0 && j < j_limit1 && algoGui.algoGui_array[5].eblCluster.Checked == true && algoGui.node1Online == true)
                        {
                            ipaddress = algoGui.ipaddress1;
                            supClusterActive = true;
                        }
                        if (j >= j_limit1 && j < j_limit2 && algoGui.algoGui_array[5].eblCluster.Checked == true && algoGui.node2Online == true)
                        {
                            ipaddress = algoGui.ipaddress2;
                            supClusterActive = true;
                        }
                        if (j >= j_limit2 && j < j_limit3 && algoGui.algoGui_array[5].eblCluster.Checked == true && algoGui.node3Online == true)
                        {
                            ipaddress = algoGui.ipaddress3;
                            supClusterActive = true;
                        }
                        if (j >= 0 && j < j_limit3 && algoGui.algoGui_array[5].eblCluster.Checked == true && supClusterActive == true)
                        {
                            try
                            {
                                var request = new HttpRequestMessage(HttpMethod.Post, "http://" + ipaddress + "/uploader");
                                var content = new MultipartFormDataContent();

                                byte[] byteArray = File.ReadAllBytes(@"X:\bidsLine" + j + ".jpg");
                                content.Add(new ByteArrayContent(byteArray), "file", "bidsLine" + j + ".jpg");
                                request.Content = content;

                                var task1 = Task.Run(() => client.SendAsync(request));
                                task1.Wait();
                                var response = task1.Result;
                                response.EnsureSuccessStatusCode();

                                var execRequest = new HttpRequestMessage(HttpMethod.Get, "http://" + ipaddress + "/execute?program=Tesseract&imagefile=bidsLine"
                                    + j + ".jpg&textfileout=bidsLine" + j + "&languageparam=-l&languages=eng_fast&psm=--psm&numbervalue=7&oem=--oem&oemMode=3&typeofnumber=digits");//languages=eng%2Bdeu%2Bfra

                                var task2 = Task.Run(() => client.SendAsync(execRequest));
                                task2.Wait();
                                var execResponse = task2.Result;
                                execResponse.EnsureSuccessStatusCode();

                                var task3 = Task.Run(() => execResponse.Content.ReadAsStringAsync());
                                task3.Wait();
                                text = task3.Result;
                                text = getBetween(text, "{\"execute\":\"", "\\n\"");
                                text += "\n";
                            }
                            catch (Exception ex)
                            {
                                if (ipaddress == algoGui.ipaddress1)
                                {
                                    algoGui.node1Online = false;
                                    algoGui.algoGui_array[5].node1Status.Text = "Node 1 Offline";
                                }
                                else if (ipaddress == algoGui.ipaddress2)
                                {
                                    algoGui.node2Online = false;
                                    algoGui.algoGui_array[5].node2Status.Text = "Node 2 Offline";
                                }
                                else if (ipaddress == algoGui.ipaddress3)
                                {
                                    algoGui.node3Online = false;
                                    algoGui.algoGui_array[5].node3Status.Text = "Node 3 Offline";
                                }
                            }
                        }
                        else
                        {
                            var stream = File.OpenRead(@"X:\bidsLine" + j + ".jpg");
                            text = bids_asksTessService.GetTextBids(j, stream);
                        }
error_check:
                        bool sizePriceFlag = false;
                        foreach (Char data in text)
                        {
                            if (!Char.IsWhiteSpace(data))
                            {
                                if(data.Equals(':'))
								{
									dataStr = dataStr + '.';
								}
								else
								{
									dataStr = dataStr + data;
								}
                            }
                            else if (Char.IsWhiteSpace(data) && dataStr != "")
                            {
                                if (sizePriceFlag == true)
                                {
                                    if (dataStr.EndsWith("K"))
                                    {
                                        dataStr = dataStr.Remove(dataStr.Length - 1, 1);
                                        dataStr = dataStr + '0' + '0' + '0';
                                    }
                                    try
                                    {
                                        bids.bidsSizeArray[j] = Convert.ToInt32(dataStr, CultureInfo.InvariantCulture.NumberFormat);
                                    }
                                    catch (Exception ex)
                                    {
                                        StreamWriter output = File.AppendText(@"X:\error_list_bids_sizes.txt");
                                        output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString());
                                        //bids.bidsSizeArray[j] = 343;
                                        algoGui.errorLineFlag[j] = true;
                                        var stream = File.OpenRead(@"X:\bidsLine" + j + ".jpg");
                                        text = bids_asksTessService.GetTextBids(j, stream);
                                        output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString() + "\n");
                                        output.Close();
                                        if (algoGui.errorLineFlag[j] == true && algoGui.errorLineFlagRunOnce[j] == false)
                                        {
                                            algoGui.errorLineFlag[j] = false;
                                            algoGui.errorLineFlagRunOnce[j] = true;
                                            goto error_check;
                                        }
                                        //algoGui.critErr = true;
                                    }
                                    sizePriceFlag = false;
                                }
                                else
                                {
                                    sizePriceFlag = true;
                                    try
                                    {
                                        bids.bidsArray[j] = (float)Convert.ToDouble(dataStr, CultureInfo.InvariantCulture.NumberFormat);
                                    }
                                    catch(Exception ex)
                                    {
                                        StreamWriter output = File.AppendText(@"X:\error_list_bids_prices.txt");
                                        output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString());
                                        //bids.bidsArray[j] = 0f;
                                        algoGui.errorLineFlag[j] = true;
                                        var stream = File.OpenRead(@"X:\bidsLine" + j + ".jpg");
                                        text = bids_asksTessService.GetTextBids(j, stream);
                                        output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString() + "\n");
                                        output.Close();
                                        if (algoGui.errorLineFlag[j] == true && algoGui.errorLineFlagRunOnce[j] == false)
                                        {
                                            algoGui.errorLineFlag[j] = false;
                                            algoGui.errorLineFlagRunOnce[j] = true;
                                            goto error_check;
                                        }
                                        //algoGui.critErr = true;
                                    }
                                }
                                dataStr = "";
                            }
                            else
                            {
                                StreamWriter output = File.AppendText(@"X:\error_list_entire_line_bids.txt");
                                output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString());
                                //algoGui.critErr = true;
                                algoGui.errorLineFlag[j] = true;
                                var stream = File.OpenRead(@"X:\bidsLine" + j + ".jpg");
                                text = bids_asksTessService.GetTextBids(j, stream);
                                output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString() + "\n");
                                output.Close();
                                if (algoGui.errorLineFlag[j] == true && algoGui.errorLineFlagRunOnce[j] == false)
                                {
                                    algoGui.errorLineFlag[j] = false;
                                    algoGui.errorLineFlagRunOnce[j] = true;
                                    goto error_check;
                                }
                                dataStr = "";
                            }
                        }
                        if (bids.bidsSizeArray[j] == 0 || bids.bidsSizeArray[j] < 100)
                        {
                            StreamWriter output = File.AppendText(@"X:\error_list2_bids_sizes.txt");
                            output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString());
                            //bids.bidsSizeArray[j] = 343;
                            algoGui.errorLineFlag[j] = true;
                            var stream = File.OpenRead(@"X:\bidsLine" + j + ".jpg");
                            text = bids_asksTessService.GetTextBids(j, stream);
                            output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString() + "\n");
                            output.Close();
                            if (algoGui.errorLineFlag[j] == true && algoGui.errorLineFlagRunOnce[j] == false)
                            {
                                algoGui.errorLineFlag[j] = false;
                                algoGui.errorLineFlagRunOnce[j] = true;
                                goto error_check;
                            }
                            //algoGui.critErr = true;
                        }
                    });
                }
                else
                {
                    Parallel.For(0, half_instances, new ParallelOptions { MaxDegreeOfParallelism = half_instances }, (j, state) =>
                    {
                        algoGui.errorLineFlagRunOnce[j + 16] = false;
                        string ipaddress = "";
                        int j_limit1 = 0;
                        int j_limit2 = 0;
                        int j_limit3 = 0;
                        bool supClusterActive = false;
                        asks.asksArray[j] = 0;
                        asks.asksSizeArray[j] = 0;
                        string text = "";
                        string dataStr = "";

                        if (algoGui.node1Online == true && algoGui.node2Online == true && algoGui.node3Online == true)
                        {
                            j_limit1 = 6;
                            j_limit2 = 12;
                            j_limit3 = 16;
                        }
                        else if (algoGui.node1Online == true || algoGui.node2Online == true)
                        {
                            if (algoGui.node1Online == true)
                            {
                                j_limit1 = 8;
                                j_limit2 = 8;
                            }
                            else if (algoGui.node2Online == true)
                            {
                                j_limit1 = 0;
                                j_limit2 = half_instances;
                            }
                        }
                        if (j >= 0 && j < j_limit1 && algoGui.algoGui_array[5].eblCluster.Checked == true && algoGui.node1Online == true)
                        {
                            ipaddress = algoGui.ipaddress1;
                            supClusterActive = true;
                        }
                        if (j >= j_limit1 && j < j_limit2 && algoGui.algoGui_array[5].eblCluster.Checked == true && algoGui.node2Online == true)
                        {
                            ipaddress = algoGui.ipaddress2;
                            supClusterActive = true;
                        }
                        if (j >= j_limit2 && j < j_limit3 && algoGui.algoGui_array[5].eblCluster.Checked == true && algoGui.node3Online == true)
                        {
                            ipaddress = algoGui.ipaddress3;
                            supClusterActive = true;
                        }
                        if (j >= 0 && j < j_limit3 && algoGui.algoGui_array[5].eblCluster.Checked == true && supClusterActive == true)
                        {
                            try
                            {
                                var request = new HttpRequestMessage(HttpMethod.Post, "http://" + ipaddress + "/uploader");
                                var content = new MultipartFormDataContent();

                                byte[] byteArray = File.ReadAllBytes(@"X:\asksLine" + j + ".jpg");
                                content.Add(new ByteArrayContent(byteArray), "file", "asksLine" + j + ".jpg");
                                request.Content = content;

                                var task1 = Task.Run(() => client.SendAsync(request));
                                task1.Wait();
                                var response = task1.Result;
                                response.EnsureSuccessStatusCode();

                                var execRequest = new HttpRequestMessage(HttpMethod.Get, "http://" + ipaddress + "/execute?program=Tesseract&imagefile=asksLine"
                                    + j + ".jpg&textfileout=asksLine" + j + "&languageparam=-l&languages=eng_fast&psm=--psm&numbervalue=7&oem=--oem&oemMode=3&typeofnumber=digits");//languages=eng%2Bdeu%2Bfra

                                var task2 = Task.Run(() => client.SendAsync(execRequest));
                                task2.Wait();
                                var execResponse = task2.Result;
                                execResponse.EnsureSuccessStatusCode();

                                var task3 = Task.Run(() => execResponse.Content.ReadAsStringAsync());
                                task3.Wait();
                                text = task3.Result;
                                text = getBetween(text, "{\"execute\":\"", "\\n\"");
                                text += "\n";
                            }
                            catch (Exception ex)
                            {
                                if (ipaddress == algoGui.ipaddress1)
                                {
                                    algoGui.node1Online = false;
                                    algoGui.algoGui_array[5].node1Status.Text = "Node 1 Offline";
                                }
                                else if (ipaddress == algoGui.ipaddress2)
                                {
                                    algoGui.node2Online = false;
                                    algoGui.algoGui_array[5].node2Status.Text = "Node 2 Offline";
                                }
                                else if (ipaddress == algoGui.ipaddress3)
                                {
                                    algoGui.node3Online = false;
                                    algoGui.algoGui_array[5].node3Status.Text = "Node 3 Offline";
                                }
                            }
                        }
                        else
                        {
                            var stream = File.OpenRead(@"X:\asksLine" + j + ".jpg");
                            text = bids_asksTessService.GetTextAsks(j, stream);
                        }

error_check:
                        bool sizePriceFlag = false;
                        foreach (Char data in text)
                        {
                            if (!Char.IsWhiteSpace(data))
                            {
								if(data.Equals(':'))
								{
									dataStr = dataStr + '.';
								}
								else
								{
									dataStr = dataStr + data;
								}
                            }
                            else if (Char.IsWhiteSpace(data) && dataStr != "")
                            {
                                if (sizePriceFlag == true)
                                {
                                    if (dataStr.EndsWith("K"))
                                    {
                                        dataStr = dataStr.Remove(dataStr.Length - 1, 1);
                                        dataStr = dataStr + '0' + '0' + '0';
                                    }
                                    try
                                    {
                                        asks.asksSizeArray[j] = Convert.ToInt32(dataStr, CultureInfo.InvariantCulture.NumberFormat);
                                    }
                                    catch (Exception ex)
                                    {
                                        StreamWriter output = File.AppendText(@"X:\error_list_asks_sizes.txt");
                                        output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString());
                                        //asks.asksSizeArray[j] = 343;
                                        algoGui.errorLineFlag[j + 16] = true;
                                        var stream = File.OpenRead(@"X:\asksLine" + j + ".jpg");
                                        text = bids_asksTessService.GetTextAsks(j, stream);
                                        output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString() + "\n");
                                        output.Close();
                                        if (algoGui.errorLineFlag[j + 16] == true && algoGui.errorLineFlagRunOnce[j + 16] == false)
                                        {
                                            algoGui.errorLineFlag[j + 16] = false;
                                            algoGui.errorLineFlagRunOnce[j + 16] = true;
                                            goto error_check;
                                        }
                                        //algoGui.critErr = true;
                                    }
                                    sizePriceFlag = false;
                                }
                                else
                                {
                                    sizePriceFlag = true;
                                    try
                                    {
                                        asks.asksArray[j] = (float)Convert.ToDouble(dataStr, CultureInfo.InvariantCulture.NumberFormat);
                                    }
                                    catch (Exception ex)
                                    {
                                        StreamWriter output = File.AppendText(@"X:\error_list_asks_prices.txt");
                                        output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString());
                                        //asks.asksArray[j] = 0f;
                                        //algoGui.critErr = true;
                                        algoGui.errorLineFlag[j + 16] = true;
                                        var stream = File.OpenRead(@"X:\asksLine" + j + ".jpg");
                                        text = bids_asksTessService.GetTextAsks(j, stream);
                                        output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString() + "\n");
                                        output.Close();
                                        if (algoGui.errorLineFlag[j + 16] == true && algoGui.errorLineFlagRunOnce[j + 16] == false)
                                        {
                                            algoGui.errorLineFlag[j + 16] = false;
                                            algoGui.errorLineFlagRunOnce[j + 16] = true;
                                            goto error_check;
                                        }
                                    }
                                }
                                dataStr = "";
                            }
                            else
                            {
                                StreamWriter output = File.AppendText(@"X:\error_list_entire_line_asks.txt");
                                output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString());
                                //algoGui.critErr = true;
                                algoGui.errorLineFlag[j + 16] = true;
                                var stream = File.OpenRead(@"X:\asksLine" + j + ".jpg");
                                text = bids_asksTessService.GetTextAsks(j, stream);
                                output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString() + "\n");
                                output.Close();
                                if (algoGui.errorLineFlag[j + 16] == true && algoGui.errorLineFlagRunOnce[j + 16] == false)
                                {
                                    algoGui.errorLineFlag[j + 16] = false;
                                    algoGui.errorLineFlagRunOnce[j + 16] = true;
                                    goto error_check;
                                }
                                dataStr = "";
                            }
                        }
                        if (asks.asksSizeArray[j] == 0 || asks.asksSizeArray[j] < 100)
                        {
                            StreamWriter output = File.AppendText(@"X:\error_list2_asks_sizes.txt");
                            output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString());
                            //asks.asksSizeArray[j] = 343;
                            algoGui.errorLineFlag[j + 16] = true;
                            var stream = File.OpenRead(@"X:\asksLine" + j + ".jpg");
                            text = bids_asksTessService.GetTextAsks(j, stream);
                            output.WriteLine(text.Remove(text.Length - 1, 1) + " line: " + j.ToString() + "\n");
                            output.Close();
                            if (algoGui.errorLineFlag[j + 16] == true && algoGui.errorLineFlagRunOnce[j + 16] == false)
                            {
                                algoGui.errorLineFlag[j + 16] = false;
                                algoGui.errorLineFlagRunOnce[j + 16] = true;
                                goto error_check;
                            }
                            //algoGui.critErr = true;
                        }
                    });
                }

                if (!bidsOrAsks)
                {
                    algoGui.asksCompletedEvent.Set();
                }
                else
                {
                    algoGui.bidsCompletedEvent.Set();
                }
            }

            public static string getBetween(string strSource, string strStart, string strEnd)
            {
                if (strSource.Contains(strStart) && strSource.Contains(strEnd))
                {
                    int Start, End;
                    Start = strSource.IndexOf(strStart, 0) + strStart.Length;
                    End = strSource.IndexOf(strEnd, Start);
                    return strSource.Substring(Start, End - Start);
                }

                return "";
            }
        }
        class ScreenCapturer
        {
            [DllImport("user32.dll")]
            private static extern IntPtr GetForegroundWindow();

            [DllImport("user32.dll")]
            private static extern IntPtr GetWindowRect(IntPtr hWnd, ref Rect rect);

            [StructLayout(LayoutKind.Sequential)]
            private struct Rect
            {
                public int Left;
                public int Top;
                public int Right;
                public int Bottom;
            }

            public Image FixedSize(Image imgPhoto, int Width, int Height)
            {
                int sourceWidth = imgPhoto.Width;
                int sourceHeight = imgPhoto.Height;
                int sourceX = 0;
                int sourceY = 0;
                int destX = 0;
                int destY = 0;

                float nPercent = 0;
                float nPercentW = 0;
                float nPercentH = 0;

                nPercentW = ((float)Width / (float)sourceWidth);
                nPercentH = ((float)Height / (float)sourceHeight);
                if (nPercentH < nPercentW)
                {
                    nPercent = nPercentH;
                    destX = System.Convert.ToInt16((Width -
                                  (sourceWidth * nPercent)) / 2);
                }
                else
                {
                    nPercent = nPercentW;
                    destY = System.Convert.ToInt16((Height -
                                  (sourceHeight * nPercent)) / 2);
                }

                int destWidth = (int)(sourceWidth * nPercent);
                int destHeight = (int)(sourceHeight * nPercent);

                Bitmap bmPhoto = new Bitmap(Width, Height,
                                  PixelFormat.Format24bppRgb);
                bmPhoto.SetResolution(imgPhoto.HorizontalResolution,
                                 imgPhoto.VerticalResolution);

                Graphics grPhoto = Graphics.FromImage(bmPhoto);
                grPhoto.Clear(Color.White);
                grPhoto.InterpolationMode =
                        InterpolationMode.HighQualityBicubic;

                grPhoto.DrawImage(imgPhoto,
                    new Rectangle(destX, destY, destWidth, destHeight),
                    new Rectangle(sourceX, sourceY, sourceWidth, sourceHeight),
                    GraphicsUnit.Pixel);

                grPhoto.Dispose();
                return bmPhoto;
            }

            public Bitmap Capture(enmScreenCaptureMode screenCaptureMode, bool bidsOrAsks)
            {
                Rectangle bounds;

                if (screenCaptureMode == enmScreenCaptureMode.Screen)
                {
                    bounds = Screen.GetBounds(System.Drawing.Point.Empty);
                    CursorPosition = Cursor.Position;
                }
                else
                {
                    IntPtr foregroundWindowsHandle;
                    if (bidsOrAsks == true)
                    {
                        foregroundWindowsHandle = Data_Scraper.algoGui.algoGui_array[0].Handle;
                    }

                    else
                    {
                        foregroundWindowsHandle = Data_Scraper.algoGui.algoGui_array[1].Handle;
                    }
                    var rect = new Rect();
                    GetWindowRect(foregroundWindowsHandle, ref rect);
                    bounds = new Rectangle(rect.Left, rect.Top, rect.Right - rect.Left, rect.Bottom - rect.Top);
                    CursorPosition = new System.Drawing.Point(Cursor.Position.X - rect.Left, Cursor.Position.Y - rect.Top);
                }

                var result = new Bitmap(bounds.Width, bounds.Height);

                using (var g = Graphics.FromImage(result))
                {
                    g.CopyFromScreen(new System.Drawing.Point(bounds.Left, bounds.Top), System.Drawing.Point.Empty, bounds.Size);
                }

                return result;
            }

            public Bitmap CaptureLast(enmScreenCaptureMode screenCaptureMode)
            {
                Rectangle bounds;

                if (screenCaptureMode == enmScreenCaptureMode.Screen)
                {
                    bounds = Screen.GetBounds(System.Drawing.Point.Empty);
                    CursorPosition = Cursor.Position;
                }
                else
                {
                    IntPtr foregroundWindowsHandle;
                    foregroundWindowsHandle = Data_Scraper.algoGui.algoGui_array[2].Handle;
                    var rect = new Rect();
                    GetWindowRect(foregroundWindowsHandle, ref rect);
                    bounds = new Rectangle(rect.Left, rect.Top, rect.Right - rect.Left, rect.Bottom - rect.Top);
                    CursorPosition = new System.Drawing.Point(Cursor.Position.X - rect.Left, Cursor.Position.Y - rect.Top);
                }

                var result = new Bitmap(bounds.Width, bounds.Height);

                using (var g = Graphics.FromImage(result))
                {
                    g.CopyFromScreen(new System.Drawing.Point(bounds.Left, bounds.Top), System.Drawing.Point.Empty, bounds.Size);
                }

                return result;
            }

            public Bitmap CaptureDow(enmScreenCaptureMode screenCaptureMode)
            {
                Rectangle bounds;

                if (screenCaptureMode == enmScreenCaptureMode.Screen)
                {
                    bounds = Screen.GetBounds(System.Drawing.Point.Empty);
                    CursorPosition = Cursor.Position;
                }
                else
                {
                    IntPtr foregroundWindowsHandle;
                    foregroundWindowsHandle = Data_Scraper.algoGui.algoGui_array[3].Handle;
                    var rect = new Rect();
                    GetWindowRect(foregroundWindowsHandle, ref rect);
                    bounds = new Rectangle(rect.Left + 50, rect.Top, rect.Right - rect.Left - 50, rect.Bottom - rect.Top);
                    CursorPosition = new System.Drawing.Point(Cursor.Position.X - rect.Left, Cursor.Position.Y - rect.Top);
                }

                var result = new Bitmap(bounds.Width, bounds.Height);

                using (var g = Graphics.FromImage(result))
                {
                    g.CopyFromScreen(new System.Drawing.Point(bounds.Left, bounds.Top), System.Drawing.Point.Empty, bounds.Size);
                }

                return result;
            }

            public Bitmap CaptureMstClk(enmScreenCaptureMode screenCaptureMode)
            {
                Rectangle bounds;

                if (screenCaptureMode == enmScreenCaptureMode.Screen)
                {
                    bounds = Screen.GetBounds(System.Drawing.Point.Empty);
                    CursorPosition = Cursor.Position;
                }
                else
                {
                    IntPtr foregroundWindowsHandle;
                    foregroundWindowsHandle = Data_Scraper.algoGui.algoGui_array[4].Handle;
                    var rect = new Rect();
                    GetWindowRect(foregroundWindowsHandle, ref rect);
                    bounds = new Rectangle(rect.Left + 30, rect.Top + 25, rect.Right - rect.Left - 113, rect.Bottom - rect.Top - 40);
                    CursorPosition = new System.Drawing.Point(Cursor.Position.X - rect.Left, Cursor.Position.Y - rect.Top);
                }

                var result = new Bitmap(bounds.Width, bounds.Height);

                using (var g = Graphics.FromImage(result))
                {
                    g.CopyFromScreen(new System.Drawing.Point(bounds.Left, bounds.Top), System.Drawing.Point.Empty, bounds.Size);
                }

                return result;
            }

            public Bitmap ResizeImage(System.Drawing.Image image, int width, int height)
            {
                var destRect = new Rectangle(0, 0, width, height);
                var destImage = new Bitmap(width, height);

                destImage.SetResolution(600, 600);

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

            public Bitmap Sharpen(Bitmap image)
            {
                Bitmap sharpenImage = (Bitmap)image.Clone();

                int filterWidth = 3;
                int filterHeight = 3;
                int width = image.Width;
                int height = image.Height;

                // Create sharpening filter.
                double[,] filter = new double[filterWidth, filterHeight];
                filter[0, 0] = filter[0, 1] = filter[0, 2] = filter[1, 0] = filter[1, 2] = filter[2, 0] = filter[2, 1] = filter[2, 2] = -1;
                filter[1, 1] = 9;

                double factor = 1.0;
                double bias = 0.0;

                Color[,] result = new Color[image.Width, image.Height];

                // Lock image bits for read/write.
                BitmapData pbits = sharpenImage.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

                // Declare an array to hold the bytes of the bitmap.
                int bytes = pbits.Stride * height;
                byte[] rgbValues = new byte[bytes];

                // Copy the RGB values into the array.
                System.Runtime.InteropServices.Marshal.Copy(pbits.Scan0, rgbValues, 0, bytes);

                int rgb;
                // Fill the color array with the new sharpened color values.
                for (int x = 0; x < width; ++x)
                {
                    for (int y = 0; y < height; ++y)
                    {
                        double red = 0.0, green = 0.0, blue = 0.0;

                        for (int filterX = 0; filterX < filterWidth; filterX++)
                        {
                            for (int filterY = 0; filterY < filterHeight; filterY++)
                            {
                                int imageX = (x - filterWidth / 2 + filterX + width) % width;
                                int imageY = (y - filterHeight / 2 + filterY + height) % height;

                                rgb = imageY * pbits.Stride + 3 * imageX;

                                red += rgbValues[rgb + 2] * filter[filterX, filterY];
                                green += rgbValues[rgb + 1] * filter[filterX, filterY];
                                blue += rgbValues[rgb + 0] * filter[filterX, filterY];
                            }
                            int r = Math.Min(Math.Max((int)(factor * red + bias), 0), 255);
                            int g = Math.Min(Math.Max((int)(factor * green + bias), 0), 255);
                            int b = Math.Min(Math.Max((int)(factor * blue + bias), 0), 255);

                            result[x, y] = Color.FromArgb(r, g, b);
                        }
                    }
                }

                // Update the image with the sharpened pixels.
                for (int x = 0; x < width; ++x)
                {
                    for (int y = 0; y < height; ++y)
                    {
                        rgb = y * pbits.Stride + 3 * x;

                        rgbValues[rgb + 2] = result[x, y].R;
                        rgbValues[rgb + 1] = result[x, y].G;
                        rgbValues[rgb + 0] = result[x, y].B;
                    }
                }

                // Copy the RGB values back to the bitmap.
                System.Runtime.InteropServices.Marshal.Copy(rgbValues, 0, pbits.Scan0, bytes);
                // Release image bits.
                sharpenImage.UnlockBits(pbits);

                return sharpenImage;
            }

            public Bitmap Dilate(Bitmap SrcImage)
            {
                // Create Destination bitmap.
                Bitmap tempbmp = new Bitmap(SrcImage.Width, SrcImage.Height);

                // Take source bitmap data.
                BitmapData SrcData = SrcImage.LockBits(new Rectangle(0, 0,
                    SrcImage.Width, SrcImage.Height), ImageLockMode.ReadOnly,
                    PixelFormat.Format24bppRgb);

                // Take destination bitmap data.
                BitmapData DestData = tempbmp.LockBits(new Rectangle(0, 0, tempbmp.Width,
                    tempbmp.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

                // Element array to used to dilate.
                byte[,] sElement = new byte[5, 5] {
                                                    {0,0,0,0,0},
                                                    {0,0,1,0,0},
                                                    {0,1,0,1,0},
                                                    {0,0,1,0,0},
                                                    {0,0,0,0,0}
                                                };

                // Element array size.
                int size = 5;
                byte max, clrValue;
                int radius = size / 2;
                int ir, jr;

                unsafe
                {

                    // Loop for Columns.
                    for (int colm = radius; colm < DestData.Height - radius; colm++)
                    {
                        // Initialise pointers to at row start.
                        byte* ptr = (byte*)SrcData.Scan0 + (colm * SrcData.Stride);
                        byte* dstPtr = (byte*)DestData.Scan0 + (colm * SrcData.Stride);

                        // Loop for Row item.
                        for (int row = radius; row < DestData.Width - radius; row++)
                        {
                            max = 0;
                            clrValue = 0;

                            // Loops for element array.
                            for (int eleColm = 0; eleColm < 5; eleColm++)
                            {
                                ir = eleColm - radius;
                                byte* tempPtr = (byte*)SrcData.Scan0 +
                                    ((colm + ir) * SrcData.Stride);

                                for (int eleRow = 0; eleRow < 5; eleRow++)
                                {
                                    jr = eleRow - radius;

                                    // Get neightbour element color value.
                                    clrValue = (byte)((tempPtr[row * 3 + jr] +
                                        tempPtr[row * 3 + jr + 1] + tempPtr[row * 3 + jr + 2]) / 3);

                                    if (max < clrValue)
                                    {
                                        if (sElement[eleColm, eleRow] != 0)
                                            max = clrValue;
                                    }
                                }
                            }

                            dstPtr[0] = dstPtr[1] = dstPtr[2] = max;

                            ptr += 3;
                            dstPtr += 3;
                        }
                    }
                }

                // Dispose all Bitmap data.
                SrcImage.UnlockBits(SrcData);
                tempbmp.UnlockBits(DestData);

                // return dilated bitmap.
                return tempbmp;
            }

            public Bitmap AdaptiveThreshold(Bitmap image, double a, double b)
            {
                int w = image.Width;
                int h = image.Height;

                BitmapData image_data = image.LockBits(
                    new Rectangle(0, 0, w, h),
                    ImageLockMode.ReadOnly,
                    PixelFormat.Format24bppRgb);

                int bytes = image_data.Stride * image_data.Height;
                byte[] buffer = new byte[bytes];
                byte[] result = new byte[bytes];

                Marshal.Copy(image_data.Scan0, buffer, 0, bytes);
                image.UnlockBits(image_data);

                //Get global mean - this works only for grayscale images
                double mg = 0;
                for (int i = 0; i < bytes; i += 3)
                {
                    mg += buffer[i];
                }
                mg /= (w * h);

                for (int x = 1; x < w - 1; x++)
                {
                    for (int y = 1; y < h - 1; y++)
                    {
                        int position = x * 3 + y * image_data.Stride;
                        double[] histogram = new double[256];

                        for (int i = -1; i <= 1; i++)
                        {
                            for (int j = -1; j <= 1; j++)
                            {
                                int nposition = position + i * 3 + j * image_data.Stride;
                                histogram[buffer[nposition]]++;
                            }
                        }

                        histogram = histogram.Select(l => l / (w * h)).ToArray();

                        double mean = 0;
                        for (int i = 0; i < 256; i++)
                        {
                            mean += i * histogram[i];
                        }

                        double std = 0;
                        for (int i = 0; i < 256; i++)
                        {
                            std += Math.Pow(i - mean, 2) * histogram[i];
                        }
                        std = Math.Sqrt(std);

                        double threshold = a * std + b * mg;
                        for (int c = 0; c < 3; c++)
                        {
                            result[position + c] = (byte)((buffer[position] > threshold) ? 255 : 0);
                        }
                    }
                }

                Bitmap res_img = new Bitmap(w, h);
                BitmapData res_data = res_img.LockBits(
                    new Rectangle(0, 0, w, h),
                    ImageLockMode.WriteOnly,
                    PixelFormat.Format24bppRgb);
                Marshal.Copy(result, 0, res_data.Scan0, bytes);
                res_img.UnlockBits(res_data);

                return res_img;
            }

            public Bitmap DilateAndErodeFilter(
                           Bitmap sourceBitmap,
                           int matrixSize,
                           int morphType,
                           bool applyBlue = true,
                           bool applyGreen = true,
                           bool applyRed = true)
            {
                BitmapData sourceData =
                           sourceBitmap.LockBits(new Rectangle(0, 0,
                           sourceBitmap.Width, sourceBitmap.Height),
                           ImageLockMode.ReadOnly,
                           PixelFormat.Format32bppArgb);


                byte[] pixelBuffer = new byte[sourceData.Stride *
                                              sourceData.Height];


                byte[] resultBuffer = new byte[sourceData.Stride *
                                               sourceData.Height];


                Marshal.Copy(sourceData.Scan0, pixelBuffer, 0,
                                           pixelBuffer.Length);


                sourceBitmap.UnlockBits(sourceData);


                int filterOffset = (matrixSize - 1) / 2;
                int calcOffset = 0;


                int byteOffset = 0;


                byte blue = 0;
                byte green = 0;
                byte red = 0;


                byte morphResetValue = 0;


                if (morphType == 1)
                {
                    morphResetValue = 255;
                }


                for (int offsetY = filterOffset; offsetY <
                    sourceBitmap.Height - filterOffset; offsetY++)
                {
                    for (int offsetX = filterOffset; offsetX <
                        sourceBitmap.Width - filterOffset; offsetX++)
                    {
                        byteOffset = offsetY *
                                     sourceData.Stride +
                                     offsetX * 4;


                        blue = morphResetValue;
                        green = morphResetValue;
                        red = morphResetValue;


                        if (morphType == 0)
                        {
                            for (int filterY = -filterOffset;
                                filterY <= filterOffset; filterY++)
                            {
                                for (int filterX = -filterOffset;
                                    filterX <= filterOffset; filterX++)
                                {
                                    calcOffset = byteOffset +
                                                 (filterX * 4) +
                                    (filterY * sourceData.Stride);


                                    if (pixelBuffer[calcOffset] > blue)
                                    {
                                        blue = pixelBuffer[calcOffset];
                                    }


                                    if (pixelBuffer[calcOffset + 1] > green)
                                    {
                                        green = pixelBuffer[calcOffset + 1];
                                    }


                                    if (pixelBuffer[calcOffset + 2] > red)
                                    {
                                        red = pixelBuffer[calcOffset + 2];
                                    }
                                }
                            }
                        }
                        else if (morphType == 1)
                        {
                            for (int filterY = -filterOffset;
                                filterY <= filterOffset; filterY++)
                            {
                                for (int filterX = -filterOffset;
                                    filterX <= filterOffset; filterX++)
                                {
                                    calcOffset = byteOffset +
                                                 (filterX * 4) +
                                    (filterY * sourceData.Stride);


                                    if (pixelBuffer[calcOffset] < blue)
                                    {
                                        blue = pixelBuffer[calcOffset];
                                    }


                                    if (pixelBuffer[calcOffset + 1] < green)
                                    {
                                        green = pixelBuffer[calcOffset + 1];
                                    }


                                    if (pixelBuffer[calcOffset + 2] < red)
                                    {
                                        red = pixelBuffer[calcOffset + 2];
                                    }
                                }
                            }
                        }


                        if (applyBlue == false)
                        {
                            blue = pixelBuffer[byteOffset];
                        }


                        if (applyGreen == false)
                        {
                            green = pixelBuffer[byteOffset + 1];
                        }


                        if (applyRed == false)
                        {
                            red = pixelBuffer[byteOffset + 2];
                        }


                        resultBuffer[byteOffset] = blue;
                        resultBuffer[byteOffset + 1] = green;
                        resultBuffer[byteOffset + 2] = red;
                        resultBuffer[byteOffset + 3] = 255;
                    }
                }


                Bitmap resultBitmap = new Bitmap(sourceBitmap.Width,
                                                 sourceBitmap.Height);


                BitmapData resultData =
                           resultBitmap.LockBits(new Rectangle(0, 0,
                           resultBitmap.Width, resultBitmap.Height),
                           ImageLockMode.WriteOnly,
                           PixelFormat.Format32bppArgb);


                Marshal.Copy(resultBuffer, 0, resultData.Scan0,
                                           resultBuffer.Length);


                resultBitmap.UnlockBits(resultData);


                return resultBitmap;
            }
            public static T[][] ArrayClone<T>(T[][] A)
            { return A.Select(a => a.ToArray()).ToArray(); }

            public bool[][] Image2Bool(System.Drawing.Image img)
            {
                Bitmap bmp = new Bitmap(img);
                bool[][] s = new bool[bmp.Height][];
                for (int y = 0; y < bmp.Height; y++)
                {
                    s[y] = new bool[bmp.Width];
                    for (int x = 0; x < bmp.Width; x++)
                        s[y][x] = bmp.GetPixel(x, y).GetBrightness() < 0.3;
                }
                return s;

            }

            public System.Drawing.Image Bool2Image(bool[][] s)
            {
                Bitmap bmp = new Bitmap(s[0].Length, s.Length);
                using (Graphics g = Graphics.FromImage(bmp)) g.Clear(Color.White);
                for (int y = 0; y < bmp.Height; y++)
                    for (int x = 0; x < bmp.Width; x++)
                        if (s[y][x]) bmp.SetPixel(x, y, Color.Black);

                return (Bitmap)bmp;
            }

            public bool[][] ZhangSuenThinning(bool[][] s)
            {
                bool[][] temp = ArrayClone(s);  // make a deep copy to start.. 
                int count = 0;
                do  // the missing iteration
                {
                    count = step(1, temp, s);
                    temp = ArrayClone(s);      // ..and on each..
                    count += step(2, temp, s);
                    temp = ArrayClone(s);      // ..call!
                }
                while (count > 0);

                return s;
            }

            int step(int stepNo, bool[][] temp, bool[][] s)
            {
                int count = 0;

                for (int a = 1; a < temp.Length - 1; a++)
                {
                    for (int b = 1; b < temp[0].Length - 1; b++)
                    {
                        if (SuenThinningAlg(a, b, temp, stepNo == 2))
                        {
                            // still changes happening?
                            if (s[a][b]) count++;
                            s[a][b] = false;
                        }
                    }
                }
                return count;
            }

            bool SuenThinningAlg(int x, int y, bool[][] s, bool even)
            {
                bool p2 = s[x][y - 1];
                bool p3 = s[x + 1][y - 1];
                bool p4 = s[x + 1][y];
                bool p5 = s[x + 1][y + 1];
                bool p6 = s[x][y + 1];
                bool p7 = s[x - 1][y + 1];
                bool p8 = s[x - 1][y];
                bool p9 = s[x - 1][y - 1];


                int bp1 = NumberOfNonZeroNeighbors(x, y, s);
                if (bp1 >= 2 && bp1 <= 6) //2nd condition
                {
                    if (NumberOfZeroToOneTransitionFromP9(x, y, s) == 1)
                    {
                        if (even)
                        {
                            if (!((p2 && p4) && p8))
                            {
                                if (!((p2 && p6) && p8))
                                {
                                    return true;
                                }
                            }
                        }
                        else
                        {
                            if (!((p2 && p4) && p6))
                            {
                                if (!((p4 && p6) && p8))
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
                return false;
            }

            int NumberOfZeroToOneTransitionFromP9(int x, int y, bool[][] s)
            {
                bool p2 = s[x][y - 1];
                bool p3 = s[x + 1][y - 1];
                bool p4 = s[x + 1][y];
                bool p5 = s[x + 1][y + 1];
                bool p6 = s[x][y + 1];
                bool p7 = s[x - 1][y + 1];
                bool p8 = s[x - 1][y];
                bool p9 = s[x - 1][y - 1];

                int A = Convert.ToInt32((!p2 && p3)) + Convert.ToInt32((!p3 && p4)) +
                        Convert.ToInt32((!p4 && p5)) + Convert.ToInt32((!p5 && p6)) +
                        Convert.ToInt32((!p6 && p7)) + Convert.ToInt32((!p7 && p8)) +
                        Convert.ToInt32((!p8 && p9)) + Convert.ToInt32((!p9 && p2));
                return A;
            }
            int NumberOfNonZeroNeighbors(int x, int y, bool[][] s)
            {
                int count = 0;
                if (s[x - 1][y]) count++;
                if (s[x - 1][y + 1]) count++;
                if (s[x - 1][y - 1]) count++;
                if (s[x][y + 1]) count++;
                if (s[x][y - 1]) count++;
                if (s[x + 1][y]) count++;
                if (s[x + 1][y + 1]) count++;
                if (s[x + 1][y - 1]) count++;
                return count;
            }

            public void CopyBmpRegion(Bitmap image, Rectangle srcRect, System.Drawing.Point destLocation)
            {
                //do some argument sanitising.
                if (!((srcRect.X >= 0 && srcRect.Y >= 0) && ((srcRect.X + srcRect.Width) <= image.Width) && ((srcRect.Y + srcRect.Height) <= image.Height)))
                    throw new ArgumentException("Source rectangle isn't within the image bounds.");

                if ((destLocation.X < 0 || destLocation.X > image.Width) || (destLocation.Y < 0 || destLocation.Y > image.Height))
                    throw new ArgumentException("Destination must be within the image.");

                // Lock the bits into memory
                BitmapData bmpData = image.LockBits(new Rectangle(System.Drawing.Point.Empty, image.Size), ImageLockMode.ReadWrite, image.PixelFormat);
                int pxlSize = (bmpData.Stride / bmpData.Width); //calculate the pixel width (in bytes) of the current image.
                int src = 0; int dest = 0; //source/destination pixels.

                //account for the fact that not all of the source rectangle may be able to copy into the destination:
                int width = (destLocation.X + srcRect.Width) <= image.Width ? srcRect.Width : (image.Width - (destLocation.X + srcRect.Width));
                int height = (destLocation.Y + srcRect.Height) <= image.Height ? srcRect.Height : (image.Height - (destLocation.Y + srcRect.Height));

                //managed buffer to hold the current pixel data.
                byte[] buffer = new byte[pxlSize];

                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        //calculate the start of the current source pixel and destination pixel.
                        src = ((srcRect.Y + y) * bmpData.Stride) + ((srcRect.X + x) * pxlSize);
                        dest = ((destLocation.Y + y) * bmpData.Stride) + ((destLocation.X + x) * pxlSize);

                        // Can replace this with unsafe code, but that's up to you.
                        Marshal.Copy(new IntPtr(bmpData.Scan0.ToInt32() + src), buffer, 0, pxlSize);
                        Marshal.Copy(buffer, 0, new IntPtr(bmpData.Scan0.ToInt32() + dest), pxlSize);
                    }
                }

                image.UnlockBits(bmpData); //unlock the data.
            }

            public Bitmap ReduceToTwoColorFade(Bitmap image, Boolean bgWhite)
            {
                // Get data out of the image, using LockBits and Marshal.Copy
                Int32 width = image.Width;
                Int32 height = image.Height;
                // LockBits can actually -convert- the image data to the requested colour depth.
                // 32 bpp is the easiest to get the colour components out.
                BitmapData sourceData = image.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
                // Not really needed for 32bpp, but technically the stride does not always match the
                // amount of used data on each line, since the stride gets rounded up to blocks of 4.
                Int32 stride = sourceData.Stride;
                Byte[] imgBytes = new Byte[stride * height];
                Marshal.Copy(sourceData.Scan0, imgBytes, 0, imgBytes.Length);
                image.UnlockBits(sourceData);
                // Make colour population histogram
                Int32 lineOffset = 0;
                Dictionary<Int32, Int32> histogram = new Dictionary<Int32, Int32>();
                for (Int32 y = 0; y < height; y++)
                {
                    Int32 offset = lineOffset;
                    for (Int32 x = 0; x < width; x++)
                    {
                        // Optional check: only handle if not mostly-transparent
                        if (imgBytes[offset + 3] > 0x7F)
                        {
                            // Get colour values from bytes, without alpha.
                            // Little-endian: UInt32 0xAARRGGBB = Byte[] { BB, GG, RR, AA }
                            Int32 val = (imgBytes[offset + 2] << 16) | (imgBytes[offset + 1] << 8) | imgBytes[offset + 0];
                            if (histogram.ContainsKey(val))
                                histogram[val] = histogram[val] + 1;
                            else
                                histogram[val] = 1;
                        }
                        offset += 4;
                    }
                    lineOffset += stride;
                }
                // Sort the histogram. This requires System.Linq
                KeyValuePair<Int32, Int32>[] histoSorted = histogram.OrderByDescending(c => c.Value).ToArray();
                // Technically these colours will be transparent when built like this, since their 
                // alpha is 0, but we won't use them directly as colours anyway.
                // Since we filter on alpha, getting a result is not 100% guaranteed.
                Color colBackgr = histoSorted.Length < 1 ? Color.Black : Color.FromArgb(histoSorted[0].Key);
                // if less than 2 colors, just default it to the same.
                Color colContent = histoSorted.Length < 2 ? colBackgr : Color.FromArgb(histoSorted[1].Key);
                // Make a new 256-colour palette, making a fade between these two colours, for feeding into GetClosestPaletteIndexMatch later
                Color[] matchPal = new Color[0x100];
                Color toBlack = bgWhite ? colContent : colBackgr;
                Color toWhite = bgWhite ? colBackgr : colContent;
                Int32 rFirst = toBlack.R;
                Int32 gFirst = toBlack.G;
                Int32 bFirst = toBlack.B;
                Double rDif = (toBlack.R - toWhite.R) / 255.0;
                Double gDif = (toBlack.G - toWhite.G) / 255.0;
                Double bDif = (toBlack.B - toWhite.B) / 255.0;
                for (Int32 i = 0; i < 0x100; i++)
                    matchPal[i] = Color.FromArgb(
                        Math.Min(0xFF, Math.Max(0, rFirst - (Int32)Math.Round(rDif * i, MidpointRounding.AwayFromZero))),
                        Math.Min(0xFF, Math.Max(0, gFirst - (Int32)Math.Round(gDif * i, MidpointRounding.AwayFromZero))),
                        Math.Min(0xFF, Math.Max(0, bFirst - (Int32)Math.Round(bDif * i, MidpointRounding.AwayFromZero))));
                // Ensure start and end point are correct, and not mangled by small rounding errors.
                matchPal[0x00] = Color.FromArgb(toBlack.R, toBlack.G, toBlack.B);
                matchPal[0xFF] = Color.FromArgb(toWhite.R, toWhite.G, toWhite.B);
                // The 8-bit stride is simply the width in this case.
                Int32 stride8Bit = width;
                // Make 8-bit array to store the result
                Byte[] imgBytes8Bit = new Byte[stride8Bit * height];
                // Reset offset for a new loop through the image data
                lineOffset = 0;
                // Make new offset var for a loop through the 8-bit image data
                Int32 lineOffset8Bit = 0;
                for (Int32 y = 0; y < height; y++)
                {
                    Int32 offset = lineOffset;
                    Int32 offset8Bit = lineOffset8Bit;
                    for (Int32 x = 0; x < width; x++)
                    {
                        Int32 toWrite;
                        // If transparent, revert to background colour.
                        if (imgBytes[offset + 3] <= 0x7F)
                        {
                            toWrite = bgWhite ? 0xFF : 0x00;
                        }
                        else
                        {
                            Color col = Color.FromArgb(imgBytes[offset + 2], imgBytes[offset + 1], imgBytes[offset + 0]);
                            toWrite = GetClosestPaletteIndexMatch(col, matchPal);
                        }
                        // Write the found colour index to the 8-bit byte array.
                        imgBytes8Bit[offset8Bit] = (Byte)toWrite;
                        offset += 4;
                        offset8Bit++;
                    }
                    lineOffset += stride;
                    lineOffset8Bit += stride8Bit;
                }
                // Make new 8-bit image and copy the data into it.
                Bitmap newBm = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
                BitmapData targetData = newBm.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, newBm.PixelFormat);
                //  get minimum data width for the pixel format.
                Int32 newDataWidth = ((System.Drawing.Image.GetPixelFormatSize(newBm.PixelFormat) * width) + 7) / 8;
                // Note that this Stride will most likely NOT match the image width; it is rounded up to the
                // next multiple of 4 bytes. For that reason, we copy the data per line, and not as one block.
                Int32 targetStride = targetData.Stride;
                Int64 scan0 = targetData.Scan0.ToInt64();
                for (Int32 y = 0; y < height; ++y)
                    Marshal.Copy(imgBytes8Bit, y * stride8Bit, new IntPtr(scan0 + y * targetStride), newDataWidth);
                newBm.UnlockBits(targetData);
                // Set final image palette to grayscale fade.
                // 'Image.Palette' makes a COPY of the palette when accessed.
                // So copy it out, modify it, then copy it back in.
                ColorPalette pal = newBm.Palette;
                for (Int32 i = 0; i < 0x100; i++)
                    pal.Entries[i] = Color.FromArgb(i, i, i);
                newBm.Palette = pal;
                return newBm;
            }

            public static Int32 GetClosestPaletteIndexMatch(Color col, Color[] colorPalette)
            {
                Int32 colorMatch = 0;
                Int32 leastDistance = Int32.MaxValue;
                Int32 red = col.R;
                Int32 green = col.G;
                Int32 blue = col.B;
                for (Int32 i = 0; i < colorPalette.Length; ++i)
                {
                    Color paletteColor = colorPalette[i];
                    Int32 redDistance = paletteColor.R - red;
                    Int32 greenDistance = paletteColor.G - green;
                    Int32 blueDistance = paletteColor.B - blue;
                    // Technically, Pythagorean distance needs to have a root taken of the result, but this is not needed for just comparing them.
                    Int32 distance = (redDistance * redDistance) + (greenDistance * greenDistance) + (blueDistance * blueDistance);
                    if (distance >= leastDistance)
                        continue;
                    colorMatch = i;
                    leastDistance = distance;
                    if (distance == 0)
                        return i;
                }
                return colorMatch;
            }
            public System.Drawing.Point CursorPosition
            {
                get;
                protected set;
            }
        }
    }
}