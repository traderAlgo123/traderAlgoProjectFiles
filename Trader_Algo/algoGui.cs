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
using System.Collections;
using System.ComponentModel;
using System.Linq;
using System.Data;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Threading;
using System.IO;
using System.Runtime.InteropServices;
using System.Timers;
using System.Diagnostics;
using System.Configuration;
using System.Data.SqlClient;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Net;
using System.Net.Http;

namespace Data_Scraper
{
    public partial class algoGui : Form
    {
        [DllImport("user32.dll", CharSet = CharSet.Auto, CallingConvention = CallingConvention.StdCall)]
        public static extern void mouse_event(uint dwFlags, int dx, int dy, uint cButtons, uint dwExtraInfo);
        //Mouse actions
        private const int MOUSEEVENTF_LEFTDOWN = 0x02;
        private const int MOUSEEVENTF_LEFTUP = 0x04;
        private const int MOUSEEVENTF_RIGHTDOWN = 0x08;
        private const int MOUSEEVENTF_RIGHTUP = 0x10;

        public static int bids_asks_height = 244;
        public static int bids_asks_width = 111; //211
        public static string bids_scrn_output = "";
        public static string bids_size_output = "";
        public static string asks_scrn_output = "";
        public static string asks_size_output = "";
        public static string mstClkOutput = "";
        public static string dowOutput = "";
        public static string lastTradePriceOutput = "";
        public static string lastTradeSizeOutput = "";
        public static string prevTradePriceOutput = "";
        public static string prevTradeSizeOutput = "";
        public static string priceTierBidsOutput = "";
        public static string priceTierAsksOutput = "";
        public static ManualResetEvent asksCompletedEvent = new ManualResetEvent(false);
        public static ManualResetEvent asksDataScrapeCompletedEvent = new ManualResetEvent(false);
        public static ManualResetEvent bidsDataScrapeCompletedEvent = new ManualResetEvent(false);
        public static ManualResetEvent bidsCompletedEvent = new ManualResetEvent(false);
        public static ManualResetEvent newTradesFoundEvent = new ManualResetEvent(false);
        public static SqlConnection conn = new SqlConnection(ConfigurationManager.ConnectionStrings["DataScraperConnString"].ConnectionString);

        private static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
        private const UInt32 SWP_NOSIZE = 0x0001;
        private const UInt32 SWP_NOMOVE = 0x0002;
        private const UInt32 TOPMOST_FLAGS = SWP_NOMOVE | SWP_NOSIZE;

        public static algoGui[] algoGui_array = new algoGui[6];
        public static Thread bidsThread = new Thread(ThreadWork.BidsWork);
        public static Thread asksThread = new Thread(ThreadWork.AsksWork);
        public static Thread lastsThread = new Thread(ThreadWork.LastTradesInit);
        public static Thread dowThread = new Thread(ThreadWork.DowInit);
        public static Thread etradeClkThread = new Thread(ThreadWork.etradeClk);
        public static Thread mainWinThread = new Thread(ThreadWork.mainAlgoWindowInit);
        public static Thread executionChecker = new Thread(ThreadWork.executionCheckerCtrl);
        public static BackgroundWorker bidsCtrl = new BackgroundWorker();
        public static BackgroundWorker asksCtrl = new BackgroundWorker();
        public static BackgroundWorker lastsCtrl = new BackgroundWorker();
        public static BackgroundWorker dowCtrl = new BackgroundWorker();
        public static BackgroundWorker etradeClkCtrl = new BackgroundWorker();

        public static BackgroundWorker clusterNodePingCtrl = new BackgroundWorker();

        public static int numEvents = 100;

        public static int eventsItr = 0;
        public static double ratesAccum = 0;

        public static vectorTensor tempTensor = new vectorTensor();
        public static vectorTensor prevTensor = new vectorTensor();
        public static vectorTensor[] tensorArray = new vectorTensor[numEvents];
        public static vectorTensor[] zeroTensorArray = new vectorTensor[numEvents];

        public static int tensor_count = 0;
        public static bool trade_event = false;
        public static bool tensorPassed = false;

        public static bool node1Online = false;
        public static bool node2Online = false;
        public static bool node3Online = false;

        public static bool stopScraper = false;
        public static bool pre_after_market_mode = false;
        public static System.Timers.Timer t = new System.Timers.Timer(60000);
        public static int tensor_index = 0;
        public static int tradesCmpIdx = 0;
        public static bool firstrun = true;
        public static bool firstStartClick = true;
        public static bool lastTradesScraperRunning = false;
        public static DateTime past = DateTime.Now;

        public static bool critErr = false;

        public static bidsObj bids = new bidsObj();
        public static asksObj asks = new asksObj();
        public static dowObj dow = new dowObj();
        public static etradeClkObj etradeClk = new etradeClkObj();
        public static lastTradesObj lastTrades = new lastTradesObj();
        public static lastTradesObj prevTrades = new lastTradesObj();

        public static string ipaddress1 = "192.168.1.121";
        public static string ipaddress3 = "192.168.1.224";
        //public static string ipaddress3 = "172.28.117.71";
        //public static string ipaddress2 = "172.28.164.120";
        //public static string ipaddress1 = "172.28.117.71";
        public static string ipaddress2 = "192.168.1.249";
        //public static string ipaddress2 = "172.28.43.111";

        public static ProcessStartInfo predictorProcess = new ProcessStartInfo("Predictor.exe");
        public static readonly HttpClient pingClient = new HttpClient();
        public algoGui()
        {
            InitializeComponent();
        }

        private void algoGui_Load(object sender, EventArgs e)
        {
            ServicePointManager.DefaultConnectionLimit = 64;
            pingClient.Timeout = TimeSpan.FromMilliseconds(3000);
            for (int i = 0; i < numEvents; i++)
            {
                tensorArray[i] = new vectorTensor();
                zeroTensorArray[i] = new vectorTensor();
            }
            //startConsole();

            bidsCtrl.DoWork += bidsCtrl_DoWork;
            bidsCtrl.RunWorkerCompleted += bidsCtrl_RunWorkerCompleted;

            asksCtrl.DoWork += asksCtrl_DoWork;
            asksCtrl.RunWorkerCompleted += asksCtrl_RunWorkerCompleted;

            lastsCtrl.DoWork += lastsCtrl_DoWork;
            lastsCtrl.RunWorkerCompleted += lastsCtrl_RunWorkerCompleted;

            dowCtrl.DoWork += dowCtrl_DoWork;
            dowCtrl.RunWorkerCompleted += dowCtrl_RunWorkerCompleted;

            etradeClkCtrl.DoWork += etradeClkCtrl_DoWork;
            etradeClkCtrl.RunWorkerCompleted += etradeClkCtrl_RunWorkerCompleted;

            clusterNodePingCtrl.DoWork += clusterNodePingCtrl_DoWork;
            clusterNodePingCtrl.RunWorkerCompleted += clusterNodePingCtrl_RunWorkerCompleted;

            if (this.Equals(algoGui_array[0]))
            {
                this.Location = Screen.AllScreens[1].WorkingArea.Location;
                algoGui_array[0].Height = bids_asks_height + 160;
                algoGui_array[0].Width = bids_asks_width + 80;
                algoGui_array[0].Top += 184;
                algoGui_array[0].Left += 956;
                SetWindowPos(algoGui_array[0].Handle, HWND_TOPMOST, 0, 0, 0, 0, TOPMOST_FLAGS);
                algoGui_array[0].Opacity = 0.5;
            }
            if (this.Equals(algoGui_array[1]))
            {
                this.Location = Screen.AllScreens[1].WorkingArea.Location;
                algoGui_array[1].Height = bids_asks_height + 160;
                algoGui_array[1].Width = bids_asks_width + 80;
                algoGui_array[1].Left += 1320;
                algoGui_array[1].Top += 184;
                SetWindowPos(algoGui_array[1].Handle, HWND_TOPMOST, 0, 0, 0, 0, TOPMOST_FLAGS);
                algoGui_array[1].Opacity = 0.5;
            }
            if (this.Equals(algoGui_array[2]))
            {
                this.Location = Screen.AllScreens[1].WorkingArea.Location;
                algoGui_array[2].Height = bids_asks_height + 364;
                algoGui_array[2].Width = bids_asks_width + 170;
                algoGui_array[2].Left += 1576;
                algoGui_array[2].Top += 65;
                algoGui_array[2].Width -= 70;
                algoGui_array[2].Height += 320;
                SetWindowPos(algoGui_array[2].Handle, HWND_TOPMOST, 0, 0, 0, 0, TOPMOST_FLAGS);
                algoGui_array[2].Opacity = 0.5;
            }
            if (this.Equals(algoGui_array[3]))
            {
                this.Location = Screen.AllScreens[1].WorkingArea.Location;
                algoGui_array[3].Height = bids_asks_height;
                algoGui_array[3].Width = bids_asks_width + 100;
                algoGui_array[3].Top += 348;
                algoGui_array[3].Left += 650;
                algoGui_array[3].Width -= 30;
                algoGui_array[3].Height -= 190;
                SetWindowPos(algoGui_array[3].Handle, HWND_TOPMOST, 0, 0, 0, 0, TOPMOST_FLAGS);
                algoGui_array[3].Opacity = 0.5;
            }
            if (this.Equals(algoGui_array[4]))
            {
                algoGui_array[4].Height = bids_asks_height;
                algoGui_array[4].Width = bids_asks_width + 100;
                algoGui_array[4].Left += 455;
                algoGui_array[4].Top += 720;
                algoGui_array[4].Width -= 20;
                algoGui_array[4].Height -= 185;
                SetWindowPos(algoGui_array[4].Handle, HWND_TOPMOST, 0, 0, 0, 0, TOPMOST_FLAGS);
                algoGui_array[4].Opacity = 0.5;
            }
            if (this.Equals(algoGui_array[5]))
            {
                this.Location = Screen.AllScreens[0].WorkingArea.Location;
                Top += 315;
                Left += 545;
                SetWindowPos(algoGui_array[5].Handle, HWND_TOPMOST, 0, 0, 0, 0, TOPMOST_FLAGS);
                t.Elapsed += OnTimedEventPause;
                t.AutoReset = true;
                t.Enabled = false;
                if (File.Exists(@"X:\parsedTensor.txt"))
                {
                    File.Delete(@"X:\parsedTensor.txt");
                }

                for (int i = 0; i < 16; i++)
                {
                    bids.bidsSizeArray[i] = 343;
                    asks.asksSizeArray[i] = 343;
                }
                clusterNodePingCtrl.RunWorkerAsync();
            }
        }

        public void startConsole()
        {
            AllocConsole();
            InitializeComponent();
        }

        [System.Runtime.InteropServices.DllImport("kernel32.dll")]
        private static extern bool AllocConsole();

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);

        private void button3_Click(object sender, EventArgs e)
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            bool bidsOrAsks;

            if (Equals(algoGui_array[0]))
            {
                bidsOrAsks = true;
                Data_Scraper.Program.Level2DataAnalyzer lvlTwo = new Data_Scraper.Program.Level2DataAnalyzer();
                lvlTwo.levelTwoDataRetriever(bidsOrAsks);
                lvlTwo.levelTwoDataExtractor(bids, asks, bidsOrAsks);
            }
            else if (Equals(algoGui_array[2]))
            {
                Data_Scraper.Program.Level2DataAnalyzer lvlTwo = new Data_Scraper.Program.Level2DataAnalyzer();
                lvlTwo.lastTradesDataRetriever();
                lvlTwo.lastTradesDataExtractor(lastTrades);
            }
            else if (Equals(algoGui_array[3]))
            {
                Data_Scraper.Program.Level2DataAnalyzer lvlTwo = new Data_Scraper.Program.Level2DataAnalyzer();
                lvlTwo.dowDataRetriever();
                lvlTwo.dowDataExtractor(dow);

            }
            else if (Equals(algoGui_array[4]))
            {
                Data_Scraper.Program.Level2DataAnalyzer lvlTwo = new Data_Scraper.Program.Level2DataAnalyzer();
                lvlTwo.masterClkRetriever();
                lvlTwo.masterClkExtractor(etradeClk);
            }
            else
            {
                bidsOrAsks = false;
                Data_Scraper.Program.Level2DataAnalyzer lvlTwo = new Data_Scraper.Program.Level2DataAnalyzer();
                lvlTwo.levelTwoDataRetriever(bidsOrAsks);
                lvlTwo.levelTwoDataExtractor(bids, asks, bidsOrAsks);
            }
            
        }

        private static void OnTimedEventPause(object sender, ElapsedEventArgs e)
        {
            bidsCtrl.RunWorkerAsync();
            asksCtrl.RunWorkerAsync();
            lastsCtrl.RunWorkerAsync();
            dowCtrl.RunWorkerAsync();
        }

        private void updateBidsAsks_Click(object sender, EventArgs e)
        {
            this.bidsPrice.Text = "";
            this.bidsSize.Text = "";
            this.asksSize.Text = "";
            this.asksPrice.Text = "";
            this.priceTierBids.Text = "";
            this.priceTierAsks.Text = "";
            algoGui_array[5].Update();
            this.bidsPrice.Text = bids_scrn_output;
            this.bidsSize.Text = bids_size_output;
            this.asksSize.Text = asks_size_output;
            this.asksPrice.Text = asks_scrn_output;
            this.priceTierBids.Text = priceTierBidsOutput;
            this.priceTierAsks.Text = priceTierAsksOutput;
            algoGui_array[5].Update();
        }

        private void updateMstClk_Click(object sender, EventArgs e)
        {
            this.mstClkText.Text = "";
            algoGui_array[5].Update();
            this.mstClkText.Text = mstClkOutput;
            algoGui_array[5].Update();
        }

        private void updateDow_Click(object sender, EventArgs e)
        {
            this.dowTxt.Text = "";
            algoGui_array[5].Update();
            this.dowTxt.Text = dowOutput;
            algoGui_array[5].Update();
        }

        private void updateLasts_Click(object sender, EventArgs e)
        {
            this.lastTradePrice.Text = "";
            this.lastTradeSize.Text = "";
            this.prevTradesPrice.Text = "";
            this.prevTradesSize.Text = "";
            algoGui_array[5].Update();
            this.lastTradePrice.Text = lastTradePriceOutput;
            this.lastTradeSize.Text = lastTradeSizeOutput;
            this.prevTradesPrice.Text = prevTradePriceOutput;
            this.prevTradesSize.Text = prevTradeSizeOutput;
            algoGui_array[5].Update();
        }

        private void exitBtn_Click(object sender, EventArgs e)
        { 
            for (int i = 0; i < 6; i++)
            {
                algoGui_array[i].Close();
            }
            Environment.Exit(1);
        }

        private void fullStop_Click(object sender, EventArgs e)
        {
            stopScraper = true;
            firstStartClick = true;
            //asksCompletedEvent.Set(); //these lines seem to break the pause functionality
            //bidsCompletedEvent.Set(); //taking them out fully restores pause capabilities

            algoGui_array[0].WindowState = FormWindowState.Minimized;
            algoGui_array[1].WindowState = FormWindowState.Minimized;
            algoGui_array[2].WindowState = FormWindowState.Minimized;
            algoGui_array[3].WindowState = FormWindowState.Minimized;
            algoGui_array[4].WindowState = FormWindowState.Minimized;

            //tensor_count = 0;
            //if (File.Exists(@"X:\parsedTensor.txt"))
            //{
            //    File.Delete(@"X:\parsedTensor.txt");
            //}

            if (!executionChecker.IsAlive)
            {
                executionChecker.Start();
            }
            else
            {
                executionChecker.Resume();
            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            this.Opacity = 0.0;
            this.button3.PerformClick();
        }

        private void fullStart_Click(object sender, EventArgs e)
        {
            if (firstStartClick)
            {
                algoGui_array[0].WindowState = FormWindowState.Normal;
                algoGui_array[1].WindowState = FormWindowState.Normal;
                algoGui_array[2].WindowState = FormWindowState.Normal;
                algoGui_array[3].WindowState = FormWindowState.Normal;
                algoGui_array[4].WindowState = FormWindowState.Normal;

                stopScraper = false;
                firstStartClick = false;
                if (executionChecker.IsAlive)
                {
                    executionChecker.Suspend();
                }
                
                if(algoGui_array[5].lvl2Disable.Checked == false)
                {
                    bidsCtrl.RunWorkerAsync();
                    asksCtrl.RunWorkerAsync();
                }
                else
                {
                    algoGui_array[0].Close();
                    algoGui_array[1].Close();
                }
                if (algoGui_array[5].lastTradesDisable.Checked == false)
                {
                    lastsCtrl.RunWorkerAsync();
                }
                else
                {
                    algoGui_array[2].Close();
                }

                if (algoGui_array[5].dowDisable.Checked == false)
                {
                    dowCtrl.RunWorkerAsync();
                }
                else
                {
                    algoGui_array[3].Close();
                }

                if (algoGui_array[5].mstClkDisable.Checked == false)
                {
                    etradeClkCtrl.RunWorkerAsync();
                }
                else
                {
                    algoGui_array[4].Close();
                }
            }
        }

        public static void clusterNodePingCtrl_DoWork(object sender, DoWorkEventArgs e)
        {
            Thread.Sleep(1000);
            try
            {
                var execRequest = new HttpRequestMessage(HttpMethod.Get, "http://" + ipaddress1);
                var task1 = Task.Run(() => pingClient.SendAsync(execRequest));
                task1.Wait();
                var response = task1.Result;
                response.EnsureSuccessStatusCode();
                node1Online = true;
            }
            catch (Exception ex)
            {
                node1Online = false;
            }
            try
            {
                var execRequest = new HttpRequestMessage(HttpMethod.Get, "http://" + ipaddress2);
                var task1 = Task.Run(() => pingClient.SendAsync(execRequest));
                task1.Wait();
                var response = task1.Result;
                response.EnsureSuccessStatusCode();
                node2Online = true;
            }
            catch (Exception ex)
            {
                node2Online = false;
            }
            try
            {
                var execRequest = new HttpRequestMessage(HttpMethod.Get, "http://" + ipaddress3);
                var task1 = Task.Run(() => pingClient.SendAsync(execRequest));
                task1.Wait();
                var response = task1.Result;
                response.EnsureSuccessStatusCode();
                node3Online = true;
            }
            catch (Exception ex)
            {
                node3Online = false;
            }

            if (node1Online == true)
            {
                algoGui.algoGui_array[5].node1Status.Text = "Node 1 Online";
            }
            else
            {
                algoGui.algoGui_array[5].node1Status.Text = "Node 1 Offline";
            }
            if (node2Online == true)
            {
                algoGui.algoGui_array[5].node2Status.Text = "Node 2 Online";
            }
            else
            {
                algoGui.algoGui_array[5].node2Status.Text = "Node 2 Offline";
            }
            if (node3Online == true)
            {
                algoGui.algoGui_array[5].node3Status.Text = "Node 3 Online";
            }
            else
            {
                algoGui.algoGui_array[5].node3Status.Text = "Node 3 Offline";
            }
        }

        public static void clusterNodePingCtrl_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (clusterNodePingCtrl.IsBusy != true)
                clusterNodePingCtrl.RunWorkerAsync();
        }

        public static void bidsCtrl_DoWork(object sender, DoWorkEventArgs e)
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            var watch = Stopwatch.StartNew();
            //StreamWriter parsedData = File.AppendText(@"X:\parsedData.txt");
            //string insertCmd = "INSERT INTO dbo.level2Tensor(price, size) VALUES(@Price, @Size)";
            //SqlCommand cmd = new SqlCommand(insertCmd, conn);
            //cmd.Parameters.Add("@Price", SqlDbType.Float);
            //cmd.Parameters.Add("@Size", SqlDbType.Int);

            int X = Cursor.Position.X;
            int Y = Cursor.Position.Y;
            algoGui_array[5].mouseCoordinates.Text = "Mouse Position:\n" + "X: " + X.ToString() + "   Y: " + Y.ToString();

            DateTime now = DateTime.Now;
            float[] avgs = new float[5];
            int cmpRetVal = 0;

            algoGui_array[1].Invoke((MethodInvoker)delegate
            {
                algoGui_array[1].button4.PerformClick();
            });

            bidsCompletedEvent.WaitOne();
            bidsCompletedEvent.Reset();

            bids_scrn_output = "";
            bids_size_output = "";
            asks_scrn_output = "";
            asks_size_output = "";
            priceTierBidsOutput = "";

            dataAnalyzer analyzer = new dataAnalyzer();
            analyzer.tensorBuild(true);

            bidsCompletedEvent.WaitOne();
            bidsCompletedEvent.Reset();

            //analyzer.bidsAsksErrorCheckerVoter();
            //analyzer.bidsAsksErrorCheckerAverager();
            //bidsCompletedEvent.WaitOne();
            //bidsCompletedEvent.Reset();

            analyzer.midpoint();

            for (int i = 15; i >= 0; i--)
            {
                priceTierBidsOutput += tempTensor.vecTensor[i].vectorPriceTier + "\n";
                bids_scrn_output += tempTensor.vecTensor[i].vectorPrice + "\n";
                bids_size_output += tempTensor.vecTensor[i].vectorSize + "\n";
            }

            for (int i = 0; i < 16; i++)
            {
                asks_scrn_output += tempTensor.vecTensor[i + 16].vectorPrice + "\n";
                asks_size_output += tempTensor.vecTensor[i + 16].vectorSize + "\n";
                //parsedData.WriteLine(bids.bidsArray[i].ToString() + "\t" + bids.bidsSizeArray[i].ToString() + "\t\t" + asks.asksArray[i].ToString() + "\t" + asks.asksSizeArray[i].ToString());
            }

            try
            {
                var execRequest = new HttpRequestMessage(HttpMethod.Get, "http://" + ipaddress3 + "/isrunning");
                var task1 = Task.Run(() => pingClient.SendAsync(execRequest));
                task1.Wait();
                var response = task1.Result;
                response.EnsureSuccessStatusCode();

                var task2 = Task.Run(() => response.Content.ReadAsStringAsync());
                task2.Wait();
                string text = task2.Result;
                text = Data_Scraper.Program.Level2DataAnalyzer.getBetween(text, "{\"isrunning\":\"", "\"}");

                if (text == "TRUE")
                {
                    var webRequest = (HttpWebRequest)WebRequest.Create("http://" + ipaddress3 + "/datauploader");
                    webRequest.ContentType = "application/json";
                    webRequest.Method = "POST";

                    string json = "{\"datalist\": [";
                    for (int i = 0; i < 32; i++)
                    {
                        json += "{\"datavalue\": [\"" + tempTensor.vecTensor[i].vectorPrice.ToString() + "\",\"" + tempTensor.vecTensor[i].vectorSize.ToString() +
                                "\"]},";
                    }
                    json = json.Remove(json.Length - 1, 1);
                    json += "]}";

                    using (var streamWriter = new StreamWriter(webRequest.GetRequestStream()))
                    {
                        streamWriter.Write(json);
                    }

                    var httpResponse = (HttpWebResponse)webRequest.GetResponse();
                    using (var streamReader = new StreamReader(httpResponse.GetResponseStream()))
                    {
                        var result = streamReader.ReadToEnd();
                    }
                }
            }
            catch(Exception ex)
            {

            }

            if (!firstrun)
            {
                cmpRetVal = prevTensor.cmpTensor(prevTensor, tempTensor);
                StreamWriter parsedTensor = File.AppendText(@"X:\parsedTensor.txt");
                if (tensor_count == numEvents)
                {
                    StreamWriter parsedTensor100 = File.AppendText(@"X:\parsedTensorInput.txt");
                    for (int n = 0; n < numEvents; n++)
                    {
                        for (int m = 0; m < 32; m++)
                        {
                            parsedTensor100.WriteLine(tensorArray[n].vecTensor[m].vectorPrice.ToString() + " " + tensorArray[n].vecTensor[m].vectorSize.ToString());
                        }
                    }
                    parsedTensor100.Close();
                    tensor_count = 0;
                    tensorPassed = true;
                }
                if (cmpRetVal == 1 && trade_event != true)
                {
                    for (int k = 0; k < 32; k++)
                    {
                        parsedTensor.WriteLine(tempTensor.vecTensor[k].vectorPrice + " " + tempTensor.vecTensor[k].vectorSize);
                    }
                    tensorArray[tensor_count].copyTensor(tempTensor, tensorArray[tensor_count]);
                    tensor_count++;
                    parsedTensor.Close();
                }
                if (trade_event == true)
                {
                    for (int k = 0; k < 32; k++)
                    {
                        parsedTensor.WriteLine(tempTensor.vecTensor[k].vectorPrice + " " + tempTensor.vecTensor[k].vectorSize);
                    }
                    parsedTensor.Close();
                    tensorArray[tensor_count].copyTensor(tempTensor, tensorArray[tensor_count]);
                    tensor_count++;
                    trade_event = false;
                }
                parsedTensor.Dispose();
            }
            else if (lastTradesScraperRunning == false)
            {
                // something bitch lol...if last trades scraper is turned off we need to change the firstRun bool to false
                firstrun = false;
            }
            eventsItr++;
            
            prevTensor.copyTensor(tempTensor, prevTensor);
            //parsedData.WriteLine(now.ToString("HH:mm:ss"));
            watch.Stop();
            ratesAccum += (double)watch.ElapsedMilliseconds / 1000F;

            asks_size_output += now.ToString("HH:mm:ss.fff\n") + "Rate (s): " + ((double)watch.ElapsedMilliseconds / 1000F).ToString() + "\n" +
                "Avg: (s): " + Math.Round((ratesAccum / eventsItr), 3).ToString() + "\n";
            //parsedData.WriteLine("");
            //parsedData.Close();

            bidsDataScrapeCompletedEvent.Set();

            algoGui_array[5].Invoke((MethodInvoker)delegate
            {
                algoGui_array[5].updateBidsAsks.PerformClick();
            });
        }

        private static void bidsCtrl_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (bidsCtrl.IsBusy != true && stopScraper != true && pre_after_market_mode == false)
            {
                if (tensorPassed)
                {
                    tensorPassed = false;
                    File.Delete(@"X:\parsedTensor.txt");
                }
                bidsCtrl.RunWorkerAsync();
            }
            else if(bidsCtrl.IsBusy != true && stopScraper != true && pre_after_market_mode == true)
            {

            }
        }

        public static void asksCtrl_DoWork(object sender, DoWorkEventArgs e)
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            algoGui_array[0].Invoke((MethodInvoker)delegate
            {
                algoGui_array[0].button4.PerformClick();
            });

            asksCompletedEvent.WaitOne();
            asksCompletedEvent.Reset();

            dataAnalyzer analyzer = new dataAnalyzer();
            priceTierAsksOutput = "";
            analyzer.tensorBuild(false);

            asksCompletedEvent.WaitOne();
            asksCompletedEvent.Reset();

            for (int i = 16; i < 32; i++)
            {
                algoGui.priceTierAsksOutput += algoGui.tempTensor.vecTensor[i].vectorPriceTier + "\n";
            }
        }

        private static void asksCtrl_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (asksCtrl.IsBusy != true && stopScraper != true && pre_after_market_mode == false)
            {
                asksCtrl.RunWorkerAsync();
            }
            else if(asksCtrl.IsBusy != true && stopScraper != true && pre_after_market_mode == true)
            {

            }
        }

        public static void lastsCtrl_DoWork(object sender, DoWorkEventArgs e)
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            DateTime now = DateTime.Now;
            int idx = 0;
            lastTradesScraperRunning = true;

            var watch = Stopwatch.StartNew();

            algoGui_array[2].Invoke((MethodInvoker)delegate
            {
                algoGui_array[2].button4.PerformClick();
            });

            //StreamWriter parsedDataLastTrades = File.AppendText(@"X:\parsedDataLastTrades.txt");
            lastTradeSizeOutput = "";
            lastTradePriceOutput = "";

            if (!firstrun)
            {
                //StreamWriter parsedDataNewTrades = File.AppendText(@"X:\parsedDataNewTrades.txt");

                for (int j = 0; j < 37; j++)
                {
                    idx = 0;
                    for (int k = j; k < 37; k++)
                    {
                        if ((prevTrades.lastTradePriceArray[j] != lastTrades.lastTradePriceArray[k]) && (prevTrades.lastTradeSizeArray[j] != lastTrades.lastTradeSizeArray[k]))
                        {
                            idx++;
                        }
                        else
                        {
                            break;
                        }
                    }
                    if (idx >= tradesCmpIdx)
                    {
                        tradesCmpIdx = idx;
                        idx = 0;
                    }
                }

                if (tradesCmpIdx != 0)
                {
                    prevTradePriceOutput = "";
                    prevTradeSizeOutput = "";

                    for (int l = 0; l < tradesCmpIdx; l++)
                    {
                        if (lastTrades.lastTradePriceArray[l] != 0 && lastTrades.lastTradeSizeArray[l] != 0)
                        {
                            //parsedDataNewTrades.WriteLine(lastTrades.lastTradePriceArray[l].ToString() + "\t" + lastTrades.lastTradeSizeArray[l].ToString());
                            prevTradePriceOutput += lastTrades.lastTradePriceArray[l].ToString() + "\n";
                            prevTradeSizeOutput += lastTrades.lastTradeSizeArray[l].ToString() + "\n";
                        }
                    }
                
                    //parsedDataNewTrades.WriteLine(now.ToString("HH:mm:ss"));
                    prevTradeSizeOutput += now.ToString("HH:mm:ss\n");
                    //parsedDataNewTrades.WriteLine("");
                    trade_event = true;
                }
                //parsedDataNewTrades.Close();
                tradesCmpIdx = 0;
            }
            else
            {
                tradesCmpIdx = 0;
                firstrun = false;
            }

            for (int x = 0; x < 37; x++)
            {
                lastTradePriceOutput += lastTrades.lastTradePriceArray[x].ToString() + "\n";
                lastTradeSizeOutput += lastTrades.lastTradeSizeArray[x].ToString() + "\n";
                //parsedDataLastTrades.WriteLine(lastTrades.lastTradePriceArray[x].ToString() + "\t" + lastTrades.lastTradeSizeArray[x].ToString());
            }
            //parsedDataLastTrades.WriteLine(now.ToString("HH:mm:ss"));
            watch.Stop();
            lastTradeSizeOutput += now.ToString("HH:mm:ss\n") + "Rate (s): " + ((double)watch.ElapsedMilliseconds / 1000F).ToString();
            //parsedDataLastTrades.WriteLine("");
            //parsedDataLastTrades.Close();

            for (int i = 0; i < 37; i++)
            {
                prevTrades.lastTradeSizeArray[i] = lastTrades.lastTradeSizeArray[i];
                prevTrades.lastTradePriceArray[i] = lastTrades.lastTradePriceArray[i];
            }

            past = DateTime.Now;
            algoGui_array[5].Invoke((MethodInvoker)delegate
            {
                algoGui_array[5].updateLasts.PerformClick();
            });
        }

        private static void lastsCtrl_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (lastsCtrl.IsBusy != true && stopScraper != true && pre_after_market_mode == false)
            {
                lastsCtrl.RunWorkerAsync();
            }
            else if(lastsCtrl.IsBusy != true && stopScraper != true && pre_after_market_mode == true)
            {

            }
        }

        public static void dowCtrl_DoWork(object sender, DoWorkEventArgs e)
        {
            DateTime now = DateTime.Now;
            var watch = Stopwatch.StartNew();

            algoGui_array[3].Invoke((MethodInvoker)delegate
            {
                algoGui_array[3].button4.PerformClick();
            });

            //StreamWriter parsedDowData = File.AppendText(@"X:\parsedDataDow.txt");
            //parsedDowData.WriteLine(dow.dowAvg + "\t" + dow.dowAvgSize);
            dowOutput = dow.dowAvg + "   " + dow.dowAvgSize;
            watch.Stop();
            //parsedDowData.WriteLine(now.ToString("HH:mm:ss"));
            dowOutput += now.ToString("      HH:mm:ss.fff") + "    Rate (s): " + ((double)watch.ElapsedMilliseconds / 1000F).ToString();
            //parsedDowData.WriteLine("");
            //parsedDowData.Close();

            algoGui_array[5].Invoke((MethodInvoker)delegate
            {
                algoGui_array[5].updateDow.PerformClick();
            });
        }

        private static void dowCtrl_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (dowCtrl.IsBusy != true && stopScraper != true && pre_after_market_mode == false)
            {
                dowCtrl.RunWorkerAsync();
            }
            else if(dowCtrl.IsBusy != true && stopScraper != true && pre_after_market_mode == true)
            {

            }
        }

        public static void etradeClkCtrl_DoWork(object sender, DoWorkEventArgs e)
        {
            DateTime now = DateTime.Now;
            var watch = Stopwatch.StartNew();

            algoGui_array[4].Invoke((MethodInvoker)delegate
            {
                algoGui_array[4].button4.PerformClick();
            });
            /*
                        if(etradeClk.etradeMstClkHour < 9 && etradeClk.etradeMstClkAMPM.Equals("AM"))
                        {
                            pre_after_market_mode = true;
                            Data_Scraper.algoGui.algoGui_array[5].hrsIndicator.Text = "Pre Market";
                            t.Start();
                        }
                        else if(etradeClk.etradeMstClkHour >= 4 && etradeClk.etradeMstClkMinute >= 0 && etradeClk.etradeMstClkAMPM.Equals("PM"))
                        {
                            pre_after_market_mode = true;
                            Data_Scraper.algoGui.algoGui_array[5].hrsIndicator.Text = "After Market";
                            t.Start();
                        }
                        else
                        {
                            pre_after_market_mode = false;
                            Data_Scraper.algoGui.algoGui_array[5].hrsIndicator.Text = "Trading open";
                            t.Stop();
                        }
            */
            watch.Stop();
            //StreamWriter parsedMstClk = File.AppendText(@"X:\parsedMstClk.txt");
            //parsedMstClk.WriteLine(etradeClk.etradeMstClkHour + ":" + etradeClk.etradeMstClkMinute + ":" + etradeClk.etradeMstClkSecond + " " + etradeClk.etradeMstClkAMPM);
            mstClkOutput = etradeClk.etradeMstClkHour + ":" + etradeClk.etradeMstClkMinute + ":" + etradeClk.etradeMstClkSecond + " " + etradeClk.etradeMstClkAMPM;
            mstClkOutput += now.ToString("      HH:mm:ss.fff") + "    Rate (s): " + ((double)watch.ElapsedMilliseconds / 1000F).ToString();
            //parsedMstClk.WriteLine(now.ToString("HH:mm:ss"));
            //parsedMstClk.WriteLine("");
            //parsedMstClk.Close();

            algoGui_array[5].Invoke((MethodInvoker)delegate
            {
                algoGui_array[5].updateMstClk.PerformClick();
            });
        }

        private static void etradeClkCtrl_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (etradeClkCtrl.IsBusy != true && stopScraper != true)
            {
                etradeClkCtrl.RunWorkerAsync();
            }
        }

        public class ThreadWork
        {
            public static void BidsWork()
            {
                Application.Run(algoGui_array[0]);
            }

            public static void AsksWork()
            {
                Application.Run(algoGui_array[1]);
            }

            public static void LastTradesInit()
            {
                Application.Run(algoGui_array[2]);
            }

            public static void DowInit()
            {
                Application.Run(algoGui_array[3]);
            }

            public static void etradeClk()
            {
                Application.Run(algoGui_array[4]);
            }

            public static void mainAlgoWindowInit()
            {
                Application.Run(algoGui_array[5]);
            }

            public static void executionCheckerCtrl()
            {
                while (true)
                {
                    if (bidsCtrl.IsBusy == false &&
                          asksCtrl.IsBusy == false &&
                          dowCtrl.IsBusy == false &&
                          lastsCtrl.IsBusy == false &&
                          etradeClkCtrl.IsBusy == false &&
                          stopScraper == true)
                    {
                        algoGui_array[0].Opacity = 100.0;
                        algoGui_array[1].Opacity = 100.0;
                        algoGui_array[2].Opacity = 100.0;
                        algoGui_array[3].Opacity = 100.0;
                        algoGui_array[4].Opacity = 100.0;
                    }
                    else
                    {
                        Thread.Sleep(1000);
                    }
                }
            }
        }

        private void synthData_Click(object sender, EventArgs e)
        {
            for (int i = 150; i <= 100000; i++)
            {
                Bitmap synth_data = new Bitmap(1080, 67);
                System.Random rand = new System.Random();

                int price_first_char = rand.Next(0, 9);
                int price_second_char = rand.Next(0, 9);
                int price_third_char = rand.Next(0, 9);
                int price_fourth_char = rand.Next(0, 9);
                int price_fifth_char = rand.Next(0, 9);

                int size_first_char = rand.Next(0, 9);
                int size_second_char = rand.Next(0, 9);
                int size_third_char = rand.Next(0, 9);
                int size_fourth_char = rand.Next(0, 9);

                int charset_idx = rand.Next(0, 3);

                string path1 = "X://eng_level2-charset/" + price_first_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic1 = new Bitmap(path1);

                string path2 = "X://eng_level2-charset/" + price_second_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic2 = new Bitmap(path2);

                string path3 = "X://eng_level2-charset/" + price_third_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic3 = new Bitmap(path3);

                string path4 = "X://eng_level2-charset/" + price_fourth_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic4 = new Bitmap(path4);

                string path5 = "X://eng_level2-charset/" + price_fifth_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic5 = new Bitmap(path5);

                string path6 = "X://eng_level2-charset/" + size_first_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic6 = new Bitmap(path6);

                string path7 = "X://eng_level2-charset/" + size_second_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic7 = new Bitmap(path7);

                string path8 = "X://eng_level2-charset/" + size_third_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic8 = new Bitmap(path8);

                string path9 = "X://eng_level2-charset/" + size_fourth_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic9 = new Bitmap(path9);

                string decimal_path = "X://eng_level2-charset/decimal" + "_" + charset_idx.ToString() + ".jpg";
                Bitmap decimal_pic = new Bitmap(decimal_path);

                int char_idx = 196;

                int choice = rand.Next(0, 6);

                if (choice == 0)
                {
                    using (Graphics grD = Graphics.FromImage(synth_data))
                    {
                        grD.DrawImage(char_pic1, char_idx, 0, char_pic1.Width, char_pic1.Height);
                        char_idx = (char_idx + char_pic1.Width);
                        grD.DrawImage(char_pic2, char_idx, 0, char_pic2.Width, char_pic2.Height);
                        char_idx = (char_idx + char_pic2.Width);
                        grD.DrawImage(decimal_pic, char_idx, 0, decimal_pic.Width, decimal_pic.Height);
                        char_idx = (char_idx + decimal_pic.Width);
                        grD.DrawImage(char_pic3, char_idx, 0, char_pic3.Width, char_pic3.Height);
                        char_idx = (char_idx + char_pic3.Width);
                        grD.DrawImage(char_pic4, char_idx, 0, char_pic4.Width, char_pic4.Height);
                        char_idx = (char_idx + char_pic4.Width + 228);
                        grD.DrawImage(char_pic6, char_idx, 0, char_pic6.Width, char_pic6.Height);
                        char_idx = (char_idx + char_pic6.Width);
                        grD.DrawImage(char_pic7, char_idx, 0, char_pic7.Width, char_pic7.Height);
                        char_idx = (char_idx + char_pic7.Width);
                        grD.DrawImage(char_pic8, char_idx, 0, char_pic8.Width, char_pic8.Height);
                    }

                    synth_data.Save("X://synth_data//eng.bids" + i.ToString() + ".tif");
                    StreamWriter transcription = File.AppendText("X://synth_data//eng.bids" + i.ToString() + ".gt.txt");
                    transcription.Write(price_first_char.ToString() + price_second_char.ToString() + "." + price_third_char.ToString() + price_fourth_char.ToString() + " " +
                        size_first_char.ToString() + size_second_char.ToString() + size_third_char.ToString());
                    transcription.Close();
                }
                else if (choice == 1)
                {
                    using (Graphics grD = Graphics.FromImage(synth_data))
                    {
                        grD.DrawImage(char_pic1, char_idx, 0, char_pic1.Width, char_pic1.Height);
                        char_idx = (char_idx + char_pic1.Width);
                        grD.DrawImage(char_pic2, char_idx, 0, char_pic2.Width, char_pic2.Height);
                        char_idx = (char_idx + char_pic2.Width);
                        grD.DrawImage(char_pic3, char_idx, 0, char_pic3.Width, char_pic3.Height);
                        char_idx = (char_idx + char_pic3.Width);
                        grD.DrawImage(decimal_pic, char_idx, 0, decimal_pic.Width, decimal_pic.Height);
                        char_idx = (char_idx + decimal_pic.Width);
                        grD.DrawImage(char_pic4, char_idx, 0, char_pic4.Width, char_pic4.Height);
                        char_idx = (char_idx + char_pic4.Width);
                        grD.DrawImage(char_pic5, char_idx, 0, char_pic5.Width, char_pic5.Height);
                        char_idx = (char_idx + char_pic5.Width + 194);
                        grD.DrawImage(char_pic6, char_idx, 0, char_pic6.Width, char_pic6.Height);
                        char_idx = (char_idx + char_pic6.Width);
                        grD.DrawImage(char_pic7, char_idx, 0, char_pic7.Width, char_pic7.Height);
                        char_idx = (char_idx + char_pic7.Width);
                        grD.DrawImage(char_pic8, char_idx, 0, char_pic8.Width, char_pic8.Height);
                        char_idx = (char_idx + char_pic8.Width);
                        grD.DrawImage(char_pic9, char_idx, 0, char_pic9.Width, char_pic9.Height);
                        char_idx = (char_idx + char_pic9.Width);
                    }

                    synth_data.Save("X://synth_data//eng.bids" + i.ToString() + ".tif");
                    StreamWriter transcription = File.AppendText("X://synth_data//eng.bids" + i.ToString() + ".gt.txt");
                    transcription.Write(price_first_char.ToString() + price_second_char.ToString() + price_third_char.ToString() + "." + price_fourth_char.ToString() +
                        price_fifth_char.ToString() + " " + size_first_char.ToString() + size_second_char.ToString() + size_third_char.ToString() + size_fourth_char.ToString());
                    transcription.Close();
                }
                else if (choice == 2)
                {
                    using (Graphics grD = Graphics.FromImage(synth_data))
                    {
                        grD.DrawImage(char_pic1, char_idx, 0, char_pic1.Width, char_pic1.Height);
                        char_idx = (char_idx + char_pic1.Width);
                        grD.DrawImage(char_pic2, char_idx, 0, char_pic2.Width, char_pic2.Height);
                        char_idx = (char_idx + char_pic2.Width);
                        grD.DrawImage(char_pic3, char_idx, 0, char_pic3.Width, char_pic3.Height);
                        char_idx = (char_idx + char_pic3.Width);
                        grD.DrawImage(decimal_pic, char_idx, 0, decimal_pic.Width, decimal_pic.Height);
                        char_idx = (char_idx + decimal_pic.Width);
                        grD.DrawImage(char_pic4, char_idx, 0, char_pic4.Width, char_pic4.Height);
                        char_idx = (char_idx + char_pic4.Width);
                        grD.DrawImage(char_pic5, char_idx, 0, char_pic5.Width, char_pic5.Height);
                        char_idx = (char_idx + char_pic5.Width + 194);
                        grD.DrawImage(char_pic6, char_idx, 0, char_pic6.Width, char_pic6.Height);
                        char_idx = (char_idx + char_pic6.Width);
                        grD.DrawImage(char_pic7, char_idx, 0, char_pic7.Width, char_pic7.Height);
                        char_idx = (char_idx + char_pic7.Width);
                        grD.DrawImage(char_pic8, char_idx, 0, char_pic8.Width, char_pic8.Height);
                        char_idx = (char_idx + char_pic8.Width);
                    }

                    synth_data.Save("X://synth_data//eng.bids" + i.ToString() + ".tif");
                    StreamWriter transcription = File.AppendText("X://synth_data//eng.bids" + i.ToString() + ".gt.txt");
                    transcription.Write(price_first_char.ToString() + price_second_char.ToString() + price_third_char.ToString() + "." + price_fourth_char.ToString() +
                        price_fifth_char.ToString() + " " + size_first_char.ToString() + size_second_char.ToString() + size_third_char.ToString());
                    transcription.Close();
                }
                else if (choice == 3)
                {
                    using (Graphics grD = Graphics.FromImage(synth_data))
                    {
                        grD.DrawImage(char_pic1, char_idx, 0, char_pic1.Width, char_pic1.Height);
                        char_idx = (char_idx + char_pic1.Width);
                        grD.DrawImage(char_pic2, char_idx, 0, char_pic2.Width, char_pic2.Height);
                        char_idx = (char_idx + char_pic2.Width);
                        grD.DrawImage(decimal_pic, char_idx, 0, decimal_pic.Width, decimal_pic.Height);
                        char_idx = (char_idx + decimal_pic.Width);
                        grD.DrawImage(char_pic3, char_idx, 0, char_pic3.Width, char_pic3.Height);
                        char_idx = (char_idx + char_pic3.Width);
                        grD.DrawImage(char_pic4, char_idx, 0, char_pic4.Width, char_pic4.Height);
                        char_idx = (char_idx + char_pic4.Width + 228);
                        grD.DrawImage(char_pic6, char_idx, 0, char_pic6.Width, char_pic6.Height);
                        char_idx = (char_idx + char_pic6.Width);
                        grD.DrawImage(char_pic7, char_idx, 0, char_pic7.Width, char_pic7.Height);
                        char_idx = (char_idx + char_pic7.Width);
                        grD.DrawImage(char_pic8, char_idx, 0, char_pic8.Width, char_pic8.Height);
                        char_idx = (char_idx + char_pic8.Width);
                        grD.DrawImage(char_pic9, char_idx, 0, char_pic9.Width, char_pic9.Height);
                        char_idx = (char_idx + char_pic9.Width);
                    }

                    synth_data.Save("X://synth_data//eng.bids" + i.ToString() + ".tif");
                    StreamWriter transcription = File.AppendText("X://synth_data//eng.bids" + i.ToString() + ".gt.txt");
                    transcription.Write(price_first_char.ToString() + price_second_char.ToString() + "." + price_third_char.ToString() + price_fourth_char.ToString() + " " +
                        size_first_char.ToString() + size_second_char.ToString() + size_third_char.ToString() + size_fourth_char.ToString());
                    transcription.Close();
                }
                else if (choice == 4)
                {
                    using (Graphics grD = Graphics.FromImage(synth_data))
                    {
                        grD.DrawImage(char_pic1, char_idx, 0, char_pic1.Width, char_pic1.Height);
                        char_idx = (char_idx + char_pic1.Width);
                        grD.DrawImage(decimal_pic, char_idx, 0, decimal_pic.Width, decimal_pic.Height);
                        char_idx = (char_idx + decimal_pic.Width);
                        grD.DrawImage(char_pic2, char_idx, 0, char_pic2.Width, char_pic2.Height);
                        char_idx = (char_idx + char_pic2.Width);
                        grD.DrawImage(char_pic3, char_idx, 0, char_pic3.Width, char_pic3.Height);
                        char_idx = (char_idx + char_pic5.Width + 303);
                        grD.DrawImage(char_pic6, char_idx, 0, char_pic6.Width, char_pic6.Height);
                        char_idx = (char_idx + char_pic6.Width);
                        grD.DrawImage(char_pic7, char_idx, 0, char_pic7.Width, char_pic7.Height);
                        char_idx = (char_idx + char_pic7.Width);
                        grD.DrawImage(char_pic8, char_idx, 0, char_pic8.Width, char_pic8.Height);
                        char_idx = (char_idx + char_pic8.Width);
                        grD.DrawImage(char_pic9, char_idx, 0, char_pic9.Width, char_pic9.Height);
                        char_idx = (char_idx + char_pic9.Width);
                    }

                    synth_data.Save("X://synth_data//eng.bids" + i.ToString() + ".tif");
                    StreamWriter transcription = File.AppendText("X://synth_data//eng.bids" + i.ToString() + ".gt.txt");
                    transcription.Write(price_first_char.ToString() + "." + price_second_char.ToString() + price_third_char.ToString() + " " +
                        size_first_char.ToString() + size_second_char.ToString() + size_third_char.ToString() + size_fourth_char.ToString());
                    transcription.Close();
                }
                else if (choice == 5)
                {
                    using (Graphics grD = Graphics.FromImage(synth_data))
                    {
                        grD.DrawImage(char_pic1, char_idx, 0, char_pic1.Width, char_pic1.Height);
                        char_idx = (char_idx + char_pic1.Width);
                        grD.DrawImage(decimal_pic, char_idx, 0, decimal_pic.Width, decimal_pic.Height);
                        char_idx = (char_idx + decimal_pic.Width);
                        grD.DrawImage(char_pic2, char_idx, 0, char_pic2.Width, char_pic2.Height);
                        char_idx = (char_idx + char_pic2.Width);
                        grD.DrawImage(char_pic3, char_idx, 0, char_pic3.Width, char_pic3.Height);
                        char_idx = (char_idx + char_pic5.Width + 303);
                        grD.DrawImage(char_pic6, char_idx, 0, char_pic6.Width, char_pic6.Height);
                        char_idx = (char_idx + char_pic6.Width);
                        grD.DrawImage(char_pic7, char_idx, 0, char_pic7.Width, char_pic7.Height);
                        char_idx = (char_idx + char_pic7.Width);
                        grD.DrawImage(char_pic8, char_idx, 0, char_pic8.Width, char_pic8.Height);
                        char_idx = (char_idx + char_pic8.Width);
                    }

                    synth_data.Save("X://synth_data//eng.bids" + i.ToString() + ".tif");
                    StreamWriter transcription = File.AppendText("X://synth_data//eng.bids" + i.ToString() + ".gt.txt");
                    transcription.Write(price_first_char.ToString() + "." + price_second_char.ToString() + price_third_char.ToString() + " " +
                        size_first_char.ToString() + size_second_char.ToString() + size_third_char.ToString());
                    transcription.Close();
                }
                Thread.Sleep(100);
            }
/*
            for (int i = 127; i < 2000; i++)
            {
                Bitmap synth_data = new Bitmap(1080, 67);
                System.Random rand = new System.Random();

                int price_first_char = rand.Next(0, 9);
                int price_second_char = rand.Next(0, 9);
                int price_third_char = rand.Next(0, 9);
                int price_fourth_char = rand.Next(0, 9);
                int price_fifth_char = rand.Next(0, 9);
                int price_sixth_char = rand.Next(0, 9);

                int charset_idx = rand.Next(0, 3);

                string path1 = "X://eng_level2-charset/" + price_first_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic1 = new Bitmap(path1);

                string path2 = "X://eng_level2-charset/" + price_second_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic2 = new Bitmap(path2);

                string path3 = "X://eng_level2-charset/" + price_third_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic3 = new Bitmap(path3);

                string path4 = "X://eng_level2-charset/" + price_fourth_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic4 = new Bitmap(path4);

                string path5 = "X://eng_level2-charset/" + price_fifth_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic5 = new Bitmap(path5);

                string path6 = "X://eng_level2-charset/" + price_sixth_char.ToString() + "_" + charset_idx.ToString() + ".jpg";
                Bitmap char_pic6 = new Bitmap(path6);

                string decimal_path = "X://eng_level2-charset/decimal" + "_" + charset_idx.ToString() + ".jpg";
                Bitmap decimal_pic = new Bitmap(decimal_path);

                string colon_path = "X://eng_level2-charset/colon" + "_" + charset_idx.ToString() + ".jpg";
                Bitmap colon_pic = new Bitmap(colon_path);

                int char_idx = 196;

                using (Graphics grD = Graphics.FromImage(synth_data))
                {
                    grD.DrawImage(char_pic1, char_idx, 0, char_pic1.Width, char_pic1.Height);
                    char_idx = (char_idx + char_pic1.Width);
                    grD.DrawImage(char_pic2, char_idx, 0, char_pic2.Width, char_pic2.Height);
                    char_idx = (char_idx + char_pic2.Width);
                    grD.DrawImage(colon_pic, char_idx, 0, colon_pic.Width, colon_pic.Height);
                    char_idx = (char_idx + colon_pic.Width);
                    grD.DrawImage(char_pic3, char_idx, 0, char_pic3.Width, char_pic3.Height);
                    char_idx = (char_idx + char_pic3.Width);
                    grD.DrawImage(char_pic4, char_idx, 0, char_pic4.Width, char_pic4.Height);
                    char_idx = (char_idx + char_pic4.Width);
                    grD.DrawImage(colon_pic, char_idx, 0, colon_pic.Width, colon_pic.Height);
                    char_idx = (char_idx + colon_pic.Width);
                    grD.DrawImage(char_pic5, char_idx, 0, char_pic5.Width, char_pic5.Height);
                    char_idx = (char_idx + char_pic6.Width);
                    grD.DrawImage(char_pic6, char_idx, 0, char_pic6.Width, char_pic6.Height);
                }

                synth_data.Save("X://synth_data//eng.lasts" + i.ToString() + ".tif");
                StreamWriter transcription = File.AppendText("X://synth_data//eng.lasts" + i.ToString() + ".gt.txt");
                transcription.Write(price_first_char.ToString() + price_second_char.ToString() + ":" + price_third_char.ToString() + price_fourth_char.ToString() + ":" +
                    price_fifth_char.ToString() + price_sixth_char.ToString());
                transcription.Close();
                Thread.Sleep(500);
            }
            */
        }

        private void predictorStart_Click(object sender, EventArgs e)
        {
            Process[] process_Array = Process.GetProcessesByName("Predictor");
            if (process_Array.Length == 0)
            {
                Process.Start(predictorProcess);
            }
        }

        private void testMouseAction_Click(object sender, EventArgs e)
        {
            //needs the data scraper to be started as administrator in order to work to click buttons on trading app.
            Point cursorPos = new Point(87, 41); //new Point(319, 289);
            Cursor.Position = cursorPos;
            Thread.Sleep(500);
            mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, cursorPos.X, cursorPos.Y, 0, 0);

            cursorPos.X = 319;
            cursorPos.Y = 289;
            Cursor.Position = cursorPos;

            //-1049, 214

            Thread.Sleep(500);
            mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, cursorPos.X, cursorPos.Y, 0, 0);

            //Bitmap main = new Bitmap(@"X:\entire_screen.jpg");
            //Bitmap sub = new Bitmap(@"X:\market_depth_window_title.jpg");

            //List<Point> points = GetSubPositions(main, sub);
            //StreamWriter writer = File.AppendText(@"X:\points.txt");
            //foreach (Point point in points)
            //{
            //    writer.WriteLine("X: " + point.X.ToString() + "  Y: " + point.Y.ToString() + "\n");
            //}
            //writer.Close();
        }

        public static List<Point> GetSubPositions(Bitmap main, Bitmap sub)
        {
            List<Point> possiblepos = new List<Point>();

            int mainwidth = main.Width;
            int mainheight = main.Height;

            int subwidth = sub.Width;
            int subheight = sub.Height;

            int movewidth = mainwidth - subwidth;
            int moveheight = mainheight - subheight;

            BitmapData bmMainData = main.LockBits(new Rectangle(0, 0, mainwidth, mainheight), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            BitmapData bmSubData = sub.LockBits(new Rectangle(0, 0, subwidth, subheight), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);

            int bytesMain = Math.Abs(bmMainData.Stride) * mainheight;
            int strideMain = bmMainData.Stride;
            System.IntPtr Scan0Main = bmMainData.Scan0;
            byte[] dataMain = new byte[bytesMain];
            System.Runtime.InteropServices.Marshal.Copy(Scan0Main, dataMain, 0, bytesMain);

            int bytesSub = Math.Abs(bmSubData.Stride) * subheight;
            int strideSub = bmSubData.Stride;
            System.IntPtr Scan0Sub = bmSubData.Scan0;
            byte[] dataSub = new byte[bytesSub];
            System.Runtime.InteropServices.Marshal.Copy(Scan0Sub, dataSub, 0, bytesSub);

            for (int y = 0; y < moveheight; ++y)
            {
                for (int x = 0; x < movewidth; ++x)
                {
                    MyColor curcolor = GetColor(x, y, strideMain, dataMain);

                    foreach (var item in possiblepos.ToArray())
                    {
                        int xsub = x - item.X;
                        int ysub = y - item.Y;
                        if (xsub >= subwidth || ysub >= subheight || xsub < 0)
                            continue;

                        MyColor subcolor = GetColor(xsub, ysub, strideSub, dataSub);

                        if (!curcolor.Equals(subcolor))
                        {
                            possiblepos.Remove(item);
                        }
                    }

                    if (curcolor.Equals(GetColor(0, 0, strideSub, dataSub)))
                        possiblepos.Add(new Point(x, y));
                }
            }

            System.Runtime.InteropServices.Marshal.Copy(dataSub, 0, Scan0Sub, bytesSub);
            sub.UnlockBits(bmSubData);

            System.Runtime.InteropServices.Marshal.Copy(dataMain, 0, Scan0Main, bytesMain);
            main.UnlockBits(bmMainData);

            return possiblepos;
        }

        private static MyColor GetColor(Point point, int stride, byte[] data)
        {
            return GetColor(point.X, point.Y, stride, data);
        }

        private static MyColor GetColor(int x, int y, int stride, byte[] data)
        {
            int pos = y * stride + x * 4;
            byte a = data[pos + 3];
            byte r = data[pos + 2];
            byte g = data[pos + 1];
            byte b = data[pos + 0];
            return MyColor.FromARGB(a, r, g, b);
        }

        struct MyColor
        {
            byte A;
            byte R;
            byte G;
            byte B;

            public static MyColor FromARGB(byte a, byte r, byte g, byte b)
            {
                MyColor mc = new MyColor();
                mc.A = a;
                mc.R = r;
                mc.G = g;
                mc.B = b;
                return mc;
            }

            public override bool Equals(object obj)
            {
                if (!(obj is MyColor))
                    return false;
                MyColor color = (MyColor)obj;
                if (color.A == this.A && color.R == this.R && color.G == this.G && color.B == this.B)
                    return true;
                return false;
            }
        }

        private void stockSelecter_SelectedIndexChanged(object sender, EventArgs e)
        {
            Point cursorPos = new Point(-1034, 75); //new Point(319, 289);
            Cursor.Position = cursorPos;

            if (stockSelecter.Text == "NVDA")
            {
                Thread.Sleep(500);
                mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, cursorPos.X, cursorPos.Y, 0, 0);

                SendKeys.SendWait("{BKSP}");
                Thread.Sleep(500);
                SendKeys.SendWait("NVDA");
                Thread.Sleep(500);
                SendKeys.SendWait("{ENTER}");
            }
            else if(stockSelecter.Text == "AMD")
            {
                Thread.Sleep(500);
                mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, cursorPos.X, cursorPos.Y, 0, 0);

                SendKeys.SendWait("{BKSP}");
                Thread.Sleep(500);
                SendKeys.SendWait("AMD");
                Thread.Sleep(500);
                SendKeys.SendWait("{ENTER}");
            }

            cursorPos.X = 2519;
            cursorPos.Y = 844;
            Cursor.Position = cursorPos;
        }
    }
}
