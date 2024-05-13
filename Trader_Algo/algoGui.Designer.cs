/*
* Copyright 2021 - Tim Prishtina, and Luke Koch
*
* All rights reserved. No part of this software may be re-produced, re-engineered, 
* re-compiled, modified, used to create derivatives, stored in a retrieval system, 
* or transmitted in any form or by any means, whether electronic, mechanical, 
* photocopying, recording, or otherwise, without the prior written permission of 
* Tim Prishtina, and Luke Koch.
*/

namespace Data_Scraper
{
    partial class algoGui
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.button3 = new System.Windows.Forms.Button();
            this.button4 = new System.Windows.Forms.Button();
            this.fullStart = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.updateBidsAsks = new System.Windows.Forms.Button();
            this.bidsPrice = new System.Windows.Forms.Label();
            this.bidsSize = new System.Windows.Forms.Label();
            this.asksSize = new System.Windows.Forms.Label();
            this.asksPrice = new System.Windows.Forms.Label();
            this.updateMstClk = new System.Windows.Forms.Button();
            this.updateDow = new System.Windows.Forms.Button();
            this.lastTradeSize = new System.Windows.Forms.Label();
            this.lastTradePrice = new System.Windows.Forms.Label();
            this.updateLasts = new System.Windows.Forms.Button();
            this.exitBtn = new System.Windows.Forms.Button();
            this.controlBtns = new System.Windows.Forms.GroupBox();
            this.label5 = new System.Windows.Forms.Label();
            this.mstClkText = new System.Windows.Forms.Label();
            this.dowTxt = new System.Windows.Forms.Label();
            this.fullStop = new System.Windows.Forms.Button();
            this.hrsIndicator = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.prevTradesSize = new System.Windows.Forms.Label();
            this.prevTradesPrice = new System.Windows.Forms.Label();
            this.priceTierBids = new System.Windows.Forms.Label();
            this.priceTierAsks = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.synthData = new System.Windows.Forms.Button();
            this.lastTradesDisable = new System.Windows.Forms.CheckBox();
            this.dowDisable = new System.Windows.Forms.CheckBox();
            this.mstClkDisable = new System.Windows.Forms.CheckBox();
            this.predictorStart = new System.Windows.Forms.Button();
            this.testMouseAction = new System.Windows.Forms.Button();
            this.mouseCoordinates = new System.Windows.Forms.Label();
            this.stockSelecter = new System.Windows.Forms.ComboBox();
            this.label8 = new System.Windows.Forms.Label();
            this.eblCluster = new System.Windows.Forms.CheckBox();
            this.lvl2Disable = new System.Windows.Forms.CheckBox();
            this.clusterNodesOnline = new System.Windows.Forms.Label();
            this.node1Status = new System.Windows.Forms.Label();
            this.node2Status = new System.Windows.Forms.Label();
            this.node3Status = new System.Windows.Forms.Label();
            this.nodeIPUpdater = new System.Windows.Forms.TextBox();
            this.node1IP = new System.Windows.Forms.Button();
            this.node2IP = new System.Windows.Forms.Button();
            this.node3IP = new System.Windows.Forms.Button();
            this.controlBtns.SuspendLayout();
            this.SuspendLayout();
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(6, 13);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(110, 25);
            this.button3.TabIndex = 2;
            this.button3.Text = "button3";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // button4
            // 
            this.button4.Location = new System.Drawing.Point(6, 44);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(110, 30);
            this.button4.TabIndex = 3;
            this.button4.Text = "button4";
            this.button4.UseVisualStyleBackColor = true;
            this.button4.Click += new System.EventHandler(this.button4_Click);
            // 
            // fullStart
            // 
            this.fullStart.Location = new System.Drawing.Point(312, 583);
            this.fullStart.Name = "fullStart";
            this.fullStart.Size = new System.Drawing.Size(53, 44);
            this.fullStart.TabIndex = 4;
            this.fullStart.Text = "Full Start";
            this.fullStart.UseVisualStyleBackColor = true;
            this.fullStart.Click += new System.EventHandler(this.fullStart_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(262, 50);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(44, 20);
            this.label1.TabIndex = 10;
            this.label1.Text = "Bids";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(609, 50);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(48, 20);
            this.label2.TabIndex = 11;
            this.label2.Text = "Asks";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.Location = new System.Drawing.Point(1094, 9);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(104, 20);
            this.label3.TabIndex = 12;
            this.label3.Text = "Last Trades";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.Location = new System.Drawing.Point(130, 506);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(114, 20);
            this.label4.TabIndex = 13;
            this.label4.Text = "Current Dow:";
            // 
            // updateBidsAsks
            // 
            this.updateBidsAsks.Location = new System.Drawing.Point(6, 80);
            this.updateBidsAsks.Name = "updateBidsAsks";
            this.updateBidsAsks.Size = new System.Drawing.Size(110, 35);
            this.updateBidsAsks.TabIndex = 15;
            this.updateBidsAsks.Text = "Update bids/asks";
            this.updateBidsAsks.UseVisualStyleBackColor = true;
            this.updateBidsAsks.Click += new System.EventHandler(this.updateBidsAsks_Click);
            // 
            // bidsPrice
            // 
            this.bidsPrice.AutoSize = true;
            this.bidsPrice.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.bidsPrice.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.bidsPrice.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.bidsPrice.Location = new System.Drawing.Point(201, 80);
            this.bidsPrice.Name = "bidsPrice";
            this.bidsPrice.Size = new System.Drawing.Size(0, 20);
            this.bidsPrice.TabIndex = 16;
            // 
            // bidsSize
            // 
            this.bidsSize.AutoSize = true;
            this.bidsSize.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.bidsSize.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.bidsSize.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.bidsSize.Location = new System.Drawing.Point(341, 80);
            this.bidsSize.Name = "bidsSize";
            this.bidsSize.Size = new System.Drawing.Size(0, 20);
            this.bidsSize.TabIndex = 17;
            // 
            // asksSize
            // 
            this.asksSize.AutoSize = true;
            this.asksSize.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.asksSize.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.asksSize.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.asksSize.Location = new System.Drawing.Point(670, 80);
            this.asksSize.Name = "asksSize";
            this.asksSize.Size = new System.Drawing.Size(0, 20);
            this.asksSize.TabIndex = 19;
            // 
            // asksPrice
            // 
            this.asksPrice.AutoSize = true;
            this.asksPrice.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.asksPrice.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.asksPrice.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.asksPrice.Location = new System.Drawing.Point(540, 80);
            this.asksPrice.Name = "asksPrice";
            this.asksPrice.Size = new System.Drawing.Size(0, 20);
            this.asksPrice.TabIndex = 18;
            // 
            // updateMstClk
            // 
            this.updateMstClk.Location = new System.Drawing.Point(6, 121);
            this.updateMstClk.Name = "updateMstClk";
            this.updateMstClk.Size = new System.Drawing.Size(110, 30);
            this.updateMstClk.TabIndex = 20;
            this.updateMstClk.Text = "Update Mst Clk";
            this.updateMstClk.UseVisualStyleBackColor = true;
            this.updateMstClk.Click += new System.EventHandler(this.updateMstClk_Click);
            // 
            // updateDow
            // 
            this.updateDow.Location = new System.Drawing.Point(6, 157);
            this.updateDow.Name = "updateDow";
            this.updateDow.Size = new System.Drawing.Size(110, 30);
            this.updateDow.TabIndex = 22;
            this.updateDow.Text = "Update Dow";
            this.updateDow.UseVisualStyleBackColor = true;
            this.updateDow.Click += new System.EventHandler(this.updateDow_Click);
            // 
            // lastTradeSize
            // 
            this.lastTradeSize.AutoSize = true;
            this.lastTradeSize.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.lastTradeSize.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lastTradeSize.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lastTradeSize.Location = new System.Drawing.Point(1198, 35);
            this.lastTradeSize.Name = "lastTradeSize";
            this.lastTradeSize.Size = new System.Drawing.Size(0, 15);
            this.lastTradeSize.TabIndex = 25;
            // 
            // lastTradePrice
            // 
            this.lastTradePrice.AutoSize = true;
            this.lastTradePrice.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.lastTradePrice.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lastTradePrice.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lastTradePrice.Location = new System.Drawing.Point(1068, 35);
            this.lastTradePrice.Name = "lastTradePrice";
            this.lastTradePrice.Size = new System.Drawing.Size(0, 15);
            this.lastTradePrice.TabIndex = 24;
            // 
            // updateLasts
            // 
            this.updateLasts.Location = new System.Drawing.Point(6, 193);
            this.updateLasts.Name = "updateLasts";
            this.updateLasts.Size = new System.Drawing.Size(110, 30);
            this.updateLasts.TabIndex = 26;
            this.updateLasts.Text = "Update Last Trades";
            this.updateLasts.UseVisualStyleBackColor = true;
            this.updateLasts.Click += new System.EventHandler(this.updateLasts_Click);
            // 
            // exitBtn
            // 
            this.exitBtn.Location = new System.Drawing.Point(251, 583);
            this.exitBtn.Name = "exitBtn";
            this.exitBtn.Size = new System.Drawing.Size(55, 43);
            this.exitBtn.TabIndex = 27;
            this.exitBtn.Text = "Exit";
            this.exitBtn.UseVisualStyleBackColor = true;
            this.exitBtn.Click += new System.EventHandler(this.exitBtn_Click);
            // 
            // controlBtns
            // 
            this.controlBtns.Controls.Add(this.button3);
            this.controlBtns.Controls.Add(this.button4);
            this.controlBtns.Controls.Add(this.updateLasts);
            this.controlBtns.Controls.Add(this.updateBidsAsks);
            this.controlBtns.Controls.Add(this.updateMstClk);
            this.controlBtns.Controls.Add(this.updateDow);
            this.controlBtns.Location = new System.Drawing.Point(12, 645);
            this.controlBtns.Name = "controlBtns";
            this.controlBtns.Size = new System.Drawing.Size(0, 0);
            this.controlBtns.TabIndex = 28;
            this.controlBtns.TabStop = false;
            this.controlBtns.Text = "Control Buttons";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label5.Location = new System.Drawing.Point(171, 544);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(73, 20);
            this.label5.TabIndex = 14;
            this.label5.Text = "Mst Clk:";
            // 
            // mstClkText
            // 
            this.mstClkText.AutoSize = true;
            this.mstClkText.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.mstClkText.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.mstClkText.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.mstClkText.Location = new System.Drawing.Point(250, 544);
            this.mstClkText.Name = "mstClkText";
            this.mstClkText.Size = new System.Drawing.Size(0, 20);
            this.mstClkText.TabIndex = 21;
            // 
            // dowTxt
            // 
            this.dowTxt.AutoSize = true;
            this.dowTxt.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.dowTxt.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.dowTxt.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.dowTxt.Location = new System.Drawing.Point(250, 506);
            this.dowTxt.Name = "dowTxt";
            this.dowTxt.Size = new System.Drawing.Size(0, 20);
            this.dowTxt.TabIndex = 23;
            // 
            // fullStop
            // 
            this.fullStop.Location = new System.Drawing.Point(371, 583);
            this.fullStop.Name = "fullStop";
            this.fullStop.Size = new System.Drawing.Size(56, 44);
            this.fullStop.TabIndex = 29;
            this.fullStop.Text = "Full Stop";
            this.fullStop.UseVisualStyleBackColor = true;
            this.fullStop.Click += new System.EventHandler(this.fullStop_Click);
            // 
            // hrsIndicator
            // 
            this.hrsIndicator.AutoSize = true;
            this.hrsIndicator.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.hrsIndicator.Location = new System.Drawing.Point(306, 30);
            this.hrsIndicator.Name = "hrsIndicator";
            this.hrsIndicator.Size = new System.Drawing.Size(0, 20);
            this.hrsIndicator.TabIndex = 30;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label6.Location = new System.Drawing.Point(826, 9);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(128, 20);
            this.label6.TabIndex = 31;
            this.label6.Text = "Newest Trades";
            // 
            // prevTradesSize
            // 
            this.prevTradesSize.AutoSize = true;
            this.prevTradesSize.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.prevTradesSize.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.prevTradesSize.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.prevTradesSize.Location = new System.Drawing.Point(943, 35);
            this.prevTradesSize.Name = "prevTradesSize";
            this.prevTradesSize.Size = new System.Drawing.Size(0, 15);
            this.prevTradesSize.TabIndex = 33;
            // 
            // prevTradesPrice
            // 
            this.prevTradesPrice.AutoSize = true;
            this.prevTradesPrice.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.prevTradesPrice.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.prevTradesPrice.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.prevTradesPrice.Location = new System.Drawing.Point(813, 35);
            this.prevTradesPrice.Name = "prevTradesPrice";
            this.prevTradesPrice.Size = new System.Drawing.Size(0, 15);
            this.prevTradesPrice.TabIndex = 32;
            // 
            // priceTierBids
            // 
            this.priceTierBids.AutoSize = true;
            this.priceTierBids.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.priceTierBids.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.priceTierBids.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.priceTierBids.Location = new System.Drawing.Point(100, 80);
            this.priceTierBids.Name = "priceTierBids";
            this.priceTierBids.Size = new System.Drawing.Size(0, 20);
            this.priceTierBids.TabIndex = 34;
            // 
            // priceTierAsks
            // 
            this.priceTierAsks.AutoSize = true;
            this.priceTierAsks.Cursor = System.Windows.Forms.Cursors.IBeam;
            this.priceTierAsks.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.priceTierAsks.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.priceTierAsks.Location = new System.Drawing.Point(443, 80);
            this.priceTierAsks.Name = "priceTierAsks";
            this.priceTierAsks.Size = new System.Drawing.Size(0, 20);
            this.priceTierAsks.TabIndex = 35;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(309, 9);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(153, 13);
            this.label7.TabIndex = 36;
            this.label7.Text = "Midpoint              Events Count";
            // 
            // synthData
            // 
            this.synthData.Location = new System.Drawing.Point(506, 583);
            this.synthData.Name = "synthData";
            this.synthData.Size = new System.Drawing.Size(85, 44);
            this.synthData.TabIndex = 37;
            this.synthData.Text = "Synthesize Training Data";
            this.synthData.UseVisualStyleBackColor = true;
            this.synthData.Click += new System.EventHandler(this.synthData_Click);
            // 
            // lastTradesDisable
            // 
            this.lastTradesDisable.AutoSize = true;
            this.lastTradesDisable.Location = new System.Drawing.Point(12, 574);
            this.lastTradesDisable.Name = "lastTradesDisable";
            this.lastTradesDisable.Size = new System.Drawing.Size(160, 17);
            this.lastTradesDisable.TabIndex = 38;
            this.lastTradesDisable.Text = "Last Trades Scraper Disable";
            this.lastTradesDisable.UseVisualStyleBackColor = true;
            // 
            // dowDisable
            // 
            this.dowDisable.AutoSize = true;
            this.dowDisable.Location = new System.Drawing.Point(12, 597);
            this.dowDisable.Name = "dowDisable";
            this.dowDisable.Size = new System.Drawing.Size(163, 17);
            this.dowDisable.TabIndex = 39;
            this.dowDisable.Text = "Current Dow Scraper Disable";
            this.dowDisable.UseVisualStyleBackColor = true;
            // 
            // mstClkDisable
            // 
            this.mstClkDisable.AutoSize = true;
            this.mstClkDisable.Location = new System.Drawing.Point(12, 620);
            this.mstClkDisable.Name = "mstClkDisable";
            this.mstClkDisable.Size = new System.Drawing.Size(166, 17);
            this.mstClkDisable.TabIndex = 40;
            this.mstClkDisable.Text = "Master Clock Scraper Disable";
            this.mstClkDisable.UseVisualStyleBackColor = true;
            // 
            // predictorStart
            // 
            this.predictorStart.Location = new System.Drawing.Point(433, 583);
            this.predictorStart.Name = "predictorStart";
            this.predictorStart.Size = new System.Drawing.Size(67, 44);
            this.predictorStart.TabIndex = 41;
            this.predictorStart.Text = "Start Predictor";
            this.predictorStart.UseVisualStyleBackColor = true;
            this.predictorStart.Click += new System.EventHandler(this.predictorStart_Click);
            // 
            // testMouseAction
            // 
            this.testMouseAction.Location = new System.Drawing.Point(597, 583);
            this.testMouseAction.Name = "testMouseAction";
            this.testMouseAction.Size = new System.Drawing.Size(64, 44);
            this.testMouseAction.TabIndex = 42;
            this.testMouseAction.Text = "Mouse Action";
            this.testMouseAction.UseVisualStyleBackColor = true;
            this.testMouseAction.Click += new System.EventHandler(this.testMouseAction_Click);
            // 
            // mouseCoordinates
            // 
            this.mouseCoordinates.AutoSize = true;
            this.mouseCoordinates.Location = new System.Drawing.Point(14, 16);
            this.mouseCoordinates.Name = "mouseCoordinates";
            this.mouseCoordinates.Size = new System.Drawing.Size(0, 13);
            this.mouseCoordinates.TabIndex = 43;
            // 
            // stockSelecter
            // 
            this.stockSelecter.FormattingEnabled = true;
            this.stockSelecter.Items.AddRange(new object[] {
            "NVDA",
            "AMD",
            "MCHP",
            "PFE"});
            this.stockSelecter.Location = new System.Drawing.Point(10, 476);
            this.stockSelecter.Name = "stockSelecter";
            this.stockSelecter.Size = new System.Drawing.Size(118, 21);
            this.stockSelecter.TabIndex = 44;
            this.stockSelecter.SelectedIndexChanged += new System.EventHandler(this.stockSelecter_SelectedIndexChanged);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(9, 455);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(104, 13);
            this.label8.TabIndex = 45;
            this.label8.Text = "Stock being viewed:";
            // 
            // eblCluster
            // 
            this.eblCluster.AutoSize = true;
            this.eblCluster.Checked = true;
            this.eblCluster.CheckState = System.Windows.Forms.CheckState.Checked;
            this.eblCluster.Location = new System.Drawing.Point(12, 506);
            this.eblCluster.Name = "eblCluster";
            this.eblCluster.Size = new System.Drawing.Size(121, 30);
            this.eblCluster.TabIndex = 46;
            this.eblCluster.Text = "Enable Supercluster\r\nProcessing";
            this.eblCluster.UseVisualStyleBackColor = true;
            // 
            // lvl2Disable
            // 
            this.lvl2Disable.AutoSize = true;
            this.lvl2Disable.Location = new System.Drawing.Point(12, 551);
            this.lvl2Disable.Name = "lvl2Disable";
            this.lvl2Disable.Size = new System.Drawing.Size(139, 17);
            this.lvl2Disable.TabIndex = 47;
            this.lvl2Disable.Text = "Level 2 Scraper Disable";
            this.lvl2Disable.UseVisualStyleBackColor = true;
            // 
            // clusterNodesOnline
            // 
            this.clusterNodesOnline.AutoSize = true;
            this.clusterNodesOnline.Location = new System.Drawing.Point(12, 642);
            this.clusterNodesOnline.Name = "clusterNodesOnline";
            this.clusterNodesOnline.Size = new System.Drawing.Size(139, 13);
            this.clusterNodesOnline.TabIndex = 48;
            this.clusterNodesOnline.Text = "Supercluster Nodes Online: ";
            // 
            // node1Status
            // 
            this.node1Status.AutoSize = true;
            this.node1Status.Location = new System.Drawing.Point(147, 642);
            this.node1Status.Name = "node1Status";
            this.node1Status.Size = new System.Drawing.Size(0, 13);
            this.node1Status.TabIndex = 49;
            // 
            // node2Status
            // 
            this.node2Status.AutoSize = true;
            this.node2Status.Location = new System.Drawing.Point(247, 642);
            this.node2Status.Name = "node2Status";
            this.node2Status.Size = new System.Drawing.Size(0, 13);
            this.node2Status.TabIndex = 50;
            // 
            // node3Status
            // 
            this.node3Status.AutoSize = true;
            this.node3Status.Location = new System.Drawing.Point(342, 642);
            this.node3Status.Name = "node3Status";
            this.node3Status.Size = new System.Drawing.Size(0, 13);
            this.node3Status.TabIndex = 51;
            // 
            // nodeIPUpdater
            // 
            this.nodeIPUpdater.Location = new System.Drawing.Point(674, 635);
            this.nodeIPUpdater.Name = "nodeIPUpdater";
            this.nodeIPUpdater.Size = new System.Drawing.Size(120, 20);
            this.nodeIPUpdater.TabIndex = 52;
            // 
            // node1IP
            // 
            this.node1IP.Location = new System.Drawing.Point(800, 633);
            this.node1IP.Name = "node1IP";
            this.node1IP.Size = new System.Drawing.Size(75, 22);
            this.node1IP.TabIndex = 53;
            this.node1IP.Text = "Node 1 IP";
            this.node1IP.UseVisualStyleBackColor = true;
            this.node1IP.Click += new System.EventHandler(this.node1IP_Click);
            // 
            // node2IP
            // 
            this.node2IP.Location = new System.Drawing.Point(879, 633);
            this.node2IP.Name = "node2IP";
            this.node2IP.Size = new System.Drawing.Size(75, 22);
            this.node2IP.TabIndex = 54;
            this.node2IP.Text = "Node 2 IP";
            this.node2IP.UseVisualStyleBackColor = true;
            this.node2IP.Click += new System.EventHandler(this.node2IP_Click);
            // 
            // node3IP
            // 
            this.node3IP.Location = new System.Drawing.Point(960, 633);
            this.node3IP.Name = "node3IP";
            this.node3IP.Size = new System.Drawing.Size(75, 22);
            this.node3IP.TabIndex = 55;
            this.node3IP.Text = "Node 3 IP";
            this.node3IP.UseVisualStyleBackColor = true;
            this.node3IP.Click += new System.EventHandler(this.node3IP_Click);
            // 
            // algoGui
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1351, 665);
            this.Controls.Add(this.node3IP);
            this.Controls.Add(this.node2IP);
            this.Controls.Add(this.node1IP);
            this.Controls.Add(this.nodeIPUpdater);
            this.Controls.Add(this.node3Status);
            this.Controls.Add(this.node2Status);
            this.Controls.Add(this.node1Status);
            this.Controls.Add(this.clusterNodesOnline);
            this.Controls.Add(this.lvl2Disable);
            this.Controls.Add(this.eblCluster);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.stockSelecter);
            this.Controls.Add(this.mouseCoordinates);
            this.Controls.Add(this.testMouseAction);
            this.Controls.Add(this.predictorStart);
            this.Controls.Add(this.mstClkDisable);
            this.Controls.Add(this.dowDisable);
            this.Controls.Add(this.lastTradesDisable);
            this.Controls.Add(this.synthData);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.priceTierAsks);
            this.Controls.Add(this.priceTierBids);
            this.Controls.Add(this.prevTradesSize);
            this.Controls.Add(this.prevTradesPrice);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.hrsIndicator);
            this.Controls.Add(this.fullStop);
            this.Controls.Add(this.controlBtns);
            this.Controls.Add(this.exitBtn);
            this.Controls.Add(this.lastTradeSize);
            this.Controls.Add(this.lastTradePrice);
            this.Controls.Add(this.dowTxt);
            this.Controls.Add(this.mstClkText);
            this.Controls.Add(this.asksSize);
            this.Controls.Add(this.asksPrice);
            this.Controls.Add(this.bidsSize);
            this.Controls.Add(this.bidsPrice);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.fullStart);
            this.Location = new System.Drawing.Point(568, 270);
            this.Name = "algoGui";
            this.StartPosition = System.Windows.Forms.FormStartPosition.Manual;
            this.Text = "Autonomous Trader Data Scraper Module";
            this.Load += new System.EventHandler(this.algoGui_Load);
            this.controlBtns.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        public System.Windows.Forms.Button button3;
        public System.Windows.Forms.Button button4;
        public System.Windows.Forms.Button fullStart;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        public System.Windows.Forms.Button updateBidsAsks;
        private System.Windows.Forms.Label bidsPrice;
        private System.Windows.Forms.Label bidsSize;
        private System.Windows.Forms.Label asksSize;
        private System.Windows.Forms.Label asksPrice;
        private System.Windows.Forms.Button updateMstClk;
        private System.Windows.Forms.Button updateDow;
        private System.Windows.Forms.Label lastTradeSize;
        private System.Windows.Forms.Label lastTradePrice;
        private System.Windows.Forms.Button updateLasts;
        private System.Windows.Forms.Button exitBtn;
        private System.Windows.Forms.GroupBox controlBtns;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label mstClkText;
        private System.Windows.Forms.Label dowTxt;
        private System.Windows.Forms.Button fullStop;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label prevTradesSize;
        private System.Windows.Forms.Label prevTradesPrice;
        private System.Windows.Forms.Label priceTierBids;
        private System.Windows.Forms.Label priceTierAsks;
        public System.Windows.Forms.Label hrsIndicator;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Button synthData;
        private System.Windows.Forms.CheckBox lastTradesDisable;
        private System.Windows.Forms.CheckBox dowDisable;
        private System.Windows.Forms.CheckBox mstClkDisable;
        private System.Windows.Forms.Button predictorStart;
        public System.Windows.Forms.Label mouseCoordinates;
        public System.Windows.Forms.Button testMouseAction;
        private System.Windows.Forms.ComboBox stockSelecter;
        private System.Windows.Forms.Label label8;
        public System.Windows.Forms.CheckBox eblCluster;
        public System.Windows.Forms.CheckBox lvl2Disable;
        public System.Windows.Forms.Label clusterNodesOnline;
        public System.Windows.Forms.Label node1Status;
        public System.Windows.Forms.Label node2Status;
        public System.Windows.Forms.Label node3Status;
        private System.Windows.Forms.TextBox nodeIPUpdater;
        private System.Windows.Forms.Button node1IP;
        private System.Windows.Forms.Button node2IP;
        private System.Windows.Forms.Button node3IP;
    }
}

