# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:55:23 2022

@author: 11519
"""

import matplotlib
import json
import numpy as np
import pandas as pd
import datetime
import time
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import copy
from IPython.display import clear_output
from scipy.special import boxcox1p, inv_boxcox
from scipy.stats import boxcox_normmax

import matplotlib
from matplotlib.font_manager import FontProperties

def evaluation_indicators(true_data,predict_data):
    true_data=np.array(true_data).reshape(-1,)
    predict_data=np.array(predict_data).reshape(-1,)
    
    mean_true_data=sum(true_data)/len(true_data)
    error=np.array(true_data)-np.array(predict_data)
    squareerror=error*error
    abserror=abs(error)
    relativeerror=abserror/true_data
    MSE=sum(squareerror)/len(squareerror)
    RMSE=MSE**(1/2)
    MAE=sum(abserror)/len(abserror)
    ARE=sum(relativeerror)/len(relativeerror)
    NSE=1-sum(squareerror)/sum((true_data-mean_true_data)**2)
    
    times=sum(relativeerror<0.2)
    
    true_data=np.array(true_data)
    predict_data=np.array(predict_data)
    Ex=true_data.mean()
    Ey=predict_data.mean()
    Exy=(true_data*predict_data).mean()
    Dx=(true_data*true_data).mean()-Ex*Ex
    Dy=(predict_data*predict_data).mean()-Ey*Ey
    r=(Exy-Ex*Ey)/((Dx*Dy)**(1/2))
    R2=1-MSE/(sum((true_data-mean_true_data)**2)/len(true_data))
    
    return [round(MSE,4),round(RMSE,4),round(MAE,4),round(ARE,4),\
            round(NSE,4),round(times/len(true_data),4),round(r,4),round(R2,4)]

class predict_model(torch.nn.Module):
    def __init__(self):
        super(predict_model, self).__init__()
        self.lstm1 = torch.nn.Sequential(        
            torch.nn.LSTM(4, 48, 1, batch_first=True),
        )        
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(48, 1),
        )
        
    def forward(self, x):        
        out, _ = self.lstm1(x)
        y = self.linear1(out[:,-1,:])
        
        return y
    
class My_loss(torch.nn.Module):
    def __init__(self, miny, maxy):
        super().__init__() 
        self.miny = miny
        self.maxy = maxy
        
    def forward(self, input, target):

        error = abs(input - target)*(self.maxy - self.miny)/(target*(self.maxy - self.miny)+self.miny)
        # error = error**2
        
        return error.mean()


trainStartDate = '1982-01-01'
trainEndDate = '2013-12-31'
testStartDate = '2014-01-1'
testEndDate = '2018-12-31'
trainDayLength = int((pd.to_datetime(trainEndDate)-pd.to_datetime(trainStartDate))/pd.Timedelta(1, 'day'))+1


##############################################################
with open(r'F:\jupyter_notebook\usa\result\stationID.txt', 'r') as f:
    all_stationID = f.readlines()

for ID_i in range(45,len(all_stationID)):
    print(ID_i)
    stationID = all_stationID[ID_i][:-1]

    allData = pd.read_json(r'F:\jupyter_notebook\usa\data\%s.json'%(stationID))
    allData.index = pd.date_range(trainStartDate,testEndDate, freq='1D')

    allData = allData[[0,1,9,10]]
    
    trainData = allData[trainStartDate:trainEndDate].copy()
    testData = allData[testStartDate:testEndDate].copy()
    
    lag = 10
    ahead = 1

    xtrain = []
    ytrain = []
    for i in range(len(trainData)-lag-(ahead-1)):
        xtrain.append(trainData.iloc[i:i+lag,:].values)
        ytrain.append(trainData.iloc[i+lag+(ahead-1),0])
    xtrain = np.array(xtrain, dtype=np.float32)   
    ytrain = np.array(ytrain, dtype=np.float32).reshape(-1, 1)
    
    train_size = int(0.9*len(xtrain))
    xtrain, xval = xtrain[:train_size], xtrain[train_size:]
    ytrain, yval = ytrain[:train_size], ytrain[train_size:]

    xtest = []
    ytest = []
    for i in range(len(testData)-lag-(ahead-1)):
        xtest.append(testData.iloc[i:i+lag,:].values)
        ytest.append(testData.iloc[i+lag+(ahead-1),0])
    xtest = np.array(xtest, dtype=np.float32)   
    ytest = np.array(ytest, dtype=np.float32).reshape(-1, 1)
    
    xmax = np.nanmax(xtrain,0)
    xmin = np.nanmin(xtrain,0)
    ymax = np.nanmax(ytrain)
    ymin = np.nanmin(ytrain)

    xtrain01 = (xtrain - xmin) / (xmax - xmin)
    ytrain01 = (ytrain - ymin) / (ymax - ymin)

    xval01 = (xval - xmin) / (xmax - xmin)
    yval01 = (yval - ymin) / (ymax - ymin)

    xtest01 = (xtest - xmin) / (xmax - xmin)
    ytest01 = (ytest - ymin) / (ymax - ymin)

    EPOCH = 600
    LR = 1e-3
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_NSE = 999999
    for time_i in range(1,6):
        
        print(time_i)

        # model
        # initialize
        model = predict_model().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        lradjust = torch.optim.lr_scheduler.StepLR(opt, step_size=EPOCH//3, gamma=1)
        # MSELoss = torch.nn.MSELoss()
        # MSELoss = torch.nn.L1Loss()
        MSELoss = My_loss(ymin,ymax)

        losstrain = []
        losstest = []
        testlossmin = 99999

        EPOCH = 5000 
        early_stop_patience = 500 
        early_stop_counter = 0
        for ep in range(EPOCH):

            model.train()
            xtrainv, ytrainv = torch.from_numpy(xtrain01).to(DEVICE), torch.from_numpy(ytrain01).to(DEVICE)
            ymodel = model(xtrainv)
            loss = MSELoss(ymodel, ytrainv)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losstrain.append(loss.item())

            model.eval() 
            xvalv, yvalv = torch.from_numpy(xval01).to(DEVICE), torch.from_numpy(yval01).to(DEVICE)
            ymodel = model(xvalv)
            loss = MSELoss(ymodel, yvalv)
            losstest.append(loss.item())
            if loss.item() < testlossmin:
                testlossmin = loss.item()
                bestParam = copy.deepcopy(model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter+=1

            lradjust.step()

            if early_stop_counter>=early_stop_patience:
                print('early_stop:%s'%(ep))
                break

        torch.save(bestParam, r'F:\jupyter_notebook\model_dl\result\lstm_mreLoss\parameter\%s_%s.pth'%(stationID, time_i))

        model.load_state_dict(bestParam) 

        model.eval()

        xtestv, ytestv = torch.from_numpy(xtest01).to(DEVICE), torch.from_numpy(ytest01).to(DEVICE)
        ymodel01 = model(xtestv).cpu().detach().numpy()
        ytest_ = (ymodel01 * (ymax - ymin)) + ymin 
           
        if best_NSE<evaluation_indicators(ytest,ytest_)[4]:
            best_NSE = evaluation_indicators(ytest,ytest_)[4]
            torch.save(bestParam, r'F:\jupyter_notebook\model_dl\result\lstm_mreLoss\parameter\%s_bestNSE.pth'%(stationID))
            
            with open(r'F:\jupyter_notebook\model_dl\result\lstm_mreLoss\%s.json'%(stationID), 'w') as f:
                json.dump(ytest_.reshape(-1,).tolist(),f)
       
        print(evaluation_indicators(ytest,ytest_))

    clear_output(wait=True)
    # break
