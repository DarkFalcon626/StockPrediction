# -*- coding: utf-8 -*-
"""
Stock Prediction Classes
----------------------------------------

File containing the data and GRU model
classes for stock prediction.
----------------------------------------

Created on Tue Oct 17 03:42:18 2023
@author: Andrew Francey

"""


import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import pylab as plt
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler


class Training_Data():
    '''
    Class for loading and converting the data to the approprate formats
    '''
    
    def __init__(self, stock_name, param):
    
        self.name = stock_name
        self.period = param['period']
        
        data = yf.download(self.name, period=self.period, interval= param['freq'])
        
        ## Scaling the data so it is better behaved when training the ML model
        x_feat = data.iloc[:,[0,4]]
        
        self.sc = StandardScaler() # Rescaling values between -1 and 1.
        x_ft = self.sc.fit_transform(x_feat.values)
        self.x_ft = pd.DataFrame(columns=x_feat.columns,
                            data=x_ft,
                            index=x_feat.index)
        
        data_raw = self.x_ft.to_numpy()
        data = []
        
        lookback = param['lookback']
        self.lookback = lookback
        train_split = param['train_split']
        
        for index in range(len(data_raw)-lookback):
            data.append(data_raw[index: index + lookback])
        
        self.data = np.array(data)
        
        train_set_size = int(np.round(train_split*self.data.shape[0]))
        
        x_train = self.data[:train_set_size,:-1,:]
        y_train = self.data[:train_set_size,-1,:]
        
        x_test = self.data[train_set_size:,:-1,:]
        y_test = self.data[train_set_size:,-1,:]
        
        self.x_train = torch.tensor(x_train)
        self.y_train = torch.tensor(y_train)
        
        self.x_test = torch.tensor(x_test)
        self.y_test = torch.tensor(y_test)
        
        self.index = np.arange(0,len(data_raw))
        
    
    def plot(self):
        
        data = self.sc.inverse_transform(self.x_ft.values)
        
        open_data = data[:,0]
        close_data = data[:,1]
        
        plt.plot(self.index,open_data,label="Open")
        plt.plot(self.index,close_data,label="Close")
        plt.title(self.name)
        plt.legend()
        plt.grid()
        plt.show()
        

class Net(nn.Module):
    
    def __init__(self, param):
        super(Net, self).__init__()
        
        input_dim = param['input_dim']
        hidden_dim = param['hidden_dim']
        num_layers = param['num_layers']
        output_dim = param['output_dim']
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, 
                          dropout=0.0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.double()
        
    def forward(self, x):
        x = x.double()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).double()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out
    
    def backprop(self, inputs, targets, loss, epoch, optimizer):
        self.train()
        
        inputs = inputs.double()
        targets = targets.double()
        
        outputs = self.forward(inputs)
        obj_val = loss(outputs, targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()
    
    def test(self, data, loss, epoch):
        
        inputs = data.x_test
        targets = data.y_test
        
        self.eval()
        with torch.no_grad():
            inputs = inputs.float()
            targets = targets.float()
            
            outputs = self.forward(inputs)
            
            cross_val = loss(outputs, targets)
        
        return cross_val
    
    
            
        
        

if __name__== '__main__':
    
    with open('param.json') as paramfile:
        param = json.load(paramfile)
        

        