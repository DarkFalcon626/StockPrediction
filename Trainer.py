# -*- coding: utf-8 -*-
"""
Stock Prdeiction model trainer
---------------------------------------

File containing the functions to train 
a model to predict stocks.
---------------------------------------

Created on Wed Oct 25 02:51:26 2023
@author: Andrew Francey

"""

import time
import pickle
import json, argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pylab as plt
from classes import Training_Data, Net


def prep(param, stock, train = True):
    
    data = Training_Data(stock, param["Data"])
    
    if train:
        model = Net(param['Net'])
    else:
        model = pickle.load(open('stock_pred_model.pkl','rb'))
    
    return data, model

def train(data, model, param):
    
    optimizer = optim.Adam(model.parameters(), lr = param['lr'])
    loss = nn.MSELoss(reduction='mean')
    
    loss_val = []
    cross_val = []
    
    num_epochs = int(param['num_epochs'])
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        train_val = model.backprop(data.x_train,data.y_train, loss, epoch, optimizer)
        loss_val.append(train_val)
        
        test_val = model.test(data, loss, epoch)
        cross_val.append(test_val)
        
        epoch_time = time.time() - epoch_start
        epoch_time = time.strftime("%H hours, %M minuts, and %S seconds", time.gmtime(epoch_time))
        if (epoch+1)%param['display_epochs'] == 0:
            epoch_time = time.time() - epoch_start
            text = f'Epoch[{epoch+1}/{num_epochs}] ({((epoch + 1)/num_epochs)*100}) Loss: {train_val} Test loss {test_val}'
            print(text)

    return loss_val, cross_val


def plot_loss(loss_val, cross_val):
    
    x_len = len(loss_val)
    x = np.arange(x_len)
    
    plt.plot(x, loss_val, label='Training loss')
    plt.plot(x, cross_val, label='Test loss')
    plt.grid()
    plt.legend()
    plt.show()

def plot_model_results(model, data):
    
    x_axis = data.index 
    
    true_data = data.sc.inverse_transform(data.x_ft)
    
    y_train = model.forward(data.x_train)
    y_train = data.sc.inverse_transform(y_train.detach())
    
    train_size = y_train.shape[0] + data.lookback -2
    
    y_test = model.forward(data.x_test)
    y_test = data.sc.inverse_transform(y_test.detach())
    
    test_size = y_test.shape[0] + data.lookback
    
    plt.figure(1)
    plt.plot(x_axis,true_data[:,0],label='True value')
    plt.plot(x_axis[data.lookback-2:train_size],y_train[:,0], label='Training values')
    plt.plot(x_axis[train_size+2:train_size+test_size],y_test[:,0], label='Test values')
    plt.title(data.name + " Opening values over the course of {}".format(data.period))
    plt.grid()
    plt.legend()
    
    plt.figure(2)
    plt.plot(x_axis,true_data[:,1],label='True value')
    plt.plot(x_axis[data.lookback-2:train_size],y_train[:,1], label='Training values')
    plt.plot(x_axis[train_size+2:train_size+test_size],y_test[:,1], label='Test values')
    plt.title(data.name + " Closing values over the course of {}".format(data.period))
    plt.grid()
    plt.legend()
    plt.show()
    
    
    

if __name__ == '__main__':
    
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Prdiction of the opening and closing stock prices using a GRU network')
    parser.add_argument('--param', default='param.json', help='JSON file for the hyperparameters to train the  GRU network')
    parser.add_argument('--stock-name', default='AMD', help='Abbrivation name for a stock example: "AMZN"')
    parser.add_argument('--model-file', default='stock_pred_model.pkl', help='Name of the file the model is saved too')
    
    args = parser.parse_args()
    
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    
    data, model = prep(param, args.stock_name)
    loss, cross = train(data, model, param['exec'])
    
    fin_time = time.time() - start_time
    fin_time = time.strftime("%H hours, %M minuts, and %S seconds", time.gmtime(fin_time))
    fin_text = 'The model was trained in '+ fin_time
    
    print(fin_text)
    
    plot_loss(loss, cross)
    
    plot_model_results(model, data)
    
    save = int(input('Enter 1 if you would like to save the file: '))
    
    if save:
        
        with open(args.model_file, 'wb') as f:
            pickle.dump(model, f)
        f.close()
        print('Model saved as {}'.format(args.model_file))
        
    
    
    
    