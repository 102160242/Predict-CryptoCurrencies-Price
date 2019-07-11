# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:52:20 2019

@author: doanh
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
#from sklearn.metrics.pairwise import cosine_similarity

def knn_pred(Y, S, u, i):
    ids = np.where(Y[:,i])[0] # Day ID cac hang khong bi khuyet du lieu o cot dang xet
    
    sim = S[u,ids] # do tuong dong cua u voi day ids
    # KNN voi k = 6
    nns = np.argsort(sim)[-6:] # lay index
    neareast_s = sim [nns]   #lay value cua 6 lang gieng > nhat
    r = Y[nns,i] #lay 6 phan tu lon nhat trong dau u no
    eps = 1e-8 #chia khac 0
    return (r*neareast_s).sum()/(np.abs(neareast_s).sum()+eps) #cong thuc 
    
def nearest_indexes(array, index):
    l = []
    k = 1
    while(len(l) < 4): # Tiếp tục tìm kiếm cho đến khi đủ 4 index gần nhất thỏa mãn điều kiện
        if(index - k >= 0 and array[index - k] != 0): # Nếu index hợp lệ (>= 0) và giá trị ứng với index khác 0
            l.append(index - k)
        if(index + k < len(array)):
            if(array[index + k] != 0):
                l.append(index + k)
        if((index - k) < 0 and (index + k) > len(array)): break
        k += 1
    return l

def add_missing_dates(dataset):
    #groupby_day = dataset.groupby(pd.DatetimeIndex(data=dataset.Date))
    #results = groupby_day.sum()
    dataset['Date'] = [int(time.mktime(t.timetuple())) for t in dataset['Date']] # Chuyển ngày sang timestamp
    idx = pd.RangeIndex(start = dataset['Date'][len(dataset) - 1], stop = dataset['Date'][0], step = 3600*24) # Tạo một range timestamp liên tục ứng với từng ngày với ngày bắt đầu & kết thúc từ dataset
    dataset.index = dataset['Date'] # Biến cột Date thành index

    #print(time.mktime(dataset['Date'][0].timetuple()))
    #idx = pd.RangeIndex(start = dataset['Date'][len(dataset) - 1], end = dataset['Date'][0])
    #for i in idx: print(i)
    #print(dataset['Date'])
    #print(idx)
    #for id in idx:
        #if(id.to_pydatetime() in dataset['Date']): print(id)
        #if(len(dataset[idx]) == 0):
        #    print(id.date)
    dataset = dataset.drop("Date", axis = 1) # Bỏ cột Date khỏi dataset
    #print(idx[-50:])
    dataset = dataset.reindex(idx, fill_value = 0) # Reindex lại dataset với range timestamp có được => Tạo thành tập dataset mới với những ngày thiếu sẽ được thêm vào
    #idx = dataset['Date'][dataset['Date'] == 0].index
    #dataset.loc[idx, 'Date'] = idx
    #ids = np.where(dataset['Date'])[0]
    #print(ids)
    return dataset

def proccess_X(X, d = 2):
    M = np.array([ [ 0 for i in range(d)] for j in range(len(X)) ]).astype(np.float) # Tạo ma trận mới có d cột, số hàng bằng số hàng của X
    M[:, 0] = X # Cột đầu tiên = X
    for i in range(1, d):
        M[:, i] = pow(M[:, 0], i + 1) # Cột thứ n bằng lũy thừa bậc n của X
    return M
        
def split_train_test(X, y, test_size = 0.3):
    k = round((1 - test_size) * len(X))
    return X[:k, :], X[k:, :], y[:k].reshape(k, 1), y[k:].reshape(len(y) - k, 1) # Trả về X_train, X_test, y_train, y_test

#def ones(n):
#    matrix  = [[1]] * n
#    return matrix

###### Ham nhan 2 ma tran #####
def dot(X, Y):
    ## Dieu kien nhan duoc la so cot ma tran X (n x m) bang so hang ma tran Y (m x p)
    n = len(X)
    p = len(Y[0])
    M = [ [ 0 for i in range(p) ] for j in range(n) ]
    #M = [[0] * p] * n # Ma tran tich kich thuoc n x p
    
    for i in range(n): ## Cot cua ma tran tich
        for j in range(p): ## Hang cua ma tran tich
            element = 0
            for k in range(len(X[0])):
                element += X[i][k] * Y[k][j]
            M[i][j] = element
    return M

def X_bar(X):
    M = np.array([ [ 0 for i in range(len(X[0]) + 1)] for j in range(len(X)) ]).astype(np.float) # Tạo ma trận M có số hàng bằng ma trận X, số cột bằng số cột ma trận X + 1
    #print(M[:, 0])
    M[:, 0] = 1 # Them cot 1
    M[:, 1:] = X
    return M

# Đọc dữ liệu
dataset = pd.read_csv('BTC.csv')
dataset['Date'] = pd.to_datetime(dataset.Date,format='%m/%d/%Y')

#dataset = pd.read_csv('BTC_minutely.csv')
#dataset['Date'] = pd.to_datetime(dataset.Date,format='%m/%d/%Y %H:%M')

# Them nhung ngay thieu
dataset = add_missing_dates(dataset)
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 5].values
    
######## Xu ly du lieu thieu ########

#S = cosine_similarity(X[:,:]).astype(np.float)
for i in range(len(X)):
    if(X[i] == 0 or X[i] == 0.0):
        #X[i, j] = knn_pred(X, S, i, j)
        indexes = nearest_indexes(X, i) # Tìm index của những giá trị khác 0 gần nhất với ô hiện tại
        avg = sum(X[indexes]) / len(indexes)
        X[i] = avg # Lấy TB
    if(y[i] == 0):
        indexes = nearest_indexes(y, i)
        avg = sum(y[indexes]) / len(indexes)
        y[i] = avg
        
        
######## Thêm các cột bậc cao ########
X = proccess_X(X, 2)

######## Chia du lieu de train, test ########           
X_train, X_test, y_train, y_test = split_train_test(X, y)           

#for i in range(len(y_test)):
#    y_test[i][0] = y_test[i][0] + 1000
###### Xay dung X du doan ######
#one = ones(len(X_train))
#Xbar = np.concatenate((one, X_train), axis = 1)  
Xbar = X_bar(X_train)

###### Tinh w ######
a = dot(Xbar.T, Xbar)
b = dot(Xbar.T, y_train)
w = dot(np.linalg.pinv(a), b) # Hệ số quy hồi riêng

#print("W = " , w)
#############################################
#ichi = np.ones((X_test.shape[0], 1))  # Tạo ma trận cột 1, shape[0]: lấy số hàng của mảng
#Xbar_test = np.concatenate((ichi,X_test), axis=1).astype(np.float) 
Xbar_test = X_bar(X_test)

Ybar = dot(Xbar_test, w) # Tập giá trị dự đoán

###### Xuat cac du lieu ra console ######
print("Hệ so hoi quy rieng")
print(w)

###### Ve bieu do ######
sticks = list(datetime.fromtimestamp(t) for t in dataset.index.values)
#plt.plot_date(sticks[:len(y_train)], y_train, 'b-')
#plt.plot_date(sticks[len(y_train):], y_test, 'g-')
plt.plot_date(sticks, y, 'g-') # Vẽ đường giá thực tế
plt.plot_date(sticks[(len(y_train)):], Ybar, 'r-')

plt.setp(plt.gca().xaxis.get_majorticklabels(),
         'rotation', 45)
plt.xlabel("Time")
plt.ylabel("Price")          
plt.show()     
