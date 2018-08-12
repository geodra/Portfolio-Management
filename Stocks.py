"""
Created on Sat Aug 11 00:40:35 2018

@author: Georgios.Drakos
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import re
print(np.__version__)

# Read & concatenate data
import glob
fields = ['Date', 'Close']
path =r'C:\Users\Georgios.Drakos\Desktop\Kaggle\Ioannis\data' 
allFiles = glob.glob(path + "/*.txt")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,usecols=fields, header=0)
    df['Id'] = file_.split(path)[-1].split('.us.txt')[-2][1:]
    list_.append(df)
frame = pd.concat(list_)
frame.info()
frame['Date'] = frame['Date'].astype('datetime64[ns]')
frame_raw = frame.copy()

df = frame.groupby('Id')['Date'].agg(['min', 'max'])
min_date = df['min'].max()
max_date = df['max'].min()
frame = frame[(frame.Date <= max_date) & (frame.Date >= min_date)]

frame['Day'] = frame['Date'].dt.day
frame['Year'] = frame['Date'].dt.year
frame['Month'] = frame['Date'].dt.month
frame.drop('Date',axis=1,inplace=True)

data = frame.groupby(['Id','Month','Year'])['Close'].agg(['min', 'max','mean','var'])
data.reset_index(inplace=True)
data.sample(10)
frame_raw.shape
data.shape

##### Plot for Year==2007###
df2=data.copy()
for i in range(1,13):
    df2=data.copy()
    df2 = df2[(df2.Month == i) & (df2.Year==2007)]
    df3 = df2[['min','var','Id']]
    #f3 = df3[df3.var < df3.var.quantile(.95)]
    df3.plot(x='var',y='min',style='.')

##### Plot removing outliers for Year==2007 ###
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
df2=data.copy()
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k',
                   2 : 'y',
                   3 : 'b'
                   }
for i in range(1,13):
    df2=data.copy()
    df2 = df2[(df2.Month == i) & (df2.Year==2007)]
    df3 = df2[['min','var']]
    df3 = df3[df3['min'] < df3.quantile(.95)['min']]
    df3 = df3[df3['var'] < df3.quantile(.95)['var']]
    scaler = MinMaxScaler()
    df3[['min','var']] = scaler.fit_transform(df3)
    estimator = KMeans(n_clusters=4)
    estimator.fit(df3)
    label_color = [LABEL_COLOR_MAP[l] for l in estimator.labels_]
    plt.title('Plot of month '+str(i))
    plt.scatter(df3['min'], df3['var'], c=label_color)
    plt.xlabel('min')
    plt.ylabel('var')
    plt.show()

## for determining optimal number of cluster applied for Month == 1 & Year==2007
from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
df2=data.copy()
df2 = df2[(df2.Month == 1) & (df2.Year==2007)]
df3 = df2[['min','var']]
df3 = df3[df3['min'] < df3.quantile(.95)['min']]
df3 = df3[df3['var'] < df3.quantile(.95)['var']]
scaler = MinMaxScaler()
df3[['min','var']] = scaler.fit_transform(df3)
for k in K:
    estimator = KMeans(n_clusters=k).fit(df3)
    distortions.append(sum(np.min(cdist(df3, estimator.cluster_centers_, 'euclidean'), axis=1)) / df3.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


