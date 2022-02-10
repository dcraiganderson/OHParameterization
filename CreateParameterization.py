# A sample script to read in the training data and create a parameterization

# requirements
import xgboost as xgb
import argparse
import sys
import random
import csv
import numpy as np
import datetime as dt
import os
import pandas as pd
from scipy import stats 
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from joblib import dump
from joblib import load
import sys


FileModifier = 'DepthX_etaY_ZTrees_M07'
savedir = '/user/dir/'

def main(args):
    # Specify any training data to be dropped
    # args.drop = ["CH4"]
    # prepare training data
    X_train,X_valid,Y_train,Y_valid = get_training_data(args)
    # train model
    bst = train(args,X_train,Y_train)
    return

def train(args,X_train,Y_train):
    train = xgb.DMatrix(X_train,Y_train)
    print('training on {:,} samples'.format(X_train.shape[0]))
    bst = xgb.XGBRegressor(max_depth=6,learning_rate=.3,n_estimators=10)
    bst.fit(X_train,Y_train)
    print('Saving model...')
    modelname = savedir + 'M2GMIOHParameterizaiton_' + FileModifier + '.joblib.dat' 
    dump(bst,modelname)
    return bst

def get_training_data(args):
    X_all,_ = read_data(args,args.featurefile,args.drop)
    Y_all,_ = read_data(args,args.targetfile)
    Y_all = np.log10(Y_all) #Train to the log of OH
    X_train,X_valid,Y_train,Y_valid = train_test_split(X_all,Y_all,test_size=args.test_size)
    return X_train,X_valid,Y_train,Y_valid

def read_data(args,filename,drop=None,meta=False):
    cnt = 0
    for line in open(filename,"r"):
        if cnt==1:
            header = line.replace("\n","").split(",")
            break
        cnt += 1
    print("reading "+filename)
    dat = pd.read_csv(filename,skiprows=3,names=header,sep=r"\s+",nrows=args.nrows)
    metadat = dat[METACOLS].copy() if meta else None
    if drop is not None:
        dat = dat.drop(columns=drop)
        print('dropped: '+",".join(drop))
    return dat,metadat
  
# argument parser
def parse_args():
    p = argparse.ArgumentParser(description='Undef certain variables')
    p.add_argument('-f','--featurefile',type=str,help='feature files template',default='/user/dir/SampleTrainingSet.dat')
    p.add_argument('-t','--targetfile',type=str,help='target files template',default='/user/dir/SampleTrainingTargets.dat')
    p.add_argument('-ts','--test-size',type=float,help='training test size',default=0.2)
    p.add_argument('-n','--nrows',type=int,help='number of rows to read',default=None)
    p.add_argument('-d','--drop',type=list,help='variables to drop from training',default=None)
    return p.parse_args()

# driver
if __name__ == '__main__':
    main(parse_args())
