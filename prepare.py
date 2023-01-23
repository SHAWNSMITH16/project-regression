
import pandas as pd
from env import get_connection
import os

import numpy as np
import matplotlib.pyplot as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from math import sqrt


from env import get_connection
import wrangle_zillow
import prepare


# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")


def train_val_test(df):
    '''This function takes in a dataframe that has been prepared and splits it into train, val, and test
    sections at 70/18/12 so it can be run through algorithms and tested for accuracy'''
    
    seed = 42
    
    train, val_test = train_test_split(df, train_size = 0.7, random_state = seed) 
        
    val, test = train_test_split(val_test, train_size = 0.6, random_state = seed)
    
    return train, val, test #these will be returned in the order in which they are sequenced

def MM_scaler(X_train, X_val, X_test):
    '''
    Takes in three pandas DataFrames: X_train, X_val, X_test
    output: scaler object, sclaer versions of X_train, X_val, and X_test
    
    This function assumes the independent variables being fed into it as arguements 
    are all consisting of continuous features (numeric variables)
    '''
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled






def scale_splits_rb(train, val, test, 
                    columns_to_scale = ['sqft'],
                    return_scaler = False):
    '''
    The purpose of this function is to accept, as input, the 
    train, val, and test data splits, and returns the scaled versions of each.
    '''

    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    # make the thing
    scaler = MinMaxScaler()
    # fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    val_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(val[columns_to_scale]),
                                                  columns=val[columns_to_scale].columns.values).set_index([val.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, val_scaled, test_scaled
    else:
        return train_scaled, val_scaled, test_scaled
    
    return train_scaled, val_scaled, test_scaled


def zillow_dummies(df):
    '''This function takes in the zillow df specifically and drops the 
    listed columns, creates dummy variables for the selected columns, and renames
    the dummy variable columns to make them easier to read'''
    
    df.drop(columns = ['rooms', 'zipcode', 'lotsize', 'full_bath', 'fin_sqft', 'yearbuilt'], inplace = True)
    
    df.fips = df.fips.astype(str)
    
    df = pd.get_dummies(df, columns = ['fips', 'bed', 'bath'])
    
    df.rename(columns = {'fips': 'fips_6037',
                         'taxvaluedollarcnt': 'taxvalue'}, inplace = True)
    
    df.taxvalue = df.taxvalue.astype(float)
 
    return df


def X_y_tvt(train, val, test):
    
    X_train = train.drop(columns=['taxvalue'])
    y_train = train.taxvalue

    X_val = val.drop(columns=['taxvalue'])
    y_val = val.taxvalue

    X_test = test.drop(columns=['taxvalue'])
    y_test = test.taxvalue
    
    return X_train, y_train, X_val, y_val, X_test, y_test