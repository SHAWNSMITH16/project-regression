import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from math import sqrt
from scipy.stats import pearsonr, spearmanr

from env import get_connection
import wrangle_zillow
import prepare


# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")







def OLS_train(X_train_scaled, y_train_scaled):
    '''This function gets train RMSE for OLS Linear Regression'''

    # create classifier object
    lm = LinearRegression()

    #fit model on training data
    lm.fit(X_train_scaled, y_train_scaled)
    
    lm_preds = lm.predict(X_train_scaled)
    
    preds_df = pd.DataFrame({'actual': y_train_scaled,'lm_preds': lm_preds})
        
    lm_rmse = sqrt(mean_squared_error(preds_df['lm_preds'], preds_df['actual']))
    
    # print result
    print(lm_rmse)
    
def OLS_val(X_val_scaled, y_val_scaled):
    '''This function gets validate RMSE for OLS Linear Regression'''

    # create classifier object
    lm = LinearRegression()

    #fit model on training data
    lm.fit(X_val_scaled, y_val_scaled)
    
    lm_val_preds = lm.predict(X_val_scaled)
    
    preds_df = pd.DataFrame({'actual': y_val_scaled,'lm_val_preds': lm_val_preds})
        
    lm_val_rmse = sqrt(mean_squared_error(preds_df['lm_val_preds'], preds_df['actual']))
    
    # print result
    print(lm_val_rmse)    
    
    
def LL_train(X_train_scaled, y_train_scaled):
    '''This function gets train RMSE for Lasso + Lars'''
    #Create
    lasso = LassoLars(alpha = 0.05)
    #Fit
    lasso.fit(X_train_scaled, y_train_scaled)

    lasso_preds = lasso.predict(X_train_scaled)
    
    preds_df = pd.DataFrame({'actual': y_train_scaled,'lasso_preds': lasso_preds})

    preds_df['lasso_preds'] = lasso_preds

    lasso_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['lasso_preds']))
    
    print(lasso_rmse)
    
def LL_val(X_val_scaled, y_val_scaled):
    '''This function gets validate RMSE for Lasso + Lars'''
    #Create
    lasso = LassoLars(alpha = 0.05)
    #Fit
    lasso.fit(X_val_scaled, y_val_scaled)

    lasso_val_preds = lasso.predict(X_val_scaled)
    
    preds_df = pd.DataFrame({'actual': y_val_scaled,'lasso_val_preds': lasso_val_preds})

    preds_df['lasso_val_preds'] = lasso_val_preds

    lasso_val_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['lasso_val_preds']))
    
    print(lasso_val_rmse)
    
    
def PR_train(X_train_scaled, y_train_scaled):
    '''This function gets train RMSE for Polynomial regression'''
    
    pf = PolynomialFeatures(degree = 3)

    pf.fit(X_train_scaled, y_train_scaled)

    X_polynomial = pf.transform(X_train_scaled)

    lmtwo = LinearRegression()

    lmtwo.fit(X_polynomial, y_train_scaled)
    
    preds_df = pd.DataFrame({'actual': y_train_scaled})

    preds_df['poly_preds'] = lmtwo.predict(X_polynomial)

    poly_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['poly_preds']))
    
    print(poly_rmse)
    
    
def PR_val(X_val_scaled, y_val_scaled):
    '''This function gets validate RMSE for Polynomial regression'''
    
    pf = PolynomialFeatures(degree = 3)

    pf.fit(X_val_scaled, y_val_scaled)

    X_polynomial = pf.transform(X_val_scaled)

    lmtwo = LinearRegression()

    lmtwo.fit(X_polynomial, y_val_scaled)
    
    preds_df = pd.DataFrame({'actual': y_val_scaled})

    preds_df['poly_preds'] = lmtwo.predict(X_polynomial)

    poly_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['poly_preds']))
    
    print(poly_rmse)
    
def BL_rmse(y_train_scaled):

    preds_df = pd.DataFrame({'actual': y_train_scaled})
    
    preds_df['yhat_baseline'] = y_train_scaled.mean()
    
    baseline_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['yhat_baseline']))
    
    print(baseline_rmse)

    
    
def model_report():
    data = {'Model': ['Linear Regression', 'Lasso + Lars', 'Polynomial Regression'],
            'Train Predictions': [221977.25, 221977.25, 220570.62],
            'Validate Predictions': [222972.52, 222972.52, 220216.19]}
    return pd.DataFrame(data)    


def PR_test(X_test_scaled, y_test_scaled):
    '''This function gets validate RMSE for Polynomial regression'''
    
    pf = PolynomialFeatures(degree = 3)

    pf.fit(X_test_scaled, y_test_scaled)

    X_polynomial = pf.transform(X_test_scaled)

    lmtwo = LinearRegression()

    lmtwo.fit(X_polynomial, y_test_scaled)
    
    preds_df = pd.DataFrame({'actual': y_test_scaled})

    preds_df['poly_preds'] = lmtwo.predict(X_polynomial)

    poly_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['poly_preds']))
    
    print(poly_rmse)