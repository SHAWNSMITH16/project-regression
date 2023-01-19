import pandas as pd
from env import get_connection
import os

def wrangle_zillow():
    '''
    This function's purpose is to pull a query from sequel ace and checks if
    there is a csv file for it, if not, it creates one
    '''
    if os.path.isfile('zillow.csv'):
        
        df = pd.read_csv('zillow.csv')
        
        return prepare_zillow(df)
        
    
    else:
        
        url = get_connection('zillow')
        
        query = '''
        SELECT bathroomcnt AS bath,
        bedroomcnt AS bed,
        calculatedfinishedsquarefeet AS sqft,
        finishedsquarefeet12 AS fin_sqft,
        fips,
        fullbathcnt AS full_bath,
        lotsizesquarefeet AS lotsize,
        regionidzip AS zip,
        roomcnt AS rooms,
        yearbuilt,
        taxvaluedollarcnt 
        FROM properties_2017
        JOIN predictions_2017 USING (parcelid)
        JOIN propertylandusetype USING (propertylandusetypeid)
        WHERE transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
        AND propertylandusetypeid LIKE 261;
        '''
        
        df = pd.read_sql(query, url)
        
        df.to_csv('zillow.csv')
        
        return df

def prepare_zillow(df):
    '''
    This function's purpose is to pull a query from sequel ace and checks if
    there is a csv file for it, if not, it creates one
    Then it prepares and cleans it to prepare it for use to train and model
    '''
        
    #Dropping Nulls as it only consisted of 1% of the original data
    df = df.dropna()
    
    #changing datatypes from floats to ints that had no decimal value
    df.bath, df.taxvaluedollarcnt, df.bed, df.yearbuilt, df.fips = (df.bath.astype(int), 
                                                                    df.taxvaluedollarcnt.astype(int), 
                                                                    df.bed.astype(int), 
                                                                    df.yearbuilt.astype(int),
                                                                    df.fips.astype(int))
    
    #dropping unnecessary column
    df.drop(columns = ('Unnamed: 0'), inplace = True)

  

    df = df[df.sqft <= 10000]
    
    return df
     