import pandas as pd
from env import get_connection
import os

def get_zillow():
    
    if os.path.isfile('zillow.csv'):
        
        return pd.read_csv('zillow.csv')
    
    else:
        
        url = get_connection('zillow')
        
        query = '''
        SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
        taxvaluedollarcnt, yearbuilt, taxamount, fips 
        FROM properties_2017
        JOIN propertylandusetype USING (propertylandusetypeid)
        WHERE propertylandusetypeid = 261;
        '''
        
        df = pd.read_sql(query, url)
        
        df.to_csv('zillow.csv')
        
        return df
