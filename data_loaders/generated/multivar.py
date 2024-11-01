import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class MyMultivarData:
    def __init__(self, csv_path, var_list=None, lag_list=None, scaler=None):
        """
        Parameters:
        - csv_path (str): path to csv file;
        - var_list (list of str): list of variable names;
        - lag_list (list of int): list of lags for each variable, default=None, no lag;
        
        - scaler (str): scaler to use, default=None, no scaling; options are 'std'/'standard' and 'mm'/'minmax';
        """
            
        self.csv_path = csv_path
        self.var_list = var_list
        self.lag_list = lag_list

        self.scaler = scaler

        self.df = self._scale_data(self._read_raw_data())

    def _read_raw_data(self):
        df=pd.read_csv(self.csv_path, index_col=None, sep=',')
        if self.var_list==None:
            self.var_list=df.columns
        else:
                assert set(self.var_list).issubset(set(df.columns)), "Some variables are not in the dataset."
                df=df[self.var_list]
        self.raw_df=df
        return df
    
    def _scale_data(self, df):
        if self.scaler==None:
            return df
        elif self.scaler=='std' or self.scaler=='standard':
            scaler=StandardScaler()
        elif self.scaler=='mm' or self.scaler=='minmax':
            scaler=MinMaxScaler()
        else:
            raise ValueError("scaler must be one of 'std'/'standard', 'mm'/'minmax'.")
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)