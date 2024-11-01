from .utils import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class ERA5MultivarData:
    def __init__(self, csv_path, var_list, lag_list=None, scaler=None, season_rm=None, winter_start_month=12, winter_end_month=2, year=None):
        """
        Parameters:
        - csv_path (str): path to csv file;
        - var_list (list of str): list of variable names;
        - lag_list (list of int): list of lags for each variable, default=None, no lag;
        
        - scaler (str): scaler to use, default=None, no scaling; options are 'std'/'standard' and 'mm'/'minmax';
        - season_rm (str): seasonality to remove, default=None, no seasonality removal; options are 'd'/'daily', 'w'/'weekly', 'm'/'monthly';
        - winter_start_month (int): start month of winter, default=12;
        - winter_end_month (int): end month of winter, default=2.
        """
            
        self.csv_path = csv_path
        self.var_list = var_list
        self.lag_list = lag_list

        self.scaler = scaler
        self.season_rm = season_rm
        self.w_start=winter_start_month
        self.w_end=winter_end_month

        self.year=year

        self.df = self._scale_data(self._addLag_rmSeason(self._read_raw_data()))

    def _read_raw_data(self):
        df=pd.read_csv(self.csv_path, index_col=None, sep=',')
        df=df[['time']+self.var_list]
        # self.raw_df=df
        return df
    
    def _addLag_rmSeason(self, df):
        """ Different from the bivar case, here the input lags are put in a list. 
        The order of the lags matches the var_list.

        For seasonality removal, apply on each variable separately.
        """
        # get consecutive chunks 
        chunks=get_consecutive_chunks_years(df, self.w_start, self.w_end, self.year)
        # add lag of each variable in each chunk
        for i in range(len(chunks)):
            chunks[i]=add_lags_to_vars(chunks[i], self.var_list, self.lag_list)
            chunks[i]=chunks[i].drop(columns=['time', 'year','month']) # drop the time column then do seasonality removal
            chunks[i] = df_seasonal_decompose(chunks[i], self.season_rm)                         
        
        # concatenate all chunks together
        df = pd.concat(chunks, axis=0, ignore_index=True)
        return df
    
    def _scale_data(self, df):
        if self.scaler==None or self.scaler=='None' or self.scaler=='none':
            return df
        elif self.scaler=='std' or self.scaler=='standard':
            scaler=StandardScaler()
        elif self.scaler=='mm' or self.scaler=='minmax':
            scaler=MinMaxScaler()
        else:
            raise ValueError("scaler must be one of 'std'/'standard', 'mm'/'minmax'.")
        # self.df=pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    

class ERA5MultivarData_WinterYearConsecutive(ERA5MultivarData):
    """ get consecutive winter chunks:
    get a input year, and retrieve the chunk 
    which starts from the 'winter_start_month' of the previous year 
    to the 'winter_end_month' of the input year.
    
    Edge cases:
    the beginning year (1981) which doesn't have a previous year, so only take the January (1) to "winter_end_month" of the input year;
    the ending year (2023) which doesn't have a next year, so only take the "winter_start_month" to December (12)."""
    def __init__(self, csv_path, var_list, lag_list=None, scaler=None, season_rm=None, winter_start_month=12, winter_end_month=2, year=1981):
        self.year=year
        # verify the year is in the range of 1981-2023
        if self.year<1981 or self.year>2023:
            raise ValueError("Year must be in the range of 1981-2023.")
        super().__init__(csv_path, var_list, lag_list, scaler, season_rm, winter_start_month, winter_end_month, year)


        

        
