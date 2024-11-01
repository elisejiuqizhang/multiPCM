from .utils import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ERA5BivarData:
    """ Note the specificity of ERA5 data: the dates might not be consecutive in certain months, 
    since we are only using Winter data (Nov, Dec, Jan, Feb, Mar / Dec, Jan, Feb); 
    this will be taken into account when applying lag to Y variable."""
    def __init__(self, csv_path, X_name, Y_name, Y_lag=0, scaler=None, season_rm=None, winter_start_month=12, winter_end_month=2):
        """
        Parameters:
        - csv_path (str): path to csv file;
        - X_name (str): name of X variable, alleged cause;    
        - Y_name (str): name of Y variable, alleged effect;
        - Y_lag (int): lag of the Y variable, default=0, will clip the length on both ends to prevent NaN;
        - scaler (str): scaler to use, default=None, no scaling; options are 'std'/'standard' and 'mm'/'minmax';
        - season_rm (int): period of the seasonality to remove, default=None, no seasonality removal; options are 'd'/'daily', 'w'/'weekly', 'm'/'monthly'.
        """
        self.csv_path = csv_path
        self.X_name = X_name
        self.Y_name = Y_name
        self.Y_lag = Y_lag
        self.scaler = scaler
        self.season_rm = season_rm
        self.w_start=winter_start_month
        self.w_end=winter_end_month

        self.df = self._scale_data(self._addLag_rmSeason(self._read_raw_data()))

    def _read_raw_data(self):
        df = pd.read_csv(self.csv_path, index_col=None, sep=',')
        df=df[['time', self.X_name, self.Y_name]] # keep the time column and the two variables
        self.raw_df=df
        return df
    
    def _addLag_rmSeason(self, df):
        if self.Y_lag==0:
            # drop the time column then do seasonality removal
            df = df.drop(columns=['time'])
            return df_seasonal_decompose(df, self.season_rm)
        else:
            """Get chunks of consecutive slices (nov-dec-jan-feb-mar)"""
            chunks=get_consecutive_chunks(df, self.w_start, self.w_end)

            """ add lag to Y w.r.t. X in each chunk, remove seasonality """
            for i in range(len(chunks)):
                chunks[i]=add_lag_to_var(chunks[i], self.Y_name, self.Y_lag)
                chunks[i]=chunks[i].drop(columns=['time', 'year','month']) # drop the time column then do seasonality removal
                chunks[i] = df_seasonal_decompose(chunks[i], self.season_rm)

            """ concatenate all chunks together """
            df = pd.concat(chunks, axis=0, ignore_index=True)

            return df
        
    def _scale_data(self, df):
        if self.scaler=='std' or self.scaler=='standard':
            scaler = StandardScaler()
        elif self.scaler=='mm' or self.scaler=='minmax':
            scaler = MinMaxScaler()
        elif self.scaler==None or 'None' or 'none':
            return df
        else:
            raise ValueError("scaler must be one of 'std'/'standard' and 'mm'/'minmax'.")
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df
    

class ERA5Bivar2CitiesData(ERA5BivarData):
    """ Note the specificity of ERA5 data: the dates might not be consecutive in certain months, 
    since we are only using Winter data (Nov, Dec, Jan, Feb, Mar / Dec, Jan, Feb); 
    this will be taken into account when applying lag to Y variable."""
    def __init__(self, csv_path_up, csv_path_down, var_name, down_lag=4, scaler=None, season_rm=None, winter_start_month=12, winter_end_month=2):
        """
        Parameters:
        - csv_path_up (str): path to csv file of the upstream region;
        - csv_path_down (str): path to csv file of the downstream region;
        - var_name (str): name of the variable;    
        - down_lag (int): lag of the variable at the downstream region, default=4, will clip the length on both ends to prevent NaN;
        - scaler (str): scaler to use, default=None, no scaling; options are 'std'/'standard' and 'mm'/'minmax';
        - season_rm (int): period of the seasonality to remove, default=None, no seasonality removal; options are 'd'/'daily', 'w'/'weekly', 'm'/'monthly'.
        """

        self.csv_path_up = csv_path_up
        self.csv_path_down = csv_path_down
        self.var_name = var_name

        self.Y_lag = down_lag
        self.X_name = var_name+'_up'
        self.Y_name = var_name+'_down'

        self.scaler = scaler
        self.season_rm = season_rm
        self.w_start=winter_start_month
        self.w_end=winter_end_month

        self.df = super()._scale_data(super()._addLag_rmSeason(self._read_raw_data()))

    def _read_raw_data(self):
        df_up=pd.read_csv(self.csv_path_up, index_col=None, sep=',')
        df_down=pd.read_csv(self.csv_path_down, index_col=None, sep=',')
        # rename the target columns in the two dfs
        df_up=df_up.rename(columns={self.var_name: self.X_name})
        df_down=df_down.rename(columns={self.var_name: self.Y_name})
        # keep one time column, and the variable of interest from two dfs, resulting in a new df of 3 columns
        df=pd.concat([df_up.loc[:,['time', self.X_name]],df_down.loc[:,[self.Y_name]]], axis=1)
        return df        