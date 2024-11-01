import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def df_seasonal_decompose(df, seasonality_rm, model="additive"): # return a new dataframe of the seasonal component
    if seasonality_rm == None:
        return df
    else:
        if seasonality_rm == "d" or seasonality_rm == "daily":
            period = 24
        elif seasonality_rm == "w" or seasonality_rm == "weekly":
            period = 24*7
        elif seasonality_rm == "m" or seasonality_rm == "monthly":
            period = 24*30
        else:
            raise ValueError("seasonality_rm must be one of 'd'/'daily', 'w'/'weekly', 'm'/'monthly'.")
        return pd.concat([pd.DataFrame({col: seasonal_decompose(df[col], model=model, period=period).seasonal}) for col in df.columns], axis=1)

def get_consecutive_chunks(df, winter_start_month=12, winter_end_month=2):
    """ 
    Note the inconsecutive dates inbetween, clip them out;
    the dates of interest will be the first and last dates of each month.
    Return a list of pd.DataFrames.
    """
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S') # convert time column to datetime format
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month

    """ 
    Get chunks of consecutive slices (nov-dec-jan-feb-mar), add lag to Y for each chunk;
    clip the two ends for each; then concatenate all chunks together.
    """
    years = df['year'].unique() # 2012 - 2019 inclusive

    chunks = []
    for year in range(years.min(), years.max()+1): # loop through all
        if year==years.min():
            df_chunk = df[(df['year']==year) & (df['month']<=winter_end_month)] 
        else:
            """ add the nov and dec of previous year and the jan-mar of this year """
            df_chunk = df[((df['year']==year-1) & (df['month']>=winter_start_month)) | ((df['year']==year) & (df['month']<=winter_end_month))]
        chunks.append(df_chunk)
    """deal with the last chunk with only nov and dec of the last year"""
    df_chunk = df[(df['year']==years.max()) & (df['month']>=winter_start_month)]
    chunks.append(df_chunk)

    return chunks

def get_consecutive_chunks_years(df, winter_start_month=12, winter_end_month=2, year=None):
    if year==None:
        return get_consecutive_chunks(df, winter_start_month, winter_end_month)
    else:
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S') # convert time column to datetime format
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month

        # verify year is in the range of 1981-2023
        if year<1981 or year>2023:
            raise ValueError("Year must be in the range of 1981-2023.")
        else:
            # Edge cases: 1981, 2023
            if year==1981:
                # then only take the January (1) to "winter_end_month" of the input year
                df_chunk = df[(df['year']==year) & (df['month']<=winter_end_month)]
            elif year==2023:
                # then only take the "winter_start_month" to December (12) of the input year
                df_chunk = df[(df['year']==year) & (df['month']>=winter_start_month)]
            else:
                # start from the "winter_start_month" of the previous year to the "winter_end_month" of the input year
                df_chunk = df[((df['year']==year-1) & (df['month']>=winter_start_month)) | ((df['year']==year) & (df['month']<=winter_end_month))]
            return [df_chunk]


def add_lag_to_var(df, Y_name, Y_lag):
    """ Add lag to one varialbe Y in a pd.DataFrame. 
    Drop the rows containing NaN values after the lag shift."""
    df[Y_name]=df[Y_name].shift(-Y_lag)
    df.dropna(inplace=True) # clip the NaN rows
    return df

def add_lags_to_vars(df, var_list, lag_list):
    if lag_list==None:
        return df
    elif len(var_list)!=len(lag_list):
        raise ValueError("var_list and lag_list must have the same length.")
    else:
        for i in range(len(var_list)):
            df[var_list[i]]=df[var_list[i]].shift(-lag_list[i])
        df.dropna(inplace=True) # clip the NaN rows
        return df