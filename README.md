# Multivariate Partial Cross Mapping (*multiPCM*) and *MXMap* Framework

A multivariate extension of the Convergent Cross Mapping (CCM) method for causal discovery in dynamical systems. 

## Acquiring ERA5 Meteorological Data
Processed data are already provided along with data extraction scripts under the folder `data_files/data/ERA5` (change path accordingly). May need to install `cdsapi` to be able to run the script. Please refer to the [source website](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download) for download or api request.

## Environment Setup
Recommended with conda. See `notes_on_env.txt` or  `requirements.txt`.

## Model Implementations
The `multiPCM` and `MXMap` implementations are under the folder `utils`. Corresponding experiment scripts to run are under the folder `exps`.