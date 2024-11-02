# Data Extraction Script Credit: Raphaël Rousseau-Rizzi
# Modified: Elise Zhang

import xarray as xr
import numpy as np

# Define bounds to (possibly) use and dictionnary for chunking the data
# Montreal Area
lon_bnds = [-75, -72]
lat_bnds = [46, 43]
area_dict = {'latitude':slice(*lat_bnds),'longitude':slice(*lon_bnds)}

# Upstream quantities
lon_bnds_up = [-78, -75]
lat_bnds_up = [44, 41]
area_dict_up = {'latitude':slice(*lat_bnds_up),'longitude':slice(*lon_bnds_up)}


fdir = "/home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/CausalTemporalDemo/ERA5_data/process_raw_nc/"

# Define function for the computation of advection
def compute_advection(ds_u,ds_v,ds_S):
    a = 6371220.0*2*np.pi/360
    lat = ds_S.latitude
    advS = ds_v*ds_S.differentiate(coord='latitude')/a + ds_u*ds_S.differentiate(coord='longitude')/(a*np.cos(np.deg2rad(lat)))
    return advS


fnms = ['Montreal', 'Upstream']
for ii, sel_dict in enumerate([area_dict, area_dict_up]):

    # Initialize timeseries
    ds_cloud = []
    ds_cloud_flux = []
    # Temperature
    ds_t2m = []
    # Advection (still 4 levels)
    ds_advT = []
    # Radiation
    ds_rad = []
    ds_rad_cs = []
    # Radiation - Longwave and shortwave separately
    ds_terr_rad = []
    ds_terr_rad_cs = []
    ds_solar_rad = []
    ds_solar_rad_cs = []

    # Open years sequentially
    for yy in range(2012, 2023):
        # Open datasets sequentially
        ds = xr.open_dataset(fdir + 'plev_nov_mar' + str(yy) + '.nc')
        ds2 = xr.open_dataset(fdir + 'slev_nov_mar' + str(yy) + '.nc')
        # Compute advection
        advT = compute_advection(ds['u'], ds['v'], ds['t'])

        # Make timeseries
        # Clouds
        ds_cloud.append((ds2['tciw'] + ds2['tclw']).sel(sel_dict).mean(dim=('latitude', 'longitude')))
        ds_cloud_flux.append((-ds2['p80.162'] - ds2['p79.162']).sel(sel_dict).mean(dim=('latitude', 'longitude')))
        # Temperature
        ds_t2m.append(ds2['t2m'].sel(sel_dict).mean(dim=('latitude', 'longitude')))
        # Advection (still 4 levels)
        ds_advT.append(-advT.sel(sel_dict).mean(dim=('latitude', 'longitude')).compute())
        # Radiation - Net
        ds_rad.append((ds2['msnlwrf'] + ds2['msnswrf']).sel(sel_dict).mean(dim=('latitude', 'longitude')))
        ds_rad_cs.append((ds2['msnlwrfcs'] + ds2['msnswrfcs']).sel(sel_dict).mean(dim=('latitude', 'longitude')))
        # Radiation - Longwave and shortwave separately
        ds_terr_rad.append((ds2['msnlwrf']).sel(sel_dict).mean(dim=('latitude', 'longitude')))  # longwave only
        ds_terr_rad_cs.append((ds2['msnlwrfcs']).sel(sel_dict).mean(dim=('latitude', 'longitude')))  # longwave only
        ds_solar_rad.append((ds2['msnswrf']).sel(sel_dict).mean(dim=('latitude', 'longitude')))  # shortwave only
        ds_solar_rad_cs.append((ds2['msnswrfcs']).sel(sel_dict).mean(dim=('latitude', 'longitude')))  # shortwave only

    # Concatenate in time
    cloud = xr.concat(ds_cloud, dim='time').assign_attrs(
        {'Long name': 'Total vertically integrated cloud water', 'Units': 'kg/m2'}).to_dataset(name='tcw')
    cloud_flux = xr.concat(ds_cloud_flux, dim='time').assign_attrs(
        {'Long name': 'Total vertically integrated cloud water flux convergence', 'Units': 'kg/m2/s'}).to_dataset(
        name='tcw_conv')
    t2m = xr.concat(ds_t2m, dim='time').assign_attrs({'Long name': 'Temperature at 2 meters', 'Units': 'K'}).to_dataset(
        name='T_2m')
    rad = xr.concat(ds_rad, dim='time').assign_attrs(
        {'Long name': 'Net surface radiative flux', 'Units': 'W/m2'}).to_dataset(name='rad')
    rad_cs = xr.concat(ds_rad_cs, dim='time').assign_attrs(
        {'Long name': 'Net clear-sky surface radiative flux', 'Units': 'W/m2'}).to_dataset(name='rad_cs')
    terr_rad = xr.concat(ds_terr_rad, dim='time').assign_attrs(
        {'Long name': 'Terrestrial longwave radiative flux', 'Units': 'W/m2'}).to_dataset(name='terr_rad')
    terr_rad_cs = xr.concat(ds_terr_rad_cs, dim='time').assign_attrs(
        {'Long name': 'Terrestrial longwave clear-sky radiative flux', 'Units': 'W/m2'}).to_dataset(name='terr_rad_cs')
    solar_rad = xr.concat(ds_solar_rad, dim='time').assign_attrs(
        {'Long name': 'Solar shortwave radiative flux', 'Units': 'W/m2'}).to_dataset(name='solar_rad')
    solar_rad_cs = xr.concat(ds_solar_rad_cs, dim='time').assign_attrs(
        {'Long name': 'Solar shortwave clear-sky radiative flux', 'Units': 'W/m2'}).to_dataset(name='solar_rad_cs')

    # Expand advection terms by level to make simpler .csv conversion
    advT_950 = xr.concat(ds_advT, dim='time').sel(level=950).assign_attrs(
        {'Long name': '950 mb temperature advection', 'Units': 'K/s'}).to_dataset(name='T_adv_950').drop('level')
    advT_850 = xr.concat(ds_advT, dim='time').sel(level=850).assign_attrs(
        {'Long name': '850 mb temperature advection', 'Units': 'K/s'}).to_dataset(name='T_adv_850').drop('level')
    advT_700 = xr.concat(ds_advT, dim='time').sel(level=700).assign_attrs(
        {'Long name': '700 mb temperature advection', 'Units': 'K/s'}).to_dataset(name='T_adv_700').drop('level')
    advT_550 = xr.concat(ds_advT, dim='time').sel(level=550).assign_attrs(
        {'Long name': '550 mb temperature advection', 'Units': 'K/s'}).to_dataset(name='T_adv_550').drop('level')
    # advT = xr.concat(ds_advT,dim='time').assign_attrs({'Long name':'Temperature advection','Units':'K/s'}).to_dataset(name='T_adv')

    # Merge and save
    ds_tmp = xr.merge([cloud, cloud_flux, t2m, advT_950, advT_850, advT_700, advT_550, rad, rad_cs, terr_rad, terr_rad_cs, solar_rad, solar_rad_cs])
    # Save both in netcdf and in .csv
    ds_tmp.to_netcdf('Timeseries_' + fnms[ii] + '.nc')
    ds_tmp.to_dataframe().to_csv('Timeseries_' + fnms[ii] + '.csv')
