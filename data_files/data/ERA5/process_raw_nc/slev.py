# Data Extraction Script Credit: Raphaël Rousseau-Rizzi - New (updated March 11, 2024)
# Modified: Elise Zhang

import cdsapi
import numpy as np

c = cdsapi.Client()

# for yy in range(1981,2023):
for yy in range(2017,2023):
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '2m_dewpoint_temperature', '2m_temperature', 'mean_surface_latent_heat_flux',
                'mean_surface_net_long_wave_radiation_flux', 'mean_surface_net_long_wave_radiation_flux_clear_sky', 'mean_surface_net_short_wave_radiation_flux',
                'mean_surface_net_short_wave_radiation_flux_clear_sky', 'mean_surface_sensible_heat_flux', 'skin_temperature',
                'total_cloud_cover', 'total_column_cloud_ice_water', 'total_column_cloud_liquid_water', 'total_column_water',
            ],
            'year': str(yy),
            'month': [
                '01', '02', '03', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                46, -78, 41,
                -72,
            ],
            'format': 'netcdf',
        },
        'slev_'+str(yy)+'.nc')




# # Data Extraction Script Credit: Raphael Rousseau-Rizzi - Old (updated March 11, 2024)
# # Modified: Elise Zhang

# import cdsapi
# import numpy as np

# c = cdsapi.Client()

# # for yy in range(2012,2023):
# for yy in range(2020,2023):
#     c.retrieve(
#         'reanalysis-era5-single-levels',
#         {
#             'product_type': 'reanalysis',
#             'variable': [
#                 '2m_dewpoint_temperature', '2m_temperature', 'mean_surface_latent_heat_flux',
#                 'mean_surface_net_long_wave_radiation_flux', 'mean_surface_net_long_wave_radiation_flux_clear_sky', 'mean_surface_net_short_wave_radiation_flux',
#                 'mean_surface_net_short_wave_radiation_flux_clear_sky', 'mean_surface_sensible_heat_flux', 'skin_temperature',
#                 'total_cloud_cover', 'total_column_cloud_ice_water', 'total_column_cloud_liquid_water',
#                 'vertical_integral_of_divergence_of_cloud_frozen_water_flux', 'vertical_integral_of_divergence_of_cloud_liquid_water_flux',
#             ],
#             'year': str(yy),
#             'month': [
#                 '01', '02','03','11', '12',
#             ],
#             'day': [
#                 '01', '02', '03',
#                 '04', '05', '06',
#                 '07', '08', '09',
#                 '10', '11', '12',
#                 '13', '14', '15',
#                 '16', '17', '18',
#                 '19', '20', '21',
#                 '22', '23', '24',
#                 '25', '26', '27',
#                 '28', '29', '30',
#                 '31',
#             ],
#             'time': [
#                 '00:00', '01:00', '02:00',
#                 '03:00', '04:00', '05:00',
#                 '06:00', '07:00', '08:00',
#                 '09:00', '10:00', '11:00',
#                 '12:00', '13:00', '14:00',
#                 '15:00', '16:00', '17:00',
#                 '18:00', '19:00', '20:00',
#                 '21:00', '22:00', '23:00',
#             ],
#             'area': [
#                 55, -80, 40,
#                 -55,
#             ],
#             'format': 'netcdf',
#         },
#         'slev_nov_mar'+str(yy)+'.nc')
