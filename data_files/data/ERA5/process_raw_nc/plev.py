# # Data Extraction Script Credit: Raphaël Rousseau-Rizzi - New(updated March 11, 2024)
# # Modified: Elise Zhang

import cdsapi
import numpy as np

c = cdsapi.Client()


# for yy in range(1981,2023): 
for yy in range(2017,2023):
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'temperature', 'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': [
                '850', '950',
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
        'plev_'+str(yy)+'.nc')






# # Data Extraction Script Credit: Raphaël Rousseau-Rizzi - Old (updated March 11, 2024)
# # Modified: Elise Zhang

# import cdsapi
# import numpy as np

# c = cdsapi.Client()


# for yy in range(2012,2023): 
#     c.retrieve(
#         'reanalysis-era5-pressure-levels',
#         {
#             'product_type': 'reanalysis',
#             'variable': [
#                 'temperature', 'u_component_of_wind', 'v_component_of_wind',
#             ],
#             'pressure_level': [
#                 '550', '700', '850',
#                 '950',
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
#         'plev_nov_mar'+str(yy)+'.nc')
