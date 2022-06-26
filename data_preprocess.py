## This is used to prepare data set
## Part 1 : Missing counter and interpolate
## Part 2 : Mda8 and Daily mean
## Part 3 : Time label (Weekly/Monthly/yearly)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import interpolate
from matplotlib import pyplot

directory_name = "./dic"
file_name = 'file.csv'
data = pd.read_csv(directory_name + file_name)
data_bf = data.values # Original
# data_bf = data.round(2).values # .00
print(data.describe())
# print('Data read')


# def missing_counter(Temp_factor):
#     Zero_count = np.zeros([1,Temp_factor.shape[1]])
# #    print(Zero_count.shape)
#     for i in range(Temp_factor.shape[0]):
#         for j in range(Temp_factor.shape[1]):
#             if Temp_factor[i,j] == -999:  
#                 Zero_count[0,j] = Zero_count[0,j]+1
#     # print('Missing data: '+str(Zero_count))
#     return Zero_count

# def missing_filled(data_used):
#     range = data_used.shape[0]
#     # data = np.zeros(range)
#     data_index = np.arange(range)
#     data_inter = data_used[data_used!= -999]
#     data_inter_index = data_index[data_used!= -999]
#     f = interpolate.interp1d(data_inter_index, data_inter, kind='nearest',fill_value="extrapolate")
#     data = np.array(f(data_index))
#     return data

# # Intepolate the missed values
# tmp = missing_counter(data_bf) # Count -999 before Interpolation
# # print(np.max(tmp),np.min(tmp),np.mean(tmp))
# # print(np.where(tmp == np.max(tmp)))
# # Interpolation by column
# for i in range(data_bf.shape[1]):
#     temp_data = data_bf[:,i]
#     temp_new_data = missing_filled(temp_data)
#     data_bf[:, i] = temp_new_data[:]

# missing_counter(data_bf) # Count -999 after Interpolation
# pd.DataFrame(data_bf).to_csv(directory_name + 'lotos_2018_inter.csv', index=False, sep=',')


# # Calculate MDA8 for ozone
# def MDA8h(ozone_data):
#     day = int(ozone_data.shape[0]/24)
#     ozone_8hsum = np.zeros(17)
#     ozone_8hmean = np.zeros(day)
#     for i in range(day):
#         for j in range(17):
#             ozone_8hsum[j] = np.sum(ozone_data[i*24+j:i*24+j+8])
#         ozone_8hmean[i] = np.max(ozone_8hsum)
#     ozone_8hmean = ozone_8hmean / 8
#     return ozone_8hmean

# # Calculate daily mean
# def daily_mean(feature):
#     day = int(feature.shape[0]/24)
#     daily_data = np.zeros(day)
#     for i in range(day):
#         daily_data[i] = np.mean(feature[i*24:(i+1)*24])
#     return daily_data


# # For MDA8
# data_mda8 = np.zeros([int(data_bf.shape[0]/24), data_bf.shape[1]])
# for i in range(data_bf.shape[1]):
#     data_mda8[:,i] = MDA8h(data_bf[:,i])
# print('Job doned')
# print(pd.DataFrame(data_mda8).describe())

# # For Daily mean
# data_daily = np.zeros([int(data_bf.shape[0]/24), data_bf.shape[1]])
# for i in range(data_bf.shape[1]):
#     data_daily[:,i] = daily_mean(data_bf[:,i])
# print(pd.DataFrame(data_daily).describe())

# pd.DataFrame(data_mda8.round(2)).to_csv(directory_name+'xx.csv',index=False, sep=',')


# # Add time label for data
# data['time'] = pd.date_range('01/01/2014', periods=len(data),freq = '1D') # add time label
# data['time']=pd.to_datetime(data['time'],format='%Y-%m-%d') # change format of time to YYYYMMDD
# # print(data)
# data = data.set_index('time') # set time as index 
# # print(data)
# def time_ave(data, time_label):
#     if time_label == 'W':
#         print('Weekly average')
#         df = data.resample('W').mean().round(2) # Weekly average
#     elif time_label == 'M':
#         print('Monthly average')
#         df = data.resample('M').mean().round(2) # Monthly average
#     elif time_label == '3M':
#         print('3 Month average')
#         df = data.resample('3M').mean().round(2) # 3 Month average
#     elif time_label == 'Y':
#         print('Yearly average')
#         df = data.resample('Y').mean().round(2) # Yearly average
#     else:
#         print('No label available')
#     return df



