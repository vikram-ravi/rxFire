"""
Created on August 10, 2016
@author: vikram
to create boxplots of the three simulations scenarios for the Rx fire, at selected sites
"""
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib
import os
from matplotlib import patches as mpatches
matplotlib.rcParams['font.family'] = 'serif' 
matplotlib.rcParams['ytick.major.size'] = 10.0
matplotlib.rcParams['ytick.minor.size'] = 5.0
matplotlib.rcParams['xtick.major.size'] = 10.0
matplotlib.rcParams['xtick.minor.size'] = 5.0
matplotlib.rcParams['ytick.major.width'] = 2.0
matplotlib.rcParams['xtick.major.width'] = 2.0
#%%
# define case - options are select / aqs / improve
case = 'select'
workdir   = 'C:/Users/vik/Documents/pythonProjects/pythonInput/rxFire/'
metdir    = 'C:/Users/vik/Documents/Projects/MCIP_data/'
indirfire = 'C:/Users/vik/Documents/pythonProjects/pythonInput/rxFire/ncrcatFiles/' #  the fire files
outdir    = 'C:/Users/vik/Documents/pythonProjects/pythonOutput/rxFire/plots/'
if not os.path.exists(outdir):
   os.makedirs(outdir)
fn  = metdir + '/2011111100/MCIP/GRIDCRO2D'
grd = Dataset(fn,'r')
lat = grd.variables['LAT'][0,0,:,:]
lon = grd.variables['LON'][0,0,:,:]
ht  = grd.variables['HT'][0,0,:,:]

grd.close()
#%%
# based on the code at
#   http://nbviewer.ipython.org/github/Unidata/unidata-python-workshop/blob/master/netcdf-by-coordinates.ipynb
def naive_fast(latvar,lonvar,lat0,lon0):
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:]
    lonvals = lonvar[:]
    ny,nx = latvals.shape
    dist_sq = (latvals-lat0)**2 + (lonvals-lon0)**2
    minindex_flattened = dist_sq.argmin()  # 1D index of min element
    iy_min,ix_min = np.unravel_index(minindex_flattened, latvals.shape)
    return int(iy_min),int(ix_min)

if case == 'select':
   st_file = workdir + 'selected_sites.csv'
   stations = pd.read_csv(st_file, skiprows=[1])
elif case == 'aqs':
   st_file = workdir + 'aqsid.csv'
   stations = pd.read_csv(st_file, skiprows=[1])
elif case == 'improve':
   st_file = workdir + 'improvesiteid.csv'
   stations = pd.read_csv(st_file, skiprows=[1])

for i in stations.index:
#    iy,ix = naive_fast(lat, lon, loc[0,0], loc[0,1])
    iy,ix = naive_fast(lat, lon, stations.ix[i, 'Latitude'], stations.ix[i, 'Longitude'])
    stations.ix[i, 'row'] = iy
    stations.ix[i, 'column'] = ix    

#%%
# convert from CMAQ/MCIP date format to gregorian format
def jul_to_greg(timestamps):
    dateTime = []
    for i in range(len(timestamps)):
        jday = timestamps[i][0]
        hour = timestamps[i][1]
        
        jday = str(int(jday))
        hour = str(int(hour/10000))
        ydoy_time = jday + hour
    #    print (ydoy_time)
        date_string = (datetime.strptime(ydoy_time, '%Y%j%H') - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')
     ##   print (date_string)
        if date_string not in dateTime:
            dateTime.append(date_string)
        else:
            continue
        
    return dateTime

    if __name__ == '__main__':
        return dateTime
#%%
fireFile     = indirfire + 'ACONC_PM25_ncrcat_2011_Fire_100_L01.ncf'
lowfireFile  = indirfire + 'ACONC_PM25_ncrcat_2011_Fire_030_L01.ncf'
nofireFile   = indirfire + 'ACONC_PM25_ncrcat_2011_Fire_000_L01.ncf'

firedata     = Dataset(fireFile,'r')
lowfiredata  = Dataset(lowfireFile,'r')
nofiredata   = Dataset(nofireFile,'r')
pmfire       = firedata.variables['PM25'][:,0,:,:]
pmlowfire    = lowfiredata.variables['PM25'][:,0,:,:]
pmnofire     = nofiredata.variables['PM25'][:,0,:,:]           
ntsteps      = len(pmfire[:,0,0])
firedata.close()
lowfiredata.close()
nofiredata.close()
#%%  

df_pmFire    = pd.DataFrame(index=np.arange(0, ntsteps), columns=stations.AQSID.astype(str))
df_pmlowFire = pd.DataFrame(index=np.arange(0, ntsteps), columns=stations.AQSID.astype(str))
df_pmnoFire  = pd.DataFrame(index=np.arange(0, ntsteps), columns=stations.AQSID.astype(str))

new_stations = stations.set_index('AQSID', inplace=False)
for t in np.arange(0, ntsteps):
    for st_id in list(new_stations.index):
        row = int(new_stations.loc[st_id, 'row'])
        col = int(new_stations.loc[st_id, 'column'])
        df_pmFire.at[t, str(st_id)]    = pmfire[t, row, col]
        df_pmlowFire.at[t, str(st_id)] = pmlowfire[t, row, col]
        df_pmnoFire.at[t, str(st_id)]  = pmnofire[t, row, col]
        
#%%
name_for_plot= {'60631007'  : 'Chester \nCA', '61050002' :'Weaverville \nCA', '160090010': 'St. Maries \nID', '160090011': 'Coeur dAlene \nTribe \nID',
                '160170005' :'Sandpoint \nID', '160210002': 'Kootenai \nTribe \nID', '160790017' :'Pinehurst \nID', '160850002': 'McCall \nID',
                '410050004' :'Portland \nSpangler Rd \nOR', '410510080': 'Portland \nLafayette \nOR', '410111036': 'Radar Hill \nOR', 
                '410170004' :'Sisters \nOR', '410391007': 'Saginaw \nOR', '410392013': 'Oakridge \nOR', '410030013':'Corvallis \nOR',
                '410390060' :'Eugene \nOR', '530630021':'Spokane \nWA','530110013':'Vancouver \nWA'}


fig, ax = plt.subplots(figsize=(26,12))
boxprops1 = dict(color='red')
medprops1 = dict(color='red',linewidth=2.5)
whisprops1 = dict(color='red')
flierprops = dict(color='orangered', marker='o', markersize=5.0)
capprops   = dict(color='black', linewidth=2.0)
meanpointprops = dict(marker='s', markeredgecolor='black',
                      markerfacecolor='black')
fire_box=df_pmFire.plot(kind='box', color={'boxes':'orangered', 'whiskers':'orangered','medians':'black'}, 
                       notch=True,widths=0.15,showfliers=True,boxprops=boxprops1, whis=[2,98],logy=True, 
                       medianprops=medprops1, flierprops=flierprops, whiskerprops=whisprops1, label='Fire', 
                       showmeans=True, meanprops=meanpointprops, capprops=capprops, patch_artist=True, ax=ax, grid=False)
fire_box.legend(label='Fire')

positions_fire = fire_box.axes.get_xticks()
positions_nofire = positions_fire+0.20
boxprops2 = dict(color='blue', facecolor='red')
medprops2 = dict(color='blue',linewidth=2.5)
whisprops2 = dict(color='blue')
flierprops = dict(color='royalblue', marker='o', markersize=5.0)
capprops   = dict(color='black', linewidth=2.0)
meanpointprops = dict(marker='s', markeredgecolor='black',
                      markerfacecolor='black')
nofire_box = df_pmnoFire.plot(kind='box', color={'boxes':'royalblue', 'whiskers':'royalblue','medians':'black'}, 
                           notch=True,widths=0.15, positions=positions_nofire,  whis=[2,98],logy=True, 
                           label='No Fire' ,showfliers=True, ax=fire_box,flierprops=flierprops, 
                           boxprops=boxprops2, medianprops=medprops2, whiskerprops=whisprops2, 
                           showmeans=True, meanprops=meanpointprops, capprops=capprops, patch_artist=True, grid=False, fontsize=15)

positions_lowfire = positions_nofire+0.20
boxprops3 = dict(color='green')
medprops3 = dict(color='green',linewidth=2.5)
whisprops3 = dict(color='green')
flierprops = dict(color='forestgreen', marker='o', markersize=5.0)
capprops   = dict(color='black', linewidth=2.0)
df_pmlowFire.plot(kind='box', color={'boxes':'forestgreen', 'whiskers':'forestgreen','medians':'black'}, 
                  notch=True,widths=0.15, positions=positions_lowfire,  whis=[2,98],logy=True, label='30% Fire',
                  showfliers=True, ax=fire_box,flierprops=flierprops, boxprops=boxprops3, medianprops=medprops3, 
                  showmeans=True, meanprops=meanpointprops, capprops=capprops, whiskerprops=whisprops3, patch_artist=True, grid=False)


fire_box.axes.set_xticks(positions_nofire)

red_patch   = mpatches.Patch(color='orangered', label='all fires')
blue_patch  = mpatches.Patch(color='royalblue', label='no fires')
green_patch = mpatches.Patch(color='forestgreen', label='30% fires')
plt.legend(handles=[red_patch, blue_patch, green_patch], frameon=True, ncol=3, loc=2)
ax.legend()
ax.set_xlim([ax.get_xlim()[0]-0.20,ax.get_xlim()[1]])
xticks = [item.get_text() for item in ax.get_xticklabels()]
new_xticks=[]
for x in xticks:
#    new_xticks.append(new_stations.ix[int(x), 'long_name'])
     new_xticks.append(name_for_plot[x])
ax.set_xticklabels(new_xticks, rotation=0, fontsize=15)    
ax.set_zorder(3)
plt.grid(True, zorder=0)
plt.savefig(outdir+'boxPlot_atSelectedSites'+'.png', pad_inches=0.1, bbox_inches='tight')
#%%