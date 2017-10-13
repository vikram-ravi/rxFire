"""
Created on November 29, 2015

@author: vikram
"""
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from netCDF4 import Dataset, MFDataset
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib
import os
matplotlib.rcParams['font.family'] = 'serif' 
#%%
# define case - options are select / aqs / improve
case = 'select'
workdir   = '/home/vikram.ravi/scripts_projects/RxFire_atSelectedSites/'
metdir    = '/home/vikram.ravi/mcip_data/'
indirfire = '/fastscratch/vikram.ravi/airpact4/rerun/'              # both the fire files
indirbase = '/data-failing/part2/vikram.ravi/airpact4/rerun/'       # without fire files
outdir  = workdir + '/'+ case
if not os.path.exists(outdir):
   os.makedirs(outdir)
fn  = metdir + '/2011111100/MCIP/GRIDCRO2D'
grd = Dataset(fn,'r')
lat = grd.variables['LAT'][0,0,:,:]
lon = grd.variables['LON'][0,0,:,:]
ht  = grd.variables['HT'][0,0,:,:]

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

#outFile = workdir + 'aqsid_withRowCol.txt'
outFile = st_file.split('.csv')[0] + '_withRowCol.csv'
stations.to_csv(outFile)

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
# MFDataset below reads in data from multiple netcdf files in a single file
# soome of it based on http://unidata.github.io/netcdf4-python/#netCDF4.MFDataset.isopen
#create a list of nc files
ncfiles_fire = []
ncfiles_lowfire = []
ncfiles_nofire = []
for m in [10, 11]:
    if m==10:
        days = np.arange(1,32)
    elif m==11:
        days = np.arange(1,31)

    for d in days:
        if d < 10:
            strd = '0'+str(d)
        else:
            strd = str(d)
        if m==10:
            doy = 274+d-1
        elif m==11:
            doy = 274+31-1+d 
        file_fire    = indirfire + '2011_Fire_100/' + '2011'+str(m)+strd + '00/POST/CCTM/ACONC_PM25_L01_2011'+str(doy)+'.ncf'
        file_lowfire = indirfire + '2011_Fire_030/' + '2011'+str(m)+strd + '00/POST/CCTM/ACONC_PM25_L01_2011'+str(doy)+'.ncf'
        file_nofire  = indirbase + '2011_Fire_000/' + '2011'+str(m)+strd + '00/POST/CCTM/ACONC_PM25_L01_2011'+str(doy)+'.ncf'
        #print (file_fire)
        #print (file_lowfire)
        ncfiles_fire.append(file_fire)
        ncfiles_lowfire.append(file_lowfire)
        ncfiles_nofire.append(file_nofire)
#met = MFDataset(indir+'2011*/MCIP/METCRO2D','r')
#ncfiles_fire.append('/data/vikram.ravi/airpact4/rerun/2011_Fire_000/2011113000/POST/CCTM/ACONC_PM25_L01_2011334.ncf')
#ncfiles_lowfire.append('/data/vikram.ravi/airpact4/rerun/2011_Fire_000/2011113000/POST/CCTM/ACONC_PM25_L01_2011334.ncf')
#ncfiles_nofire.append('/data/vikram.ravi/airpact4/rerun/2011_Fire_000/2011113000/POST/CCTM/ACONC_PM25_L01_2011334.ncf')

# now read all the files as a single netcdf file and extract variables
firedata     = MFDataset(ncfiles_fire,'r')
lowfiredata  = MFDataset(ncfiles_lowfire,'r')
nofiredata   = MFDataset(ncfiles_nofire,'r')
pmfire       = firedata.variables['PM25'][:,0,:,:]
pmlowfire    = lowfiredata.variables['PM25'][:,0,:,:]
pmnofire     = nofiredata.variables['PM25'][:,0,:,:]
tstamps      = firedata.variables['TFLAG'][:,0,:]


dateTime = jul_to_greg(tstamps[:])
#print (dateTime)
#date = lambda i: (dateTime[i] if i < len(dateTime) else datetime.strftime((datetime.strptime(dateTime[0], '%Y-%m-%d %H:%M')+timedelta(hours=len(dateTime))), '%Y-%m-%d %H:%M'))
date = lambda i: (dateTime[i] if i < len(dateTime) else datetime.strftime((datetime.strptime(dateTime[-1], '%Y-%m-%d %H:%M')+timedelta(hours=24)), '%Y-%m-%d %H:%M'))
#date = lambda i: (dateTime[i] if i < len(dateTime) else datetime.strftime((datetime.strptime(dateTime[-1], '%Y-%m-%d %H:%M'), '%Y-%m-%d %H:%M'))
print (pmnofire[:,0,0].shape)
print (len(dateTime))

#%%
if case == 'select':
   r=0; c=0
   fig, ax = plt.subplots(7, 2, figsize=(12, 12))
   for i in stations.index:
     # print (pmfire[:,0,0].shape)
     # print (len(dateTime))
      ax[r,c].plot(range(len(dateTime)), pmfire[:, stations.ix[i, 'row']   , stations.ix[i, 'column']], 'r-' , label='100% Fire')
      ax[r,c].plot(range(len(dateTime)), pmlowfire[:, stations.ix[i, 'row'], stations.ix[i, 'column']], 'b-', label='30% Fire')
      ax[r,c].plot(range(len(dateTime)), pmnofire[:, stations.ix[i, 'row'] , stations.ix[i, 'column']], 'k-', label='No Fire')
      if (r==0 and c==0):
         ax[r,c].legend(fontsize=8)
      ax[r,c].text(0.02,0.88, stations.ix[i, 'long_name'], fontsize=8, bbox=dict(facecolor='white', alpha=0.8, linewidth=0.0), transform=ax[r,c].transAxes)
      ax[r,c].set_xlim(0, len(dateTime))
      fig.canvas.draw()
      tic = [item.get_text() for item in ax[r,c].get_xticklabels()]
      
      if i==0: print(tic)
      new_tic = [date(int(a)).split(' ')[0][5:] for a in tic]
      ax[r,c].set_xticklabels(new_tic, fontsize=8, rotation='0')
      ax[r,c].set_yticklabels([item.get_text() for item in ax[r,c].get_yticklabels()], fontsize=8)
      ax[r,c].set_ylabel('PM$_{2.5}$, $\mu$g/m${^3}$', fontsize=10)
      ax[r,c].grid(axis='both')
      c = c+1
      if c>1:c=0
      if c==1:
         r=r
      else:
         r=r+1
      if r==6:
         ax[r,c].set_xlabel('Date in 2011', fontsize=10)
   plt.savefig(outdir +'/'+'all_sites.png', pad_inches=0.1, bbox_inches='tight')

elif case == 'aqs' or case == 'improve':
   for i in stations.index:
      fig, ax = plt.subplots(1, 1, figsize=(5, 3))
      ax.plot(range(len(dateTime)), pmfire[:, stations.ix[i, 'row'], stations.ix[i, 'column']], 'r-', label='Fire')
      ax.plot(range(len(dateTime)), pmlowfire[:, stations.ix[i, 'row'], stations.ix[i, 'column']], 'b-', label='30% Fire')
      ax.text(0.02,0.90, stations.ix[i, 'long_name'], fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0.0), transform=ax.transAxes)
      fig.canvas.draw()
      tic = [item.get_text() for item in ax.get_xticklabels()]
      if i==0: print(tic)
      new_tic = [date(int(a)).split(' ')[0][5:] for a in tic]
      ax.set_xticklabels(new_tic, fontsize=10, rotation='0')
      ax.set_yticklabels([item.get_text() for item in ax.get_yticklabels()], fontsize=10)
      ax.set_ylabel('PM$_{2.5}$, $\mu$g/m${^3}$', fontsize=10)
      #ax.set_title(stations.ix[i, 'long_name'], fontsize=10)
      ax.grid(axis='both')
      ax.set_xlabel('Date in 2011', fontsize=10)
      plt.savefig(outdir +'/'+ stations.ix[i, 'long_name'] + '.png', pad_inches=0.1, bbox_inches='tight')
