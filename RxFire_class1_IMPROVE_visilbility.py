'''
author - vikram, 15/1/2016
modified Aug 3, 2016 for using the plume corrected CCTM files
purpose - calculate the 20% highest and lowest visibility days and compares how different fire case 
affect the visibility during these days
difference between this and RxFire__lowest-highest_visibility.py script is that this calculates the 
deciview from 24h avg RH and 24 h avg concentrations and at each of the IMPROVE sites it calculates both 
modelled and observed deciviews from the modelled and observed concentrations combined with FRH factors 
that were taken from literature (modeled at IMPROVE sites is based on modeled concentration and literature FRH)
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from netCDF4 import Dataset
from math import ceil
import math
import time
mpl.rcParams['font.family'] = 'serif'

#%%
inputDir  = 'C:/Users/vik/Documents/pythonProjects/pythonInput/rxFire/'
cases = ['000', '100', '030']
if not os.path.exists(inputDir):
    print 'no such directory as:', inputDir

class1_file = inputDir + 'AP4_grids_in_Class1_areas_orig.csv'
plotDir = 'C:/Users/vik/Documents/pythonProjects/pythonOutput/rxFire/plots/'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

#%%
# read the files and the calculated deciviews for each day
# open the grid info file

grd_file  = 'C:/Users/vik/Documents/Projects/MCIP_data/2011111100/MCIP/GRIDCRO2D'
grd = Dataset(grd_file,'r')
lat = grd.variables['LAT'][0,0,:,:]
lon = grd.variables['LON'][0,0,:,:]
ht  = grd.variables['HT'][0,0,:,:]

# open the Deciview and RH files
vis_fire_file    = inputDir + 'rx_fire_ncrcatted_avgDCV_RH/' + 'AVG_DAILY_DCV_ncrcat_2011_Fire_100.ncf'
vis_lowfire_file = inputDir + 'rx_fire_ncrcatted_avgDCV_RH/' + 'AVG_DAILY_DCV_ncrcat_2011_Fire_030.ncf'
vis_nofire_file  = inputDir + 'rx_fire_ncrcatted_avgDCV_RH/' + 'AVG_DAILY_DCV_ncrcat_2011_Fire_000.ncf'
rel_humid_file   = inputDir + 'rx_fire_ncrcatted_avgDCV_RH/' + 'AVG_RH_ncrcat_2011_Fire_000.ncf'

vis_fire    = Dataset(vis_fire_file, 'r')
vis_lowfire = Dataset(vis_lowfire_file, 'r')
vis_nofire  = Dataset(vis_nofire_file, 'r')
rh_file     = Dataset(rel_humid_file, 'r')

dv_fire_avg    = vis_fire.variables['DCV_Recon'][:, 0, :, :]
dv_lowfire_avg = vis_lowfire.variables['DCV_Recon'][:, 0, :, :]
dv_nofire_avg  = vis_nofire.variables['DCV_Recon'][:, 0, :, :] 
rh_avg         = rh_file.variables['RH_24hAVG'][:,0,:,:]
nrows = vis_fire.NROWS
ncols = vis_fire.NCOLS
nlays = vis_fire.NLAYS

vis_fire.close()
vis_lowfire.close()
vis_nofire.close()
rh_file.close()
#%%
# get the row and column numbers for the IMPROVE sites
improve_file  = inputDir + 'improve_sites.xlsx'
improve_sites = pd.read_excel(improve_file)
# based on the code at
# http://nbviewer.ipython.org/github/Unidata/unidata-python-workshop/blob/master/netcdf-by-coordinates.ipynb
def naive_fast(latvar,lonvar,lat0,lon0):
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:]
    lonvals = lonvar[:]
    ny,nx = latvals.shape
    dist_sq = (latvals-lat0)**2 + (lonvals-lon0)**2
    minindex_flattened = dist_sq.argmin()  # 1D index of min element
    iy_min,ix_min = np.unravel_index(minindex_flattened, latvals.shape)
    return int(iy_min),int(ix_min)

for i in improve_sites.index:
#    iy,ix = naive_fast(lat, lon, loc[0,0], loc[0,1])
    iy,ix = naive_fast(lat, lon, improve_sites.ix[i, 'Lat'], improve_sites.ix[i, 'Long'])
    improve_sites.ix[i, 'ap4_row'] = iy + 1
    improve_sites.ix[i, 'ap4_col'] = ix + 1
    
#%%
# this block takes 15 minutes to run
stime = time.time()
# read the grid cell info for class 1 areas and create a dataframe for further analysis
class1_grids = pd.read_csv(class1_file)
#columns_to_use = [u'DUMMY_ID', u'COL', u'ROW', u'LAT', u'LON', u'STATE', u'NAME', u'DV_FIRE', u'DV_LOWFIRE', u'DV_NOFIRE']
#df_class1 = pd.DataFrame(columns=columns_to_use, index=pd.date_range(start='2011-10-01', end='2011-11-30', freq='D'))
#dd = class1_grids.groupby('DUMMY_ID')[['COL','ROW']]

dates = pd.date_range(start='2011-10-01', end='2011-11-30', freq='D')
df_dv = pd.DataFrame(index=np.arange(0, len(class1_grids.index)*61, 1))
ind   = 0 # initialize index for writing to df_dv
for i in class1_grids.index:
    # info about the class I area 
    area_code = class1_grids.ix[i, 'DUMMY_ID']
    ap4_row   = class1_grids.ix[i, 'ROW']
    ap4_col   = class1_grids.ix[i, 'COL']
    area_name = class1_grids.ix[i, 'NAME']
    # get the deciview for the grid cell at this index location
    for day in np.arange(0, 61):
#        for case in ['fire', 'lowfire', 'nofire']:
        dv_cell_fire    = dv_fire_avg[day, ap4_row-1, ap4_col-1]
        dv_cell_lowfire = dv_lowfire_avg[day, ap4_row-1, ap4_col-1]
        dv_cell_nofire  = dv_nofire_avg[day, ap4_row-1, ap4_col-1]
        
        # now write to the columns of the dataframe
        df_dv.ix[ind, 'datetime']   = dates[day]
        df_dv.ix[ind, 'area_code']  = area_code
        df_dv.ix[ind, 'area_name']  = area_name
        df_dv.ix[ind, 'ap4_row']    = ap4_row
        df_dv.ix[ind, 'ap4_col']    = ap4_col
        df_dv.ix[ind, 'dv_fire']    = dv_cell_fire
        df_dv.ix[ind, 'dv_lowfire'] = dv_cell_lowfire
        df_dv.ix[ind, 'dv_nofire']  = dv_cell_nofire
        # increment to next index 
        ind += 1

# now let's do the same for improve sites (since not all IMPROVE sites are in the class 1 areas)
df_dv_improve = pd.DataFrame(index=np.arange(0, len(improve_sites.index)*61, 1))
ind   = 0 # initialize index for writing to df_dv
for i in improve_sites.index:
    # info about the class I area 
    site_id   = improve_sites.ix[i, 'SiteID']
    ap4_row   = improve_sites.ix[i, 'ap4_row']
    ap4_col   = improve_sites.ix[i, 'ap4_col']
    site_name = improve_sites.ix[i, 'Sitename2']
    # get the deciview for the grid cell at this index location
    for day in np.arange(0, 61):
        dv_cell_fire    = dv_fire_avg[day, ap4_row-1, ap4_col-1]
        dv_cell_lowfire = dv_lowfire_avg[day, ap4_row-1, ap4_col-1]
        dv_cell_nofire  = dv_nofire_avg[day, ap4_row-1, ap4_col-1]
        
        # now write to the columns of the dataframe
        df_dv_improve.ix[ind, 'datetime']   = dates[day]
        df_dv_improve.ix[ind, 'site_id']    = site_id
        df_dv_improve.ix[ind, 'site_name']  = site_name
        df_dv_improve.ix[ind, 'ap4_row']    = ap4_row
        df_dv_improve.ix[ind, 'ap4_col']    = ap4_col
        df_dv_improve.ix[ind, 'dv_fire']    = dv_cell_fire
        df_dv_improve.ix[ind, 'dv_lowfire'] = dv_cell_lowfire
        df_dv_improve.ix[ind, 'dv_nofire']  = dv_cell_nofire
        # increment to next index 
        ind += 1

etime = time.time()
print 'time taken=', (etime-stime)   
     
#%%
# plotting etc
     # also llook at this http://stackoverflow.com/questions/19280336/overlaying-multiple-histograms-using-pandas
colors_to_use = ['red', 'green', 'blue']
colors_to_use = ['red', 'green', 'blue']
fig, ax = plt.subplots(figsize=(13,7))        
df_dv.ix[:,['dv_fire', 'dv_lowfire', 'dv_nofire']].plot(kind='hist', bins=np.arange(0,45,0.5), logy=True,  
                                                             color=colors_to_use, alpha=0.6, grid=True, 
                                                             fontsize=15, ax=ax)
                                                             
ax.set_title('Distribution of deciview at grid cells in national parks / wilderness areas, Oct - Nov 2011', fontsize=15)
ax.set_xlabel('Deciview',fontsize=15)
ax.set_ylabel('Grid Cell x Days',fontsize=15)
ax.legend(['With Fire', '30% Fire', 'No Fire'], fontsize=15)
plt.savefig(plotDir + 'hist_fire_lowfire_nofire_avgRH..png', pad_inches=0.1, bbox_inches='tight') 
#%%
# lets get % of the area under different qualitatively visibility categories
#fig, ax = plt.subplots(figsize=(13,7))        
dcv_category = [(0,13.999), (13.999,19.999), (19.999, 23.999), (23.999, 27.999), (27.999, 50.999)]
categories = ['Very good', 'Good', 'Moderate', 'Poor', 'Very poor']
dcv_category_count = pd.DataFrame(index=['dv_fire', 'dv_lowfire', 'dv_nofire'], columns=categories)
for case in ['dv_fire', 'dv_lowfire', 'dv_nofire']:
    for i, ctg in enumerate(dcv_category):
        dcv_category_count.ix[case, categories[i]] = df_dv[(df_dv[case]>ctg[0]) & (df_dv[case]<ctg[1])].count()[case]
dcv_category_count.plot(kind='bar', logy=True, grid=True)
dcv_category_percent = dcv_category_count.copy()
dcv_category_percent = dcv_category_percent*100/dcv_category_count.sum(axis=1)['dv_fire']
transposed = dcv_category_percent.transpose()
transposed.plot(kind='bar', logy=True, grid=True, color=['red', 'green', 'blue'], alpha=0.5)
#plt.savefig(plotDir + 'hist_fire_lowfire_nofire_avgRH.png', pad_inches=0.1, bbox_inches='tight') 

#%%
# plot at deciview distribution at each Class 1 area
for area_code in set(df_dv['area_code']):
    df_area   = df_dv.loc[df_dv['area_code']==area_code, df_dv.columns]
    area_name = list(set(df_area['area_name']))[0]
#    fig, ax   = plt.plot()        
    ax = df_area.ix[:,['dv_fire', 'dv_lowfire', 'dv_nofire']].plot(kind='hist', bins=np.arange(0,40,0.5), logy=True, figsize=(13,7), 
                                                                 color=['red', 'green', 'blue'], alpha=0.5, grid=True, 
                                                                 fontsize=15)
    ax.set_title('Distribution of Deciview: '+area_name, fontsize=15)
    ax.set_xlabel('Deciview',fontsize=15)
    ax.set_ylabel('Frequency',fontsize=15)
    x_lower = 0
    x_upper = df_area['dv_fire'].max() + (5-(df_area['dv_fire'].max())%5)
    ax.set_xlim(x_lower, x_upper)
    ax.legend(['Fire', '30% Fire', 'No Fire'], fontsize=15)
    plt.savefig(plotDir + 'hist_'+area_name+'fire_lowfire_nofire_avgRH.png', pad_inches=0.1, bbox_inches='tight')


#%%
def ranked_by_deciview(df, order):
    '''first top 20% or bottom 20% deciview days'''
    df_ordered = pd.DataFrame()
    if ('site_id' in df.columns):
        code = 'site_id'
    elif ('EPACode' in df.columns):
        code = 'EPACode'
        
    if order == 'lower20':
        for site in ap_sites:
            #print site
            df_tmp = df.loc[df[code] == site, df.columns]
            #df_tmp = df_tmp.nsmallest(int(ceil(0.2*df_tmp['dv:Value'].size)), 'dv:Value', keep='first')
            df_tmp = df_tmp.nsmallest(int(ceil(0.2*df_tmp['new_dcv'].size)), 'new_dcv', keep='first')
            df_ordered = pd.concat([df_ordered, df_tmp], join='outer')
    if order == 'upper20':
        for site in ap_sites:
            #print site
            df_tmp = df.loc[df[code] == site, df.columns]
            #df_tmp = df_tmp.nlargest(int(ceil(0.2*df_tmp['dv:Value'].size)), 'dv:Value', keep='first')
            df_tmp = df_tmp.nlargest(int(ceil(0.2*df_tmp['new_dcv'].size)), 'new_dcv', keep='first')
            df_ordered = pd.concat([df_ordered, df_tmp], join='outer')
    return df_ordered
    
#%%
def deciview_from_frh(frh, ammsulfate_a, ammnitrate_a, organic_a, elemental_a, soils_a):
    RAY   = 0.01
    RAY1  = 1.0/RAY
    SCALE = 1.0E-03    
    extinction = SCALE * ( 3.0 * frh * (ammsulfate_a + ammnitrate_a) 
                         + 4.0 * (organic_a) + 10.0 * (elemental_a)
                         + 1.0 * (soils_a)) + RAY
    dcv  = 10.0 * math.log( extinction  * RAY1 )
     #  note if extinction < RAY then DCV is negative.
     #  this implies that visual range is greater than the Rayleigh limit.
     #  The definition of deciviews is based upon the Rayleigh limit
     #  being the maximum visual range thus, set a floor of 0.0 on DCV.
    deciview  = max( 0.0, dcv )
    return deciview
#%%    
monthwise_RH_file  = inputDir + 'monthwise-RH-factors.xlsx'    
monthly_frh = pd.read_excel(monthwise_RH_file)
monthly_frh.set_index('Site ID', inplace=True)
int_to_str_month = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 
                    7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}

# for analysis with respect to the observations
indir = inputDir + 'AQS_2011_AP4_domainData'

ap_sites = [410050010, 320079000, 530410007, 300479000, 410330010, 530090013, 530090020, 530730022, 300819000, 60150002, 410390070, 300899000,
            300499000, 60893003, 61059000, 160370002, 530370004, 60930005, 410610010, 530530014, 410358001, 410630002, 300779000, 300299001, 530390010, 530390011,530470012, 160230101]    
improvefile = indir + '/'+'IMPROVE_2011_withPotassium.txt'
df_tmp = pd.read_csv(improvefile, skiprows=4358)
df_tmp = df_tmp.loc[df_tmp['Dataset']=='IMPRHR2', df_tmp.columns]
chk_site = lambda x: (True if x in ap_sites else False)
for i in df_tmp.index:
    if df_tmp.ix[i, 'EPACode'] in ap_sites:
        df_tmp.ix[i, 'ap_site'] = True
    else:
        df_tmp.ix[i, 'ap_site'] = False
df = df_tmp.loc[df_tmp['ap_site']==True, df_tmp.columns]
df['datetime']   = pd.to_datetime(df['Date']) 

# attach ap4 row and column with the sites for merging later
for i in df.index:
    iy,ix = naive_fast(lat, lon, df.ix[i, 'Latitude'], df.ix[i, 'Longitude'])
    df.ix[i, 'ap4_row'] = iy + 1
    df.ix[i, 'ap4_col'] = ix + 1   
columns_to_use = [u'SiteCode', u'SiteName', u'Latitude', u'Longitude', u'Elevation', u'EPACode', 
                  u'ammNO3f:Value', u'ammSO4f:Value', u'ECf:Value', u'OCf:Value', u'dv:Value', 
                  u'OMCf:Value', u'SOILf:Value', u'aerosol_bext:Value', 
                  u'total_bext:Value', u'datetime', u'ap4_row', u'ap4_col']
df = df.loc[:, columns_to_use]

for idx in df.index:
    datestamp = df.ix[idx, 'datetime'].to_datetime()
    month = datestamp.month
    month = int_to_str_month[month]
    ammNO3  = df.ix[idx, u'ammNO3f:Value']
    ammSO4  = df.ix[idx, u'ammSO4f:Value']
    ECf     = df.ix[idx, u'ECf:Value']
    OMCf    = df.ix[idx, u'OCf:Value']
    soilf   = df.ix[idx, u'OCf:Value']
    epacode = df.ix[idx, u'EPACode']
    frh     = monthly_frh.ix[epacode, month]
        
    if (ammNO3>=0 and ammSO4>=0 and ECf>=0 and OMCf>=0 and soilf>=0):
        new_dv  = deciview_from_frh(frh, ammSO4, ammNO3, OMCf, ECf, soilf)
    else:
        new_dv  = np.nan
        
    df.ix[idx, 'new_dcv'] = new_dv

#df1 = df.set_index('datetime')
#df2 = df1.loc[df1['dv:Value']>0]
#for epacode in list(set(df2.EPACode)):
#    df3 = df2.loc[df2['EPACode']==epacode]
#    df3[['dv:Value', 'new_dcv']].plot(kind='line')        
#
#inputDir  = 'C:/Users/vik/Documents/rx_fire_ncrcatted_avgDCV_RH/'
#cases = ['000', '100', '030']
#if not os.path.exists(inputDir):
#    print 'no such directory as:', inputDir
#
#class1_file = inputDir + 'AP4_grids_in_Class1_areas_orig.csv'
#plotDir = 'C:/Users/vik/Documents/Projects/prescribed_fires/plots/'
#if not os.path.exists(plotDir):
#    os.makedirs(plotDir)
df_oo = df.copy()
df_r = df_oo.loc[:, ['new_dcv','datetime', 'SiteName', 'dv:Value','ammSO4f:Value', 'ammSO4f_bext:Value', 'SO4f_bext:Value', 'ammNO3f:Value', 'OCf:Value', 'OMCf_bext:Value', 'ECf:Value', 'SO4f:Value', 'SeaSaltf:Value', 'EPACode', 'ap4_row', 'ap4_col']]
df_r.loc[df_r['ammSO4f:Value'] < 0] = np.nan 
df_r.loc[df_r['ammSO4f_bext:Value'] < 0] = np.nan 
df_r.loc[df_r['SO4f:Value'] < 0] = np.nan 
df_r.loc[df_r['SO4f_bext:Value'] < 0] = np.nan 
df_r.loc[df_r['ammNO3f:Value'] < 0] = np.nan
df_r.loc[df_r['OCf:Value'] < 0] = np.nan
df_r.loc[df_r['OMCf_bext:Value'] < 0] = np.nan
df_r.loc[df_r['ECf:Value'] < 0] = np.nan
df_r.loc[df_r['SeaSaltf:Value'] < 0] = np.nan
df_r.loc[df_r['dv:Value'] < 0] = np.nan
df_r['OC/EC'] = df_r['OCf:Value']/df_r['ECf:Value']

df_rn = df_r.loc[:, ['new_dcv','datetime','OC/EC', 'dv:Value','ammSO4f:Value', 'OCf:Value', 'ECf:Value', 'SO4f:Value', 'EPACode', 'SO4f_bext:Value', 'OMCf_bext:Value', 'ap4_row', 'ap4_col']]
#df_rn = df_rn.dropna()

df_highest = ranked_by_deciview(df_rn, 'upper20')
df_lowest  = ranked_by_deciview(df_rn, 'lower20')
#%%
# read the species concentration files and calculate the average deciviews for each day
# open the grid info file

grd_file  = 'C:/Users/vik/Documents/Projects/MCIP_data/2011111100/MCIP/GRIDCRO2D'
grd = Dataset(grd_file,'r')
lat = grd.variables['LAT'][0,0,:,:]
lon = grd.variables['LON'][0,0,:,:]
ht  = grd.variables['HT'][0,0,:,:]

concDir = inputDir + 'rx_fire_ncrcatted_avg_concentration/'
cases = ['000', '100', '030']
if not os.path.exists(inputDir):
    print 'no such directory as:', inputDir
    
# open the concentration files
conc_fire_file    = concDir + 'AVG_DAILY_PM25_SPECIES_for_DCV_ncrcat_2011_Fire_100.ncf'
conc_lowfire_file = concDir + 'AVG_DAILY_PM25_SPECIES_for_DCV_ncrcat_2011_Fire_030.ncf'
conc_nofire_file  = concDir + 'AVG_DAILY_PM25_SPECIES_for_DCV_ncrcat_2011_Fire_000.ncf'

conc_fire    = Dataset(conc_fire_file, 'r')
conc_lowfire = Dataset(conc_lowfire_file, 'r')
conc_nofire  = Dataset(conc_nofire_file, 'r')

#for case in ['fire', 'low_fire', 'nofire']:
#for var in ['ANH4_DAILY_AVG', 'ASO4_DAILY_AVG', 'ANO3_DAILY_AVG', 'EC_DAILY_AVG', 'OA_DAILY_AVG', 'SOIL_DAILY_AVG']:
fire_avg_nh4  = conc_fire.variables['ANH4_DAILY_AVG'][:,0,:,:]
fire_avg_so4  = conc_fire.variables['ANH4_DAILY_AVG'][:,0,:,:]
fire_avg_no3  = conc_fire.variables['ANH4_DAILY_AVG'][:,0,:,:]
fire_avg_ec   = conc_fire.variables['ANH4_DAILY_AVG'][:,0,:,:]
fire_avg_oa   = conc_fire.variables['ANH4_DAILY_AVG'][:,0,:,:]
fire_avg_soil = conc_fire.variables['ANH4_DAILY_AVG'][:,0,:,:]

lowfire_avg_nh4  = conc_lowfire.variables['ANH4_DAILY_AVG'][:,0,:,:]
lowfire_avg_so4  = conc_lowfire.variables['ANH4_DAILY_AVG'][:,0,:,:]
lowfire_avg_no3  = conc_lowfire.variables['ANH4_DAILY_AVG'][:,0,:,:]
lowfire_avg_ec   = conc_lowfire.variables['ANH4_DAILY_AVG'][:,0,:,:]
lowfire_avg_oa   = conc_lowfire.variables['ANH4_DAILY_AVG'][:,0,:,:]
lowfire_avg_soil = conc_lowfire.variables['ANH4_DAILY_AVG'][:,0,:,:]

nofire_avg_nh4  = conc_nofire.variables['ANH4_DAILY_AVG'][:,0,:,:]
nofire_avg_so4  = conc_nofire.variables['ANH4_DAILY_AVG'][:,0,:,:]
nofire_avg_no3  = conc_nofire.variables['ANH4_DAILY_AVG'][:,0,:,:]
nofire_avg_ec   = conc_nofire.variables['ANH4_DAILY_AVG'][:,0,:,:]
nofire_avg_oa   = conc_nofire.variables['ANH4_DAILY_AVG'][:,0,:,:]
nofire_avg_soil = conc_nofire.variables['ANH4_DAILY_AVG'][:,0,:,:]
    
nrows = conc_fire.NROWS
ncols = conc_fire.NCOLS
nlays = conc_fire.NLAYS

conc_fire.close()
conc_lowfire.close()
conc_nofire.close()
grd.close()
#%%
stime = time.time()

# now let's do the same for improve sites (since not all IMPROVE sites are in the class 1 areas)
dates = pd.date_range(start='2011-10-01', end='2011-11-30', freq='D')
df_dcv_improve = pd.DataFrame(index=np.arange(0, len(improve_sites.index)*61, 1))
ind   = 0 # initialize index for writing to df_dv

# define an array to store the conc of 
# ['ANH4_DAILY_AVG', 'ASO4_DAILY_AVG', 'ANO3_DAILY_AVG', 'EC_DAILY_AVG', 'OA_DAILY_AVG', 'SOIL_DAILY_AVG']
# the columns for each are 0,1,2,3,4,5 respectively
conc_cell_fire    = np.zeros((6)) 
conc_cell_lowfire = np.zeros((6)) 
conc_cell_nofire  = np.zeros((6)) 

for i in improve_sites.index:
    # info about the class I area 
    site_id   = improve_sites.ix[i, 'SiteID']
    ap4_row   = improve_sites.ix[i, 'ap4_row']
    ap4_col   = improve_sites.ix[i, 'ap4_col']
    site_name = improve_sites.ix[i, 'Sitename2']
    # get the deciview for the grid cell at this index location
    for day in np.arange(0, 61):
        conc_cell_fire[0]    = fire_avg_nh4[day, ap4_row-1, ap4_col-1]
        conc_cell_fire[1]    = fire_avg_so4[day, ap4_row-1, ap4_col-1]
        conc_cell_fire[2]    = fire_avg_no3[day, ap4_row-1, ap4_col-1]
        conc_cell_fire[3]    = fire_avg_ec[day, ap4_row-1, ap4_col-1]
        conc_cell_fire[4]    = fire_avg_oa[day, ap4_row-1, ap4_col-1]
        conc_cell_fire[5]    = fire_avg_soil[day, ap4_row-1, ap4_col-1]
        
        conc_cell_lowfire[0] = lowfire_avg_nh4[day, ap4_row-1, ap4_col-1]
        conc_cell_lowfire[1] = lowfire_avg_so4[day, ap4_row-1, ap4_col-1]
        conc_cell_lowfire[2] = lowfire_avg_no3[day, ap4_row-1, ap4_col-1]
        conc_cell_lowfire[3] = lowfire_avg_ec[day, ap4_row-1, ap4_col-1]
        conc_cell_lowfire[4] = lowfire_avg_oa[day, ap4_row-1, ap4_col-1]
        conc_cell_lowfire[5] = lowfire_avg_soil[day, ap4_row-1, ap4_col-1]
        
        conc_cell_nofire[0]  = nofire_avg_nh4[day, ap4_row-1, ap4_col-1]
        conc_cell_nofire[1]  = nofire_avg_so4[day, ap4_row-1, ap4_col-1]
        conc_cell_nofire[2]  = nofire_avg_no3[day, ap4_row-1, ap4_col-1]
        conc_cell_nofire[3]  = nofire_avg_ec[day, ap4_row-1, ap4_col-1]
        conc_cell_nofire[4]  = nofire_avg_oa[day, ap4_row-1, ap4_col-1]
        conc_cell_nofire[5]  = nofire_avg_soil[day, ap4_row-1, ap4_col-1]
        
        if (day >= 0  and day <=30):
            month = 10
        elif (day>30):
            month =11
        month = int_to_str_month[month]
        frh   = monthly_frh.ix[site_id, month]        
        dcv_cell_fire    = deciview_from_frh(frh, (2.0/3.0)*conc_cell_fire[0] + conc_cell_fire[1], 
                                            (1.0/3.0)*conc_cell_fire[0] + conc_cell_fire[2], 
                                            conc_cell_fire[3], conc_cell_fire[4], conc_cell_fire[5])
                                         
        dcv_cell_lowfire = deciview_from_frh(frh, (2.0/3.0)*conc_cell_lowfire[0] + conc_cell_lowfire[1], 
                                            (1.0/3.0)*conc_cell_lowfire[0] + conc_cell_lowfire[2], 
                                            conc_cell_lowfire[3], conc_cell_lowfire[4], conc_cell_lowfire[5])
                                         
        dcv_cell_nofire = deciview_from_frh(frh, (2.0/3.0)*conc_cell_nofire[0] + conc_cell_nofire[1], 
                                           (1.0/3.0)*conc_cell_nofire[0] + conc_cell_nofire[2], 
                                           conc_cell_nofire[3], conc_cell_nofire[4], conc_cell_nofire[5])                                         

                                          
        # now write to the columns of the dataframe
        df_dcv_improve.ix[ind, 'datetime']   = dates[day]
        df_dcv_improve.ix[ind, 'site_id']    = site_id
        df_dcv_improve.ix[ind, 'site_name']  = site_name
        df_dcv_improve.ix[ind, 'ap4_row']    = ap4_row
        df_dcv_improve.ix[ind, 'ap4_col']    = ap4_col
        df_dcv_improve.ix[ind, 'dv_fire']    = dcv_cell_fire
        df_dcv_improve.ix[ind, 'dv_lowfire'] = dcv_cell_lowfire
        df_dcv_improve.ix[ind, 'dv_nofire']  = dcv_cell_nofire
        
        # increment to next index 
        ind += 1

etime = time.time()
print 'time taken=', (etime-stime)   
     
#%%
# plot 20% highest and 20% lowest days 
# merge with the fire DF for extracting only IMPROVE site cells
df_improve_high = pd.merge(df_dcv_improve, df_highest, on=['ap4_row', 'ap4_col', 'datetime'])    
df_improve_low  = pd.merge(df_dcv_improve, df_lowest, on=['ap4_row', 'ap4_col', 'datetime'])    

df_improve_high['bias'] = df_improve_high['dv_fire'] - df_improve_high['new_dcv']
df_improve_low['bias']  = df_improve_low['dv_fire'] - df_improve_low['new_dcv']

r_fire    = df_improve_high.corr().ix['new_dcv', 'dv_fire'] # first calculate correlation matrix and then get relevant data
r_lowfire = df_improve_high.corr().ix['new_dcv', 'dv_lowfire']
r_nofire  = df_improve_high.corr().ix['new_dcv', 'dv_nofire']

label_fire    = '%s%.2f%s'%('with fire (r=', r_fire,  ')')
label_lowfire = '%s%.2f%s'%('30% fire (r=', r_lowfire,  ')')
label_nofire  = '%s%.2f%s'%('no fire    (r=', r_nofire,')')
#label_fire    = '%s'%('with fire')
#label_lowfire = '%s'%('30% fire')
#label_nofire  = '%s'%('no fire ')

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
df_improve_high.plot(kind='scatter', x='new_dcv', y='dv_fire', marker='D', label=label_fire, s=50, alpha=0.6,color='red', ax=ax1, fontsize=15)
#df_improve_high.plot(kind='scatter', x='dv:Value', y='dv_lowfire', marker='s',label=label_lowfire, s=50, alpha=0.6,color='yellow', ax=ax1, fontsize=15)
df_improve_high.plot(kind='scatter', x='new_dcv', y='dv_nofire', marker='o',label=label_nofire, s=50,alpha=0.6, color='blue', ax=ax1, fontsize=15)
#ax1.set_title('Deciview at IMPROVE sites during 20% highest DV days in 2011', fontsize=15)
ax1.set_title('20% highest DV days in 2011', fontsize=15)
ax1.set_xlabel('Deciview Observed',fontsize=15)
ax1.set_ylabel('Deciview Modeled',fontsize=15)
ax1.set_xlim(0,35)
ax1.set_ylim(0,35)
ax1.legend(loc=1, fontsize=15, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
#ax1.legend(['Fire', '30% Fire', 'No Fire'], fontsize=15)

r_fire    = df_improve_low.corr().ix['new_dcv', 'dv_fire'] # first calculate correlation matrix and then get relevant data
r_lowfire = df_improve_low.corr().ix['new_dcv', 'dv_lowfire']
r_nofire  = df_improve_low.corr().ix['new_dcv', 'dv_nofire']

label_fire    = '%s%.2f%s'%('with fire (r=', r_fire,  ')')
label_lowfire = '%s%.2f%s'%('30% fire (r=', r_lowfire,  ')')
label_nofire  = '%s%.2f%s'%('no fire    (r=', r_nofire,')')
#label_fire    = '%s'%('with fire')
#label_lowfire = '%s'%('30% fire')
#label_nofire  = '%s'%('no fire ')

df_improve_low.plot(kind='scatter', x='new_dcv', y='dv_fire', marker='D', label=label_fire, s=50, alpha=0.6, color='red', ax=ax2, fontsize=15)
#df_improve_low.plot(kind='scatter', x='new_dcv', y='dv_lowfire', marker='s', label=label_lowfire, s=50, alpha=0.6, color='yellow', ax=ax2, fontsize=15)
df_improve_low.plot(kind='scatter', x='new_dcv', y='dv_nofire', marker='o', label=label_nofire, s=50, alpha=0.6, color='blue', ax=ax2, fontsize=15)

#ax2.set_title('Deciview at IMPROVE sites during 20% lowest DV days in 2011', fontsize=15)
ax2.set_title('20% lowest DV days in 2011', fontsize=15)
ax2.set_xlabel('Deciview Observed',fontsize=15)
ax2.set_ylabel('Deciview Modeled',fontsize=15)
ax2.set_xlim(0,12)
ax2.set_ylim(0,12)
ax2.legend(loc=1, fontsize=15, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
plt.savefig(plotDir + 'improvesites_20_percent_dv_fire_lowfire_nofire_literature_FRH.png', pad_inches=0.1, bbox_inches='tight') 

#%%
# plot all days and 20% highesest
# merge with the fire DF for extracting only IMPROVE site cells
df_improve_all  = pd.merge(df_dcv_improve, df_rn, on=['ap4_row', 'ap4_col', 'datetime'])    
df_improve_all['bias'] = df_improve_all['dv_fire'] - df_improve_all['new_dcv']

r_fire    = df_improve_all.corr().ix['new_dcv', 'dv_fire'] # first calculate correlation matrix and then get relevant data
r_lowfire = df_improve_all.corr().ix['new_dcv', 'dv_lowfire']
r_nofire  = df_improve_all.corr().ix['new_dcv', 'dv_nofire']

label_fire    = '%s%.2f%s'%('with fire (r=', r_fire,  ')')
label_nofire  = '%s%.2f%s'%('no fire    (r=', r_nofire,')')

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
df_improve_all.plot(kind='scatter', x='new_dcv', y='dv_fire', marker='D', label=label_fire, s=50, alpha=0.6,color='red', ax=ax1, fontsize=15)
df_improve_all.plot(kind='scatter', x='new_dcv', y='dv_nofire', marker='o',label=label_nofire, s=50,alpha=0.6, color='blue', ax=ax1, fontsize=15)
ax1.set_title('All days in October-November 2011', fontsize=15)
ax1.set_xlabel('Deciview Observed',fontsize=15)
ax1.set_ylabel('Deciview Modeled',fontsize=15)
ax1.set_xlim(0,35)
ax1.set_ylim(0,35)
ax1.legend(loc=1, fontsize=15, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)

r_fire    = df_improve_high.corr().ix['new_dcv', 'dv_fire'] # first calculate correlation matrix and then get relevant data
r_lowfire = df_improve_high.corr().ix['new_dcv', 'dv_lowfire']
r_nofire  = df_improve_high.corr().ix['new_dcv', 'dv_nofire']

label_fire    = '%s%.2f%s'%('with fire (r=', r_fire,  ')')
label_lowfire = '%s%.2f%s'%('30% fire (r=', r_lowfire,  ')')
label_nofire  = '%s%.2f%s'%('no fire    (r=', r_nofire,')')

df_improve_high.plot(kind='scatter', x='new_dcv', y='dv_fire', marker='D', label=label_fire, s=50, alpha=0.6, color='red', ax=ax2, fontsize=15)
df_improve_high.plot(kind='scatter', x='new_dcv', y='dv_nofire', marker='o', label=label_nofire, s=50, alpha=0.6, color='blue', ax=ax2, fontsize=15)
ax2.set_title('20% highest DV days in October-November 2011', fontsize=15)
ax2.set_xlabel('Deciview Observed',fontsize=15)
ax2.set_ylabel('Deciview Modeled',fontsize=15)
ax2.set_xlim(0,35)
ax2.set_ylim(0,35)
ax2.legend(loc=1, fontsize=15, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
#plt.savefig(plotDir + 'improvesites_all_and_20_percent_highest_dv_fire_lowfire_nofire_literature_FRH.png', pad_inches=0.1, bbox_inches='tight')
#%%
def add_sitename(dff):
    ''''attach site name to each site id'''
    df = dff.copy()
    for i in df.index:
        df.ix[i, 'Sitename'] = monthly_frh.ix[int(i), 'Sitename']
    return df
    
# merge with the fire DF for extracting only IMPROVE site cells
# this is different from previous cell since we want a year long time series here
# to calculate annual average and 20% highest / lowest deciview days
df_obs = df_rn.loc[:, [u'datetime',u'new_dcv', u'EPACode', u'ap4_row', u'ap4_col']]
df_obs.rename(columns={'EPACode': 'site_id'}, inplace=True)
df_yearly = pd.merge(df_dcv_improve, df_obs, on=['ap4_row', 'ap4_col', 'datetime', 'site_id'], how='outer')    
for i in df_yearly.index:
    for col in ['dv_fire', 'dv_lowfire', 'dv_nofire']:
        if np.isnan(df_yearly.ix[i, col]):
            df_yearly.ix[i, col] = df_yearly.ix[i, 'new_dcv']

df_year = df_yearly.dropna(subset=['new_dcv'])  # all rows with dv:Value == nan dropped          
df_year['dv_fire-nofire']    = df_year['dv_fire']    - df_year['dv_nofire']
df_year['dv_fire-lowfire']   = df_year['dv_fire']    - df_year['dv_lowfire']
df_year['dv_lowfire-nofire'] = df_year['dv_lowfire'] - df_year['dv_nofire']

df_year_benefit = df_year.groupby(by='site_id').mean()
df_year_benefit['count'] = df_year.groupby(by='site_id')['new_dcv'].count()
df_year_benefit = add_sitename(df_year_benefit)
df_year_benefit.to_csv(plotDir + 'yearly_dv_benefit_literatureFRH.csv')
# benefits during highest and lowest visibility days
df_yearly_high  = ranked_by_deciview(df_year, 'upper20')
df_yearly_low   = ranked_by_deciview(df_year, 'lower20')
df_yearly_high_benefit = df_yearly_high.groupby(by='site_id').mean()
df_yearly_low_benefit  = df_yearly_low.groupby(by='site_id').mean()
df_yearly_high_benefit['count'] = df_yearly_high.groupby(by='site_id')['new_dcv'].count()
df_yearly_low_benefit['count']  = df_yearly_low.groupby(by='site_id')['new_dcv'].count()
df_yearly_high_benefit = add_sitename(df_yearly_high_benefit)
df_yearly_low_benefit = add_sitename(df_yearly_low_benefit)
df_yearly_high_benefit.to_csv(plotDir + '20_percent_highest_dv_benefit_literatureFRH.csv')
df_yearly_low_benefit.to_csv(plotDir + '20_percent_lowest_dv_benefit_literatureFRH.csv')     
#%%    
