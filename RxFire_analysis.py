'''
author - vikram, 12/10/2015
purpose - it calculates various stats (FB/FE; MB/ME etc) for IMPROVE sites, 
currently it only considers the IMPROVE dataset for PM2.5, 
but for complete analysis, needs to include AQS data also for PM2.5
AQS analysis added
modified Aug 6, 2016 for analyzing to use the plume corrected reruns CCTM data
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import math
from datetime import datetime, timedelta

mpl.rcParams['font.family'] = 'serif'

#%%
inputDir  = 'C:/Users/vik/Documents/pythonProjects/pythonInput/rxFire/'
obsDir    = inputDir + '/EPA_AQ_Data_for_AIRPACT/'

cases = ['000', '100']

plotDir = 'C:/Users/vik/Documents/pythonProjects/pythonOutput/rxFire/plots/'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

#%% 
#---------IMPROVE SITES----------
# below for IMRPOVE Sites
file_list = []
for case in cases:
    for month in ['10', '11']:
        file_case = inputDir + 'improveSpeciation/' + '2011_Fire_' + case + "/sitecmp_improve_csn_2011"+month+".csv"
        file_list.append(file_case)

col_names = [u'SiteId', u'Latitude', u'Longitude', u'Column', u'Row', u'Time On',
       u'Time Off', u'PM_OC_Obs', u'PM_OC_Mod', u'PM_EC_Obs', u'PM_EC_Mod', u'PM25_Obs', u'PM25_Mod', 
       u'PM_NO3_Obs', u'PM_NO3_Mod', u'PM_SO4_Obs', u'PM_SO4_Mod', u'PM_NH4_Obs', u'PM_NH4_Mod']

df_fire = pd.DataFrame()
df_nofire = pd.DataFrame()
for f in file_list:
    if '100' in f:
        df_tmp = pd.read_csv(f, skiprows=np.arange(0,6), names=col_names, na_values=-999)
        df_fire = pd.concat([df_fire, df_tmp], ignore_index=True)
    elif '000' in f:
        df_tmp = pd.read_csv(f, skiprows=np.arange(0,6), names=col_names, na_values=-999)
        df_nofire = pd.concat([df_nofire, df_tmp], ignore_index=True)

for idx in df_fire.index:
    for species in ['PM25', 'PM_OC', 'PM_EC', 'PM_SO4', 'PM_NH4', 'PM_NO3']:
        if (df_fire.ix[idx, species+'_Obs'] == 0.0):
            df_fire.ix[idx, species+'_Obs'] = np.nan
for idx in df_nofire.index:
    for species in ['PM25', 'PM_OC', 'PM_EC', 'PM_SO4', 'PM_NH4', 'PM_NO3']:
        if (df_nofire.ix[idx, species+'_Obs'] == 0.0):
            df_nofire.ix[idx, species+'_Obs'] = np.nan
        
for species in ['PM25', 'PM_OC', 'PM_EC', 'PM_SO4', 'PM_NH4', 'PM_NO3']:
    df_fire[species+'_mean_conc']    = 0.5*(df_fire[species + '_Mod'] + df_fire[species + '_Obs'])
    df_nofire[species+'_mean_conc']  = 0.5*(df_nofire[species + '_Mod'] + df_nofire[species + '_Obs'])
    
    df_fire[species+'_bias']  = df_fire[species + '_Mod']         - df_fire[species + '_Obs']
    df_fire[species+'_error'] = abs(df_fire[species + '_Mod']     - df_fire[species + '_Obs'])
    df_fire[species+'_fb']    = 200*(df_fire[species + '_Mod']    - df_fire[species + '_Obs'])/(df_fire[species + '_Mod'] + df_fire[species + '_Obs'])
    df_fire[species+'_fe']    = 200*abs(df_fire[species + '_Mod'] - df_fire[species + '_Obs'])/(df_fire[species + '_Mod'] + df_fire[species + '_Obs'])
    df_fire[species+'_nb']    = 100*(df_fire[species + '_Mod']    - df_fire[species + '_Obs'])/(df_fire[species + '_Obs'])
    df_fire[species+'_ne']    = 100*abs(df_fire[species + '_Mod'] - df_fire[species + '_Obs'])/(df_fire[species + '_Obs'])
    df_fire[species+'_sqe']   = (df_fire[species + '_Mod'] - df_fire[species + '_Obs'])**2
    
    df_nofire[species+'_bias']  = df_nofire[species + '_Mod']         - df_nofire[species + '_Obs']
    df_nofire[species+'_error'] = abs(df_nofire[species + '_Mod']     - df_nofire[species + '_Obs'])
    df_nofire[species+'_fb']    = 200*(df_nofire[species + '_Mod']    - df_nofire[species + '_Obs'])/(df_nofire[species + '_Mod'] + df_nofire[species + '_Obs'])
    df_nofire[species+'_fe']    = 200*abs(df_nofire[species + '_Mod'] - df_nofire[species + '_Obs'])/(df_nofire[species + '_Mod'] + df_nofire[species + '_Obs'])
    df_nofire[species+'_nb']    = 100*(df_nofire[species + '_Mod']    - df_nofire[species + '_Obs'])/(df_nofire[species + '_Obs'])
    df_nofire[species+'_ne']    = 100*abs(df_nofire[species + '_Mod'] - df_nofire[species + '_Obs'])/(df_nofire[species + '_Obs'])
    df_nofire[species+'_sqe']   = (df_nofire[species + '_Mod'] - df_nofire[species + '_Obs'])**2
    
# calculate the mean of different stats for each site
df_nofire_avg = df_nofire.groupby(by='SiteId').mean()  #.add_prefix('mean_')
df_fire_avg   = df_fire.groupby(by='SiteId').mean()  #.add_prefix('mean_')

# write the stats to a file
species = ['PM25', 'PM_OC', 'PM_EC', 'PM_SO4', 'PM_NH4', 'PM_NO3']
stats   = [u'Obs', u'Mod', u'bias', u'error', u'fb', u'fe', u'nb', u'ne', u'sqe']

#%%
# open file
dfs = [df_nofire, df_fire]
name= {0:'improve_nofire', 1:'improve_fire'}
for i,d in enumerate(dfs):
    d = d.copy()
    stats_file = plotDir + name[i] + '_'+'mean_stats_file.csv'
    outfile = open(stats_file, 'w')
    outLine = '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n'%('species', 'Number observed', 'mean_obs', 'mean_mod', 'mean_bias', 'mean_error', 'mean_fracBias', 'mean_fracError', 'mean_normBias', 'mean_normError', 'rmse')
    outfile.write(outLine)
    
    for sp in species:
        stats_list = []
        outfile.write('%s,%s,'% (sp, d[sp + '_Obs'].dropna().count()))
        for st in stats:
            col_name = sp + '_' + st
            if 'sqe' in col_name:
                mean_stat = math.sqrt(np.average(d[col_name].dropna()))
            else:
                mean_stat = np.average(d[col_name].dropna())
            outLine = '%s,'%(mean_stat)
            outfile.write(outLine)
        outfile.write('\n')
    outfile.close()

#%%

code_to_name = {300499000	: 'GAMO1',
                410358001	: 'CRLA1',
                160370002: 'SAWT1',
                320079000: 'JARB1',
                530410007: 'WHPA1',
                60893003: 'LAVO1',
                530470012: 'PASA1',
                300479000: 'FLAT1',
                410050010: 'MOHO1',
                60930005: 'LABE1',
                300899000: 'CABI1',
                300779000: 'MONT1',
                410610010: 'STAR1',
                530370004: 'SNPA1',
                61059000: 'TRIN1',
                300299001: 'GLAC1',
                410390070: 'THSI1',
                410630002: 'HECA1',
                530090020: 'OLYM1',
                530730022: 'NOCA1',
                530090013: 'MAKA2',
                530530014: 'MORA1',
                160230101: 'CRMO1',
                60150002: 'REDW1',
                530390010: 'COGO1',
                530390011: 'CORI1',
                410330010: 'KALM1',
                300819000: 'SULA1'}

#%%                
# lets make a bar plot of the mean modeled and observed concentration for each of the IMPROVE site              

df_improve_merged = pd.merge(df_fire_avg, df_nofire_avg, how='inner', suffixes=('_fire', '_nofire'), left_index=True, right_index=True)
df_pm25 = df_improve_merged.ix[:, ['PM25_Obs_fire', 'PM25_Obs_nofire', 'PM25_Mod_fire', 'PM25_Mod_nofire', 'PM25_bias_fire', 'PM25_bias_nofire', 'PM25_fb_fire', 'PM25_fb_nofire', 'PM25_fe_fire', 'PM25_fe_nofire']]
pm25_order = df_pm25.nlargest(df_pm25.index.size, df_pm25.columns, keep='first')
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16,12.5))
pm25_order.ix[:, ['PM25_Obs_fire', 'PM25_Mod_fire', 'PM25_Mod_nofire']].plot(kind='bar', color=['black', 'red', 'blue'], fontsize=15, grid=True, alpha=0.95, ax=ax1)
labels = [item.get_text() for item in ax1.get_xticklabels()]
new_labels = list(code_to_name[int(i)] for i in labels)
ax1.set_xticklabels(new_labels)
ax1.set_xlabel('', fontsize=15)
ax1.set_ylabel('Mean PM$_{2.5}$ Concentration (${\mu}g/m^{3}$)', fontsize=15)

bars = ax1.patches
hatches = ''.join(h*len(pm25_order) for h in '-/ ')

for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
ax1.legend(['Observed', 'Modeled - with fire', 'Modeled - no fire'], fontsize=15, ncol=3, fancybox=True, loc=1)
#ax1.set_title('Mean PM$_{2.5}$ Concentration at IMPROVE sites', fontsize=15)
#plt.savefig(plotDir + 'IMPROVE_PM25.png', pad_inches=0.1, bbox_inches='tight')

df_pmOC = df_improve_merged.ix[:, ['PM_OC_Obs_fire', 'PM_OC_Obs_nofire', 'PM_OC_Mod_fire', 'PM_OC_Mod_nofire', 'PM_OC_bias_fire', 'PM_OC_bias_nofire', 'PM_OC_fb_fire', 'PM_OC_fb_nofire', 'PM_OC_fe_fire', 'PM_OC_fe_nofire']]
pmOC_order = df_pmOC.nlargest(df_pmOC.index.size, df_pmOC.columns, keep='first')
#fig, ax = plt.figure(figsize=(12,9))
pmOC_order.ix[:, ['PM_OC_Obs_fire', 'PM_OC_Mod_fire', 'PM_OC_Mod_nofire']].plot(kind='bar',  color=['black', 'red', 'blue'], fontsize=15, grid=True, alpha=0.95, ax=ax2, legend=False)
labels = [item.get_text() for item in ax2.get_xticklabels()]
new_labels = list(code_to_name[int(i)] for i in labels)
ax2.set_xticklabels(new_labels, fontsize=15)
ax2.set_xlabel('Site ID', fontsize=15)
ax2.set_ylabel('Mean OC Concentration (${\mu}g/m^{3}$)', fontsize=15)
bars = ax2.patches
hatches = ''.join(h*len(pm25_order) for h in '-/ ')

for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

#ax2.legend(['Observed', 'Modeled - with fire', 'Modeled - no fire'], fontsize=15)
#ax2.set_title('Mean OC Concentration at IMPROVE sites', fontsize=15)
#plt.savefig(plotDir + 'IMPROVE_PM_OC.png', pad_inches=0.1, bbox_inches='tight')
plt.savefig(plotDir + 'IMPROVE_PM_25_OC.png', pad_inches=0.1, bbox_inches='tight')


#%%
df_pmEC = df_improve_merged.ix[:, ['PM_EC_Obs_fire', 'PM_EC_Obs_nofire', 'PM_EC_Mod_fire', 'PM_EC_Mod_nofire', 'PM_EC_bias_fire', 'PM_EC_bias_nofire', 'PM_EC_fb_fire', 'PM_EC_fb_nofire', 'PM_EC_fe_fire', 'PM_EC_fe_nofire']]
pmEC_order = df_pmEC.nlargest(df_pmEC.index.size, df_pmEC.columns, keep='first')
#fig, ax = plt.figure(figsize=(12,9))
ax = pmEC_order.ix[:, ['PM_EC_Obs_fire', 'PM_EC_Mod_fire', 'PM_EC_Mod_nofire']].plot(kind='bar', figsize=(10,4), color=['green', 'red', 'blue'], fontsize=15, grid=True, alpha=0.7)
labels = [item.get_text() for item in ax.get_xticklabels()]
new_labels = list(code_to_name[int(i)] for i in labels)
ax.set_xticklabels(new_labels)
ax.set_xlabel('Site ID', fontsize=15)
ax.set_ylabel('Mean Concentration', fontsize=15)
ax.legend(['Observed', 'Modeled - with fire', 'Modeled - no fire'], fontsize=15)
ax.set_title('Mean EC Concentration at IMPROVE sites', fontsize=15)
plt.savefig(plotDir + 'IMPROVE_PM_EC.png', pad_inches=0.1, bbox_inches='tight')

df_pmSO4 = df_improve_merged.ix[:, ['PM_SO4_Obs_fire', 'PM_SO4_Obs_nofire', 'PM_SO4_Mod_fire', 'PM_SO4_Mod_nofire', 'PM_SO4_bias_fire', 'PM_SO4_bias_nofire', 'PM_SO4_fb_fire', 'PM_SO4_fb_nofire', 'PM_SO4_fe_fire', 'PM_SO4_fe_nofire']]
pmSO4_order = df_pmSO4.nlargest(df_pmSO4.index.size, df_pmSO4.columns, keep='first')
#fig, ax = plt.figure(figsize=(12,9))
ax = pmSO4_order.ix[:, ['PM_SO4_Obs_fire', 'PM_SO4_Mod_fire', 'PM_SO4_Mod_nofire']].plot(kind='bar', figsize=(10,4), color=['green', 'red', 'blue'], fontsize=15, grid=True, alpha=0.7)
labels = [item.get_text() for item in ax.get_xticklabels()]
new_labels = list(code_to_name[int(i)] for i in labels)
ax.set_xticklabels(new_labels)
ax.set_xlabel('Site ID', fontsize=15)
ax.set_ylabel('Mean Concentration', fontsize=15)
ax.legend(['Observed', 'Modeled - with fire', 'Modeled - no fire'], fontsize=15)
ax.set_title('Mean SO4 Concentration at IMPROVE sites', fontsize=15)
plt.savefig(plotDir + 'IMPROVE_PM_SO4.png', pad_inches=0.1, bbox_inches='tight')

#%%
df_improve_merged_all = pd.merge(df_fire, df_nofire, how='inner', on= ['SiteId', 'Time On'], suffixes=('_fire', '_nofire'), left_index=True, right_index=True)
df_pm25_all = df_improve_merged_all.ix[:, ['SiteId', 'Time On', 'PM25_Obs_fire', 'PM25_Obs_nofire', 'PM25_Mod_fire', 'PM25_Mod_nofire', 'PM25_bias_fire', 'PM25_bias_nofire', 'PM25_fb_fire', 'PM25_fb_nofire', 'PM25_fe_fire', 'PM25_fe_nofire']]

r2_fire   = df_pm25_all.corr().ix['PM25_Obs_fire', 'PM25_Mod_fire']**2 # first calculate correlation matrix and then get relevant data
r2_nofire = df_pm25_all.corr().ix['PM25_Obs_nofire', 'PM25_Mod_nofire']**2
label_fire   = '%s%.2f%s'%('with fire (r${^2}$=', r2_fire,  ')')
label_nofire = '%s%.2f%s'%('no fire   (r${^2}$=', r2_nofire,')')
ax = df_pm25_all.plot(x='PM25_Obs_fire', y='PM25_Mod_fire', kind='scatter', label=label_fire, marker='d', s=50.0, facecolor='None', edgecolors = 'Red', alpha = 1, grid=True, color='DarkBlue', ylim=[-2,20], xlim=[-2,20], figsize=(7,6), fontsize=15)
df_pm25_all.plot(x='PM25_Obs_fire', y='PM25_Mod_nofire',kind='scatter', label=label_nofire, marker='o', s=50.0, facecolor='None', color='DarkGreen', edgecolors = 'Blue', alpha = 1, grid=True,linewidth=1.25, ylim=[-2,20], xlim=[-2,20],ax=ax,figsize=(7,6), fontsize=15)
ax.legend(loc=2, fontsize=12, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
ax.plot((-2,20), (-2,20), 'k--')
ax.set_xlabel('Observed (ug/m3)', fontsize=15)
ax.set_ylabel('Modeled (ug/m3)', fontsize=15)
ax.set_title('Scatter plot: IMPROVE sites', fontsize=15, weight='bold')
plt.savefig(plotDir+'IMPROVE_PM25_scatter.png',  pad_inches=0.1, bbox_inches='tight')   # using pad_inches leaves some whitespace, else use of bbox_inches cuts off some part of figure also

r2_fire   = df_improve_merged_all.corr().ix['PM_SO4_Obs_fire', 'PM_SO4_Mod_fire']**2 # first calculate correlation matrix and then get relevant data
r2_nofire = df_improve_merged_all.corr().ix['PM_SO4_Obs_nofire', 'PM_SO4_Mod_nofire']**2
label_fire   = '%s%.2f%s'%('with fire (r${^2}$=', r2_fire,  ')')
label_nofire = '%s%.2f%s'%('no fire   (r${^2}$=', r2_nofire,')')
ax = df_improve_merged_all.plot(x='PM_SO4_Obs_fire', y='PM_SO4_Mod_fire', kind='scatter', label=label_fire, marker='o', s=50.0, facecolor='None', edgecolors = 'Red', alpha = 1, grid=True, color='DarkBlue', ylim=[-2,2], xlim=[-2,2], figsize=(7,6), fontsize=15)
df_improve_merged_all.plot(x='PM_SO4_Obs_fire', y='PM_SO4_Mod_nofire',kind='scatter', label=label_nofire, marker='o', s=50.0, facecolor='None', color='DarkGreen', edgecolors = 'Blue', alpha = 1, grid=True,linewidth=1.25, ylim=[-2,2], xlim=[-2,2],ax=ax,figsize=(7,6), fontsize=15)
ax.legend(loc=2, fontsize=12, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
ax.plot((-2,2), (-2,2), 'k--')
ax.set_xlabel('Observed (ug/m3)', fontsize=15)
ax.set_ylabel('Modeled (ug/m3)', fontsize=15)
ax.set_title('Scatter plot: IMPROVE sites', fontsize=15, weight='bold')
#%%
df_improve_merged_all = pd.merge(df_fire, df_nofire, how='inner', on= ['SiteId', 'Time On'], suffixes=('_fire', '_nofire'), left_index=True, right_index=True)
df_pm25_all = df_improve_merged_all.ix[:, ['SiteId', 'Time On', 'PM25_Obs_fire', 'PM25_Obs_nofire', 'PM25_Mod_fire', 'PM25_Mod_nofire', 'PM25_bias_fire', 'PM25_bias_nofire', 'PM25_fb_fire', 'PM25_fb_nofire', 'PM25_fe_fire', 'PM25_fe_nofire']]

# modeled to observed ratio
df_pm25_all['fire_mod/fire_obs'] = df_pm25_all['PM25_Mod_fire']/df_pm25_all['PM25_Obs_fire']
df_pm25_all['nofire_mod/nofire_obs'] = df_pm25_all['PM25_Mod_nofire']/df_pm25_all['PM25_Obs_nofire']
r2_fire   = df_pm25_all.corr().ix['PM25_Obs_fire', 'PM25_Mod_fire']**2 # first calculate correlation matrix and then get relevant data
r2_nofire = df_pm25_all.corr().ix['PM25_Obs_nofire', 'PM25_Mod_nofire']**2
label_fire   = '%s%.2f%s'%('with fire (r${^2}$=', r2_fire,  ')')
label_nofire = '%s%.2f%s'%('no fire   (r${^2}$=', r2_nofire,')')
ax = df_pm25_all.plot(x='PM25_Obs_fire', y='fire_mod/fire_obs', kind='scatter', label=label_fire, marker='D', s=50.0, facecolor='None', edgecolors = 'Red', alpha = 1, grid=True, color='DarkBlue', xlim=[-2,35], figsize=(7,6), fontsize=20, logy=True)
df_pm25_all.plot(x='PM25_Obs_fire', y='nofire_mod/nofire_obs',kind='scatter', label=label_nofire, marker='o', s=50.0, facecolor='None', color='DarkGreen', edgecolors = 'Blue', alpha = 1, grid=True,linewidth=1.25, xlim=[-2,35],ax=ax,figsize=(7,6), fontsize=20, logy=True)
ax.legend(loc=1, fontsize=15, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
ax.plot((-2,35), (1,1), '-', color = (0.5,0.5,0.5))
ax.set_xlabel('Observed (${\mu}g/m^{3}$)', fontsize=20)
ax.set_ylabel('Modeled / Observed', fontsize=20)
ax.set_title('IMPROVE sites', fontsize=20)
plt.savefig(plotDir+'IMPROVE_PM25_obs-mod_Ratio.png',  pad_inches=0.1, bbox_inches='tight')   # using pad_inches leaves some whitespace, else use of bbox_inches cuts off some part of figure also

r2_fire   = df_improve_merged_all.corr().ix['PM_SO4_Obs_fire', 'PM_SO4_Mod_fire']**2 # first calculate correlation matrix and then get relevant data
r2_nofire = df_improve_merged_all.corr().ix['PM_SO4_Obs_nofire', 'PM_SO4_Mod_nofire']**2
label_fire   = '%s%.2f%s'%('with fire (r${^2}$=', r2_fire,  ')')
label_nofire = '%s%.2f%s'%('no fire   (r${^2}$=', r2_nofire,')')
ax = df_improve_merged_all.plot(x='PM_SO4_Obs_fire', y='PM_SO4_Mod_fire', kind='scatter', label=label_fire, marker='o', s=50.0, facecolor='None', edgecolors = 'Red', alpha = 1, grid=True, color='DarkBlue', ylim=[-2,2], xlim=[-2,2], figsize=(7,6), fontsize=15)
df_improve_merged_all.plot(x='PM_SO4_Obs_fire', y='PM_SO4_Mod_nofire',kind='scatter', label=label_nofire, marker='o', s=50.0, facecolor='None', color='DarkGreen', edgecolors = 'Blue', alpha = 1, grid=True,linewidth=1.25, ylim=[-2,2], xlim=[-2,2],ax=ax,figsize=(7,6), fontsize=15)
ax.legend(loc=2, fontsize=12, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
ax.plot((-2,2), (-2,2), 'k--')
ax.set_xlabel('Observed (ug/m3)', fontsize=15)
ax.set_ylabel('Modeled (ug/m3)', fontsize=15)
ax.set_title('Scatter plot: IMPROVE sites', fontsize=15, weight='bold')

df_pm25_all_IMPROVE = df_pm25_all.copy()
#%%
# at individual sites
df_improve_merged_all = pd.merge(df_fire, df_nofire, how='inner', on= ['SiteId', 'Time On'], suffixes=('_fire', '_nofire'), left_index=True, right_index=True)
df_pm25_all = df_improve_merged_all.ix[:, ['SiteId', 'Time On', 'PM25_Obs_fire', 'PM25_Obs_nofire', 'PM25_Mod_fire', 'PM25_Mod_nofire', 'PM25_bias_fire', 'PM25_bias_nofire', 'PM25_fb_fire', 'PM25_fb_nofire', 'PM25_fe_fire', 'PM25_fe_nofire']]
df_pm25_all = df_pm25_all.set_index('SiteId')
pm25_site = df_pm25_all.loc[df_pm25_all.index == 60930005]
pm25_site['datetime'] = pd.to_datetime(pm25_site['Time On'])
pm25_site = pm25_site.set_index('datetime')
pm25_site.ix[:, ['PM25_Obs_fire', 'PM25_Mod_fire', 'PM25_Mod_nofire']].plot(kind='line', grid=True)

# OC now
df_pmOC_all = df_improve_merged_all.ix[:, ['SiteId', 'Time On', 'PM_OC_Obs_fire', 'PM_OC_Obs_nofire', 'PM_OC_Mod_fire', 'PM_OC_Mod_nofire', 'PM_OC_bias_fire', 'PM_OC_bias_nofire', 'PM_OC_fb_fire', 'PM_OC_fb_nofire', 'PM_OC_fe_fire', 'PM_OC_fe_nofire']]
df_pmOC_all = df_pmOC_all.set_index('SiteId')
pmOC_site = df_pmOC_all.loc[df_pmOC_all.index == 60930005] #530390010
pmOC_site['datetime'] = pd.to_datetime(pmOC_site['Time On'])
pmOC_site = pmOC_site.set_index('datetime')
pmOC_site.ix[:, ['PM_OC_Obs_fire', 'PM_OC_Mod_fire', 'PM_OC_Mod_nofire']].plot(kind='line')

# EC now
df_pmEC_all = df_improve_merged_all.ix[:, ['SiteId', 'Time On', 'PM_EC_Obs_fire', 'PM_EC_Obs_nofire', 'PM_EC_Mod_fire', 'PM_EC_Mod_nofire', 'PM_EC_bias_fire', 'PM_EC_bias_nofire', 'PM_EC_fb_fire', 'PM_EC_fb_nofire', 'PM_EC_fe_fire', 'PM_EC_fe_nofire']]
df_pmEC_all = df_pmEC_all.set_index('SiteId')
pmEC_site = df_pmEC_all.loc[df_pmEC_all.index == 61059000] #530390010, 60893003
pmEC_site['datetime'] = pd.to_datetime(pmEC_site['Time On'])
pmEC_site = pmEC_site.set_index('datetime')
pmEC_site.ix[:, ['PM_EC_Obs_fire', 'PM_EC_Mod_fire', 'PM_EC_Mod_nofire']].plot(kind='line')

# SO4 now
df_pmSO4_all = df_improve_merged_all.ix[:, ['SiteId', 'Time On', 'PM_SO4_Obs_fire', 'PM_SO4_Obs_nofire', 'PM_SO4_Mod_fire', 'PM_SO4_Mod_nofire', 'PM_SO4_bias_fire', 'PM_SO4_bias_nofire', 'PM_SO4_fb_fire', 'PM_SO4_fb_nofire', 'PM_SO4_fe_fire', 'PM_SO4_fe_nofire']]
df_pmSO4_all = df_pmSO4_all.set_index('SiteId')
pmSO4_site = df_pmSO4_all.loc[df_pmSO4_all.index == 530390010] #530390010, 60893003
pmSO4_site['datetime'] = pd.to_datetime(pmSO4_site['Time On'])
pmSO4_site = pmSO4_site.set_index('datetime')
pmSO4_site.ix[:, ['PM_SO4_Obs_fire', 'PM_SO4_Mod_fire', 'PM_SO4_Mod_nofire']].plot(kind='line', grid=True)

#%%
#---------AQS SITES----------
# below for AQS sites for PM2.5 only
frm_file    = obsDir + 'hourly_88101_2011_PM25_airpact.csv'
nofrm_file  = obsDir + 'hourly_88502_2011_PM25_airpact.csv'
fire_file   = inputDir + '/aqsSitesPM25AIRPACT4/PM25_88101_88501_airpact_RxFire_2011_Fire_100.dat'
nofire_file = inputDir + '/aqsSitesPM25AIRPACT4/PM25_88101_88501_airpact_RxFire_2011_Fire_000.dat'

df_frm      = pd.read_csv(frm_file, sep=',')
df_nofrm    = pd.read_csv(nofrm_file, sep=',')
df_firePM   = pd.read_csv(fire_file, sep='|', names = ['date', 'time', 'siteID', 'pollutant', 'conc_mod'])
df_nofirePM = pd.read_csv(nofire_file, sep='|', names = ['date', 'time', 'siteID', 'pollutant', 'conc_mod'])

# convert to datetime as a new column
# note that GMT time is used here since model result is reported as GMT
df_frm['datetime_gmt']      = pd.to_datetime(df_frm['Date GMT'] + ' ' + df_frm['Time GMT'])
df_nofrm['datetime_gmt']    = pd.to_datetime(df_nofrm['Date GMT'] + ' ' + df_nofrm['Time GMT'])
df_frm['datetime_local']    = pd.to_datetime(df_frm['Date Local'] + ' ' + df_frm['Time Local'])
df_nofrm['datetime_local']  = pd.to_datetime(df_nofrm['Date Local'] + ' ' + df_nofrm['Time Local'])
df_firePM['datetime_gmt']   = pd.to_datetime(df_firePM['date'] + ' ' + df_firePM['time']) # date and time here are GMT
df_nofirePM['datetime_gmt'] = pd.to_datetime(df_nofirePM['date'] + ' ' + df_nofirePM['time']) # date and time here are GMT

# drop the columns not needed for analysis
df_frm.drop([u'County Code', u'Site Num', u'POC', u'Datum',
       u'Parameter Name', u'Date Local', u'Time Local', u'Date GMT', u'Time GMT', u'MDL',
       u'Uncertainty', u'Qualifier', u'Method Type', u'Method Name',
       u'State Name', u'County Name', u'Date of Last Change'], axis=1, inplace=True)

df_nofrm.drop([u'County Code', u'Site Num', u'POC', u'Datum',
       u'Parameter Name', u'Date Local', u'Time Local', u'Date GMT', u'Time GMT', u'MDL',
       u'Uncertainty', u'Qualifier', u'Method Type', u'Method Code', u'Method Name',
       u'State Name', u'County Name', u'Date of Last Change'], axis=1, inplace=True)
 
df_firePM.drop([u'date', u'time', u'pollutant'], axis=1, inplace=True)
df_nofirePM.drop([u'date', u'time', u'pollutant'], axis=1, inplace=True)

# do some other operations and merging
df_aqs        = pd.concat([df_frm, df_nofrm], join='outer', ignore_index=True)
df_aqs.rename(columns={'Sample Measurement':'conc_obs'}, inplace=True) # change the name of column
df_aqs = df_aqs[df_aqs['conc_obs']>0] # remove rows with negative concentrations

df_fire_aqs   = pd.merge(df_aqs, df_firePM, how='inner', on=['siteID', 'datetime_gmt'])
df_nofire_aqs = pd.merge(df_aqs, df_nofirePM, how='inner', on=['siteID', 'datetime_gmt'])

#%%
def pm_24h_avg(df):
    '''
    calculates 24 hr average values for PM2.5 based on the datetime_gmt
    must supply the input dataframe with datetime_gmt as the index
    '''
    df_in = df.copy()
#    df_in = df.set_index('datetime_gmt', inplace=False)
#    df_in.head(n=5)
    df_out = pd.DataFrame(columns = ['avg_24hr_obs', 'avg_24hr_mod', 'siteID'])
#    print list(set(df_in['siteID']))
    for site in list(set(df_in['siteID'])):
        df_site = df_in[df_in['siteID']==site]
        df_site_out  = pd.DataFrame(index=pd.date_range(start='2011-10-01', end='2011-11-30', freq='D'))
        for dt in df_site_out.index:
            hr = 8
            print dt
            sum_mod = 0; sum_obs = 0
            cnt_mod = 0; cnt_obs = 0
            for hr in np.arange(8, (24+8)):
                # if first hour use the dt as date
                next_time = dt + timedelta(hours = hr)
#                print next_time
                # now ccalculation   
                if (next_time in df_site.index and df_site.index.is_unique):
#                    print df_site.ix[next_time, 'conc_obs']
#                    print df_site.ix[next_time, :]
                    if (np.logical_not(np.isnan(df_site.ix[next_time, 'conc_obs']))):
                        sum_obs += df_site.ix[next_time, 'conc_obs']
                        cnt_obs += 1
#                        print next_time
#                        print sum_obs
                    if (np.logical_not(np.isnan(df_site.ix[next_time, 'conc_mod']))):
                        sum_mod += df_site.ix[next_time, 'conc_mod']
                        cnt_mod += 1
#                        print sum_mod
                else:
                    continue
#                hr = hr + 1
            if (cnt_mod==0 and cnt_obs==0):
                avg_obs=np.nan
                avg_mod=np.nan
            else:
                avg_obs = sum_obs / cnt_obs
                avg_mod = sum_mod / cnt_mod
            df_site_out.ix[dt, 'avg_24hr_obs'] = avg_obs
            df_site_out.ix[dt, 'avg_24hr_mod'] = avg_mod
            df_site_out.ix[dt, 'count_obs'] = cnt_obs
            df_site_out.ix[dt, 'count_mod'] = cnt_mod
            df_site_out.ix[dt, 'siteID'] = site
            print site, dt, avg_mod, avg_obs, cnt_mod, cnt_obs
        df_out = pd.concat([df_site_out, df_out], join='outer')
    return df_out
    
    if __name__ == '__main__':
        return df_out

#%%

df_fire24avg   = pm_24h_avg(df_fire_aqs.set_index('datetime_gmt'))
df_nofire24avg = pm_24h_avg(df_nofire_aqs.set_index('datetime_gmt'))
df_fire24avg['date']   = df_fire24avg.index
df_nofire24avg['date'] = df_nofire24avg.index
#%%
# histogram of distribution at all AQS sites
#df_24avg_aqs_merged = pd.merge(df_fire24avg, df_nofire24avg, how='inner', suffixes=('_fire', '_nofire'), left_index=True, right_index=True)
df_24avg_aqs_merged = pd.merge(df_fire24avg, df_nofire24avg, how='inner', on=['siteID', 'date'], suffixes=('_fire', '_nofire'))
df_pm25_24avg = df_24avg_aqs_merged.ix[:, ['date','avg_24hr_obs_fire', 'avg_24hr_obs_nofire', 'avg_24hr_mod_fire', 'avg_24hr_mod_nofire']]
ax = df_pm25_24avg.ix[:, ['avg_24hr_obs_fire', 'avg_24hr_mod_fire', 'avg_24hr_mod_nofire']].plot(kind='hist', logy=True, logx=False, bins=np.arange(0,70,2), histtype='stepfilled', linewidth = 2, figsize=(12,8), color=['green', 'red', 'blue'], fontsize=15, grid=True, alpha=0.5)
ax.set_xlabel('Concentration (ug/m3)', fontsize=15)
#ax.set_xlim([0,61])
ax.set_ylabel('Frequency', fontsize=15)
ax.legend(['Observed', 'Modeled - with fire', 'Modeled - no fire'], fontsize=15)
ax.set_title('Mean PM2.5 distribution at AQS sites', fontsize=15)
plt.savefig(plotDir + 'AQS_PM25_hist.png', pad_inches=0.1, bbox_inches='tight')

# scatter plot for all AQS sites together
r2_fire   = df_pm25_24avg.corr().ix['avg_24hr_obs_fire', 'avg_24hr_mod_fire']**2 # first calculate correlation matrix and then get relevant data
r2_nofire = df_pm25_24avg.corr().ix['avg_24hr_obs_nofire', 'avg_24hr_mod_nofire']**2
label_fire   = '%s%.2f%s'%('with fire (r${^2}$=', r2_fire,  ')')
label_nofire = '%s%.2f%s'%('no fire   (r${^2}$=', r2_nofire,')')
ax = df_pm25_24avg.plot(x='avg_24hr_obs_fire', y='avg_24hr_mod_fire', kind='scatter', label=label_fire, marker='D', s=50.0, facecolor='None', edgecolors = 'Red', alpha = 1, grid=True, color='DarkBlue', ylim=[-5,75], xlim=[-5,75], figsize=(7,6), fontsize=15)
df_pm25_24avg.plot(x='avg_24hr_obs_fire', y='avg_24hr_mod_nofire',kind='scatter', label=label_nofire, marker='o', s=50.0, facecolor='None', color='DarkGreen', edgecolors = 'Blue', alpha = 1, grid=True,linewidth=1.25, ylim=[-5,75], xlim=[-5,75],ax=ax,figsize=(7,6), fontsize=15)
ax.legend(loc=1, fontsize=12, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
ax.plot((-5,75), (-5,75), 'k--')
ax.set_xlabel('Observed (${\mu}g/m{^3}$)', fontsize=15)
ax.set_ylabel('Modeled (${\mu}g/m{^3}$)', fontsize=15)
ax.set_title('Scatter plot: AQS sites', fontsize=15, weight='bold')
plt.savefig(plotDir+'AQS_PM25_scatter.png',  pad_inches=0.1, bbox_inches='tight')   # using pad_inches leaves some whitespace, else use of bbox_inches cuts off some part of figure also

#%%
df_pm25_24avg['fire_mod/fire_obs'] = df_pm25_24avg['avg_24hr_mod_fire']/df_pm25_24avg['avg_24hr_obs_fire']
df_pm25_24avg['nofire_mod/nofire_obs'] = df_pm25_24avg['avg_24hr_mod_nofire']/df_pm25_24avg['avg_24hr_obs_nofire']
# scatter plot for all AQS sites together
r2_fire   = df_pm25_24avg.corr().ix['avg_24hr_obs_fire', 'avg_24hr_mod_fire']**2 # first calculate correlation matrix and then get relevant data
r2_nofire = df_pm25_24avg.corr().ix['avg_24hr_obs_nofire', 'avg_24hr_mod_nofire']**2
label_fire   = '%s%.2f%s'%('with fire (r${^2}$=', r2_fire,  ')')
label_nofire = '%s%.2f%s'%('no fire   (r${^2}$=', r2_nofire,')')
ax = df_pm25_24avg.plot(x='avg_24hr_obs_fire', y='fire_mod/fire_obs', kind='scatter', label=label_fire, marker='D', s=50.0, facecolor='None', edgecolors = 'Red', alpha = 1, grid=True, color='DarkBlue', xlim=[-2,55], figsize=(7,6), fontsize=20, logy=True)
df_pm25_24avg.plot(x='avg_24hr_obs_nofire', y='nofire_mod/nofire_obs',kind='scatter', label=label_nofire, marker='o', s=50.0, facecolor='None', color='DarkGreen', edgecolors = 'Blue', alpha = 1, grid=True,linewidth=1.25, xlim=[-2,55],ax=ax,figsize=(7,6), fontsize=20, logy=True)
ax.legend(loc=1, fontsize=15, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
ax.plot((-2,55), (1,1), '-', color=(0.5,0.5,0.5))
ax.set_xlabel('Observed (${\mu}g/m{^3}$)', fontsize=20)
ax.set_ylabel('Modeled / Observed', fontsize=20)
ax.set_title('AQS sites', fontsize=20)
plt.savefig(plotDir+'AQS_PM25_obs-mod_Ratio.png',  pad_inches=0.1, bbox_inches='tight')   # using pad_inches leaves some whitespace, else use of bbox_inches cuts off some part of figure also

df_pm25_all_AQS = df_pm25_24avg.copy()
#%%
# plot AQS and IMPROVE Mod/obs vs obs on the same plot below:
r_fire   = df_pm25_all_IMPROVE.corr().ix['PM25_Obs_fire', 'PM25_Mod_fire'] # first calculate correlation matrix and then get relevant data
r_nofire = df_pm25_all_IMPROVE.corr().ix['PM25_Obs_nofire', 'PM25_Mod_nofire']
label_fire   = '%s%.2f%s'%('with fire (r=', r_fire,  ')')
label_nofire = '%s%.2f%s'%('no fire    (r=', r_nofire,  ')')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,8))
df_pm25_all_IMPROVE.plot(x='PM25_Obs_fire', y='fire_mod/fire_obs', kind='scatter', label=label_fire, marker='D', s=50.0, facecolor='None', edgecolors = 'Red', alpha = 1, grid=True, color='DarkBlue', xlim=[-2,35],  fontsize=20, logy=True, ax=ax[0])
df_pm25_all_IMPROVE.plot(x='PM25_Obs_fire', y='nofire_mod/nofire_obs',kind='scatter', label=label_nofire, marker='o', s=50.0, facecolor='None', color='DarkGreen', edgecolors = 'Blue', alpha = 1, grid=True,linewidth=1.25, xlim=[-2,35],ax=ax[0],fontsize=20, logy=True)
ax[0].legend(loc=1, fontsize=20, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
ax[0].plot((-2,35), (1,1), '-', color = (0.5,0.5,0.5))
ax[0].set_xlabel('Observed (${\mu}g/m^{3}$)', fontsize=20)
ax[0].set_ylabel('Modeled / Observed', fontsize=20)
ax[0].set_title('IMPROVE sites', fontsize=20)

# scatter plot for all AQS sites together
r_fire   = df_pm25_all_AQS.corr().ix['avg_24hr_obs_fire', 'avg_24hr_mod_fire'] # first calculate correlation matrix and then get relevant data
r_nofire = df_pm25_all_AQS.corr().ix['avg_24hr_obs_nofire', 'avg_24hr_mod_nofire']
label_fire   = '%s%.2f%s'%('with fire (r=', r_fire,  ')')
label_nofire = '%s%.2f%s'%('no fire    (r=', r_nofire,  ')')

df_pm25_all_AQS.plot(x='avg_24hr_obs_fire', y='fire_mod/fire_obs', kind='scatter', label=label_fire, marker='D', s=50.0, facecolor='None', edgecolors = 'Red', alpha = 1, grid=True, color='DarkBlue', xlim=[-2,55], fontsize=20, logy=True, ax=ax[1])
df_pm25_all_AQS.plot(x='avg_24hr_obs_nofire', y='nofire_mod/nofire_obs',kind='scatter', label=label_nofire, marker='o', s=50.0, facecolor='None', color='DarkGreen', edgecolors = 'Blue', alpha = 1, grid=True,linewidth=1.25, xlim=[-2,55],ax=ax[1],fontsize=20, logy=True)
ax[1].legend(loc=1, fontsize=20, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
ax[1].plot((-2,55), (1,1), '-', color=(0.5,0.5,0.5))
ax[1].set_xlabel('Observed (${\mu}g/m{^3}$)', fontsize=20)
ax[1].set_ylabel('Modeled / Observed', fontsize=20)
ax[1].set_title('AQS sites', fontsize=20)
plt.savefig(plotDir+'IMPROVE_AQS_PM25_obs-mod_Ratio.png',  pad_inches=0.1, bbox_inches='tight')   # using pad_inches leaves some whitespace, else use of bbox_inches cuts off some part of figure also


#%% calculate some stats now
df_fire24avg['pm25_mean_conc']    = 0.5*(df_fire24avg['avg_24hr_mod'] + df_fire24avg['avg_24hr_obs'])
df_nofire24avg['pm25_mean_conc']  = 0.5*(df_nofire24avg['avg_24hr_mod'] + df_nofire24avg['avg_24hr_obs'])

df_fire24avg['pm25_bias']  = df_fire24avg['avg_24hr_mod']         - df_fire24avg['avg_24hr_obs']
df_fire24avg['pm25_error'] = abs(df_fire24avg['avg_24hr_mod']     - df_fire24avg['avg_24hr_obs'])
df_fire24avg['pm25_fb']    = 200*(df_fire24avg['avg_24hr_mod']    - df_fire24avg['avg_24hr_obs'])/(df_fire24avg['avg_24hr_mod'] + df_fire24avg['avg_24hr_obs'])
df_fire24avg['pm25_fe']    = 200*abs(df_fire24avg['avg_24hr_mod'] - df_fire24avg['avg_24hr_obs'])/(df_fire24avg['avg_24hr_mod'] + df_fire24avg['avg_24hr_obs'])
df_fire24avg['pm25_nb']    = 100*(df_fire24avg['avg_24hr_mod']    - df_fire24avg['avg_24hr_obs'])/(df_fire24avg['avg_24hr_obs'])
df_fire24avg['pm25_ne']    = 100*abs(df_fire24avg['avg_24hr_mod'] - df_fire24avg['avg_24hr_obs'])/(df_fire24avg['avg_24hr_obs'])
df_fire24avg['pm25_sqe']   = (df_fire24avg['avg_24hr_mod'] - df_fire24avg['avg_24hr_obs'])**2
    
df_nofire24avg['pm25_bias']  = df_nofire24avg['avg_24hr_mod']         - df_nofire24avg['avg_24hr_obs']
df_nofire24avg['pm25_error'] = abs(df_nofire24avg['avg_24hr_mod']     - df_nofire24avg['avg_24hr_obs'])
df_nofire24avg['pm25_fb']    = 200*(df_nofire24avg['avg_24hr_mod']    - df_nofire24avg['avg_24hr_obs'])/(df_nofire24avg['avg_24hr_mod'] + df_nofire24avg['avg_24hr_obs'])
df_nofire24avg['pm25_fe']    = 200*abs(df_nofire24avg['avg_24hr_mod'] - df_nofire24avg['avg_24hr_obs'])/(df_nofire24avg['avg_24hr_mod'] + df_nofire24avg['avg_24hr_obs'])
df_nofire24avg['pm25_nb']    = 100*(df_nofire24avg['avg_24hr_mod']    - df_nofire24avg['avg_24hr_obs'])/(df_nofire24avg['avg_24hr_obs'])
df_nofire24avg['pm25_ne']    = 100*abs(df_nofire24avg['avg_24hr_mod'] - df_nofire24avg['avg_24hr_obs'])/(df_nofire24avg['avg_24hr_obs'])
df_nofire24avg['pm25_sqe']   = (df_nofire24avg['avg_24hr_mod'] - df_nofire24avg['avg_24hr_obs'])**2
    
# calculate the mean of different stats for each site
df_nofire24avg_bysite = df_nofire24avg.groupby(by='siteID').mean()  #.add_prefix('mean_')
df_fire24avg_bysite   = df_fire24avg.groupby(by='siteID').mean()  #.add_prefix('mean_')

# write the stats to a file
stats   = [u'avg_24hr_obs', u'avg_24hr_mod', u'pm25_bias', u'pm25_error', u'pm25_fb', u'pm25_fe', u'pm25_nb', u'pm25_ne', u'pm25_sqe']

# open file
dfs = [df_nofire24avg, df_fire24avg]
name= {0:'aqs_nofire_24havg', 1:'aqs_fire_24havg'}
for i,d in enumerate(dfs):
    d = d.copy()
    stats_file = plotDir + name[i] + '_'+'mean_stats_file.csv'
    outfile = open(stats_file, 'w')
    outLine = '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n'%('species', 'Number observed', 'mean_obs', 'mean_mod', 'mean_bias', 'mean_error', 'mean_fracBias', 'mean_fracError', 'mean_normBias', 'mean_normError', 'rmse')
    outfile.write(outLine)
    
    outfile.write('%s,%s,'% ('PM25', d['avg_24hr_obs'].dropna().count()))
    for st in stats:
        col_name = st
        if 'sqe' in col_name:
            mean_stat = math.sqrt(np.average(d[col_name].dropna()))
        else:
            mean_stat = np.average(d[col_name].dropna())
        outLine = '%s,'%( mean_stat)
        outfile.write(outLine)
    outfile.write('\n')
    outfile.close()        
###################
## below not used
#%% calculate some stats now

df_fire_aqs['pm25_mean_conc']    = 0.5*(df_fire_aqs['conc_mod'] + df_fire_aqs['conc_obs'])
df_nofire_aqs['pm25_mean_conc']  = 0.5*(df_nofire_aqs['conc_mod'] + df_nofire_aqs['conc_obs'])

df_fire_aqs['pm25_bias']  = df_fire_aqs['conc_mod']         - df_fire_aqs['conc_obs']
df_fire_aqs['pm25_error'] = abs(df_fire_aqs['conc_mod']     - df_fire_aqs['conc_obs'])
df_fire_aqs['pm25_fb']    = 200*(df_fire_aqs['conc_mod']    - df_fire_aqs['conc_obs'])/(df_fire_aqs['conc_mod'] + df_fire_aqs['conc_obs'])
df_fire_aqs['pm25_fe']    = 200*abs(df_fire_aqs['conc_mod'] - df_fire_aqs['conc_obs'])/(df_fire_aqs['conc_mod'] + df_fire_aqs['conc_obs'])
df_fire_aqs['pm25_nb']    = 100*(df_fire_aqs['conc_mod']    - df_fire_aqs['conc_obs'])/(df_fire_aqs['conc_obs'])
df_fire_aqs['pm25_ne']    = 100*abs(df_fire_aqs['conc_mod'] - df_fire_aqs['conc_obs'])/(df_fire_aqs['conc_obs'])
df_fire_aqs['pm25_sqe']   = (df_fire_aqs['conc_mod'] - df_fire_aqs['conc_obs'])**2
    
df_nofire_aqs['pm25_bias']  = df_nofire_aqs['conc_mod']         - df_nofire_aqs['conc_obs']
df_nofire_aqs['pm25_error'] = abs(df_nofire_aqs['conc_mod']     - df_nofire_aqs['conc_obs'])
df_nofire_aqs['pm25_fb']    = 200*(df_nofire_aqs['conc_mod']    - df_nofire_aqs['conc_obs'])/(df_nofire_aqs['conc_mod'] + df_nofire_aqs['conc_obs'])
df_nofire_aqs['pm25_fe']    = 200*abs(df_nofire_aqs['conc_mod'] - df_nofire_aqs['conc_obs'])/(df_nofire_aqs['conc_mod'] + df_nofire_aqs['conc_obs'])
df_nofire_aqs['pm25_nb']    = 100*(df_nofire_aqs['conc_mod']    - df_nofire_aqs['conc_obs'])/(df_nofire_aqs['conc_obs'])
df_nofire_aqs['pm25_ne']    = 100*abs(df_nofire_aqs['conc_mod'] - df_nofire_aqs['conc_obs'])/(df_nofire_aqs['conc_obs'])
df_nofire_aqs['pm25_sqe']   = (df_nofire_aqs['conc_mod'] - df_nofire_aqs['conc_obs'])**2
    
# calculate the mean of different stats for each site
df_nofire_aqs_avg = df_nofire_aqs.groupby(by='siteID').mean()  #.add_prefix('mean_')
df_fire_aqs_avg   = df_fire_aqs.groupby(by='siteID').mean()  #.add_prefix('mean_')

# write the stats to a file
stats   = [u'conc_obs', u'conc_mod', u'pm25_bias', u'pm25_error', u'pm25_fb', u'pm25_fe', u'pm25_nb', u'pm25_ne', u'pm25_sqe']

# open file
dfs = [df_nofire_aqs, df_fire_aqs]
name= {0:'aqs_nofire', 1:'aqs_fire'}
for i,d in enumerate(dfs):
    d = d.copy()
    stats_file = plotDir + name[i] + '_'+'mean_stats_file.csv'
    outfile = open(stats_file, 'w')
    outLine = '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n'%('species', 'Number observed', 'mean_obs', 'mean_mod', 'mean_bias', 'mean_error', 'mean_fracBias', 'mean_fracError', 'mean_normBias', 'mean_normError', 'rmse')
    outfile.write(outLine)
    
    outfile.write('%s,%s,'% ('PM25', d['conc_obs'].dropna().count()))
    for st in stats:
        col_name = st
        if 'sqe' in col_name:
            mean_stat = math.sqrt(np.average(d[col_name].dropna()))
        else:
            mean_stat = np.average(d[col_name].dropna())
        outLine = '%s,'%( mean_stat)
        outfile.write(outLine)
    outfile.write('\n')
    outfile.close()

#%%
# lets make a bar plot of the mean modeled and observed concentration for each of the IMPROVE site              
df_aqs_merged = pd.merge(df_fire_aqs_avg, df_nofire_aqs_avg, how='inner', suffixes=('_fire', '_nofire'), left_index=True, right_index=True)
df_pm25 = df_aqs_merged.ix[:, ['conc_obs_fire', 'conc_obs_nofire', 'conc_mod_fire', 'conc_mod_nofire', 'pm25_bias_fire', 'pm25_bias_nofire', 'pm25_fb_fire', 'pm25_fb_nofire', 'pm25_fe_fire', 'pm25_fe_nofire']]
pm25_order = df_pm25.nlargest(df_pm25.index.size, df_pm25.columns, keep='first')
#fig, ax = plt.figure(figsize=(12,9))
ax = pm25_order.ix[:, ['conc_obs_fire', 'conc_mod_fire', 'conc_mod_nofire']].plot(kind='bar', figsize=(30,8), color=['green', 'red', 'blue'], fontsize=15, grid=True, alpha=0.7)
ax.set_xlabel('Site ID', fontsize=15)
ax.set_ylabel('Mean Concentration', fontsize=15)
ax.legend(['Observed', 'Modeled - with fire', 'Modeled - no fire'], fontsize=15)
ax.set_title('Mean PM2.5 Concentration at AQS sites', fontsize=15)
plt.savefig(plotDir + 'AQS_PM25.png', pad_inches=0.1, bbox_inches='tight')

df_pmOC = df_improve_merged.ix[:, ['PM_OC_Obs_fire', 'PM_OC_Obs_nofire', 'PM_OC_Mod_fire', 'PM_OC_Mod_nofire', 'PM_OC_bias_fire', 'PM_OC_bias_nofire', 'PM_OC_fb_fire', 'PM_OC_fb_nofire', 'PM_OC_fe_fire', 'PM_OC_fe_nofire']]
pmOC_order = df_pmOC.nlargest(df_pmOC.index.size, df_pmOC.columns, keep='first')
#fig, ax = plt.figure(figsize=(12,9))
ax = pmOC_order.ix[:, ['PM_OC_Obs_fire', 'PM_OC_Mod_fire', 'PM_OC_Mod_nofire']].plot(kind='bar', figsize=(10,4), color=['green', 'red', 'blue'], fontsize=15, grid=True, alpha=0.7)
ax.set_xlabel('Site ID', fontsize=15)
ax.set_ylabel('Mean Concentration', fontsize=15)
ax.legend(['Observed', 'Modeled - with fire', 'Modeled - no fire'], fontsize=15)
ax.set_title('Mean OC Concentration at IMPROVE sites', fontsize=15)
plt.savefig(plotDir + 'IMPROVE_PM_OC.png', pad_inches=0.1, bbox_inches='tight')

df_pmEC = df_improve_merged.ix[:, ['PM_EC_Obs_fire', 'PM_EC_Obs_nofire', 'PM_EC_Mod_fire', 'PM_EC_Mod_nofire', 'PM_EC_bias_fire', 'PM_EC_bias_nofire', 'PM_EC_fb_fire', 'PM_EC_fb_nofire', 'PM_EC_fe_fire', 'PM_EC_fe_nofire']]
pmEC_order = df_pmEC.nlargest(df_pmEC.index.size, df_pmEC.columns, keep='first')
#fig, ax = plt.figure(figsize=(12,9))
ax = pmEC_order.ix[:, ['PM_EC_Obs_fire', 'PM_EC_Mod_fire', 'PM_EC_Mod_nofire']].plot(kind='bar', figsize=(10,4), color=['green', 'red', 'blue'], fontsize=15, grid=True, alpha=0.7)
ax.set_xlabel('Site ID', fontsize=15)
ax.set_ylabel('Mean Concentration', fontsize=15)
ax.legend(['Observed', 'Modeled - with fire', 'Modeled - no fire'], fontsize=15)
ax.set_title('Mean EC Concentration at IMPROVE sites', fontsize=15)
plt.savefig(plotDir + 'IMPROVE_PM_EC.png', pad_inches=0.1, bbox_inches='tight')

#%%
# histogram of distribution at all AQS sites
df_aqs_merged_all = pd.merge(df_fire_aqs, df_nofire_aqs, how='inner', suffixes=('_fire', '_nofire'), left_index=True, right_index=True)
df_pm25_all = df_aqs_merged_all.ix[:, ['conc_obs_fire', 'conc_obs_nofire', 'conc_mod_fire', 'conc_mod_nofire', 'pm25_bias_fire', 'pm25_bias_nofire', 'pm25_fb_fire', 'pm25_fb_nofire', 'pm25_fe_fire', 'pm25_fe_nofire']]
ax = df_pm25_all.ix[:, ['conc_obs_fire', 'conc_mod_fire', 'conc_mod_nofire']].plot(kind='hist', logy=False, bins=np.arange(0,100,1), histtype='stepfilled', linewidth = 2, figsize=(12,8), color=['green', 'red', 'blue'], fontsize=15, grid=True, alpha=0.5)
ax.set_xlabel('Concentration (ug/m3)', fontsize=15)
ax.set_ylabel('Frequency', fontsize=15)
ax.legend(['Observed', 'Modeled - with fire', 'Modeled - no fire'], fontsize=15)
ax.set_title('Mean PM2.5 distribution at AQS sites', fontsize=15)
plt.savefig(plotDir + 'AQS_PM25_hist.png', pad_inches=0.1, bbox_inches='tight')

# scatter plot for all AQS sites together
r2_fire   = df_pm25_all.corr().ix['conc_obs_fire', 'conc_mod_fire']**2 # first calculate correlation matrix and then get relevant data
r2_nofire = df_pm25_all.corr().ix['conc_obs_nofire', 'conc_mod_nofire']**2
label_fire   = '%s%.2f%s'%('with fire (r${^2}$=', r2_fire,  ')')
label_nofire = '%s%.2f%s'%('no fire   (r${^2}$=', r2_nofire,')')
ax = df_pm25_all.plot(x='conc_obs_fire', y='conc_mod_fire', kind='scatter', label=label_fire, marker='o', s=20.0, facecolor='None', edgecolors = 'Red', alpha = 1, grid=True, color='DarkBlue', ylim=[-5,300], xlim=[-5,300], figsize=(7,6), fontsize=15)
df_pm25_all.plot(x='conc_obs_fire', y='conc_mod_nofire',kind='scatter', label=label_nofire, marker='o', s=20.0, facecolor='None', color='DarkGreen', edgecolors = 'Blue', alpha = 1, grid=True,linewidth=1.25, ylim=[-5,300], xlim=[-5,300],ax=ax,figsize=(7,6), fontsize=15)
ax.legend(loc=2, fontsize=12, frameon=True, framealpha=1, scatterpoints=1, scatteryoffsets=[0.5], fancybox=True, handletextpad=0.01)
ax.plot((-5,300), (-5,300), 'k--')
ax.set_xlabel('Observed (ug/m3)', fontsize=15)
ax.set_ylabel('Modeled (ug/m3)', fontsize=15)
ax.set_title('Scatter plot at AQS sites', fontsize=15, weight='bold')
plt.savefig(plotDir+'AQS_PM25_scatter.png',  pad_inches=0.1, bbox_inches='tight')   # using pad_inches leaves some whitespace, else use of bbox_inches cuts off some part of figure also


#%%
# at individual sites
df_improve_merged_all = pd.merge(df_fire, df_nofire, how='inner', on= ['SiteId', 'Time On'], suffixes=('_fire', '_nofire'), left_index=True, right_index=True)
df_pm25_all = df_improve_merged_all.ix[:, ['SiteId', 'Time On', 'PM25_Obs_fire', 'PM25_Obs_nofire', 'PM25_Mod_fire', 'PM25_Mod_nofire', 'PM25_bias_fire', 'PM25_bias_nofire', 'PM25_fb_fire', 'PM25_fb_nofire', 'PM25_fe_fire', 'PM25_fe_nofire']]
df_pm25_all = df_pm25_all.set_index('SiteId')
pm25_site = df_pm25_all.loc[df_pm25_all.index == 60893003]
pm25_site['datetime'] = pd.to_datetime(pm25_site['Time On'])
pm25_site = pm25_site.set_index('datetime')
pm25_site.ix[:, ['PM25_Obs_fire', 'PM25_Mod_fire', 'PM25_Mod_nofire']].plot(kind='line')

# OC now
df_pmOC_all = df_improve_merged_all.ix[:, ['SiteId', 'Time On', 'PM_OC_Obs_fire', 'PM_OC_Obs_nofire', 'PM_OC_Mod_fire', 'PM_OC_Mod_nofire', 'PM_OC_bias_fire', 'PM_OC_bias_nofire', 'PM_OC_fb_fire', 'PM_OC_fb_nofire', 'PM_OC_fe_fire', 'PM_OC_fe_nofire']]
df_pmOC_all = df_pmOC_all.set_index('SiteId')
pmOC_site = df_pmOC_all.loc[df_pmOC_all.index == 60893003] #530390010
pmOC_site['datetime'] = pd.to_datetime(pmOC_site['Time On'])
pmOC_site = pmOC_site.set_index('datetime')
pmOC_site.ix[:, ['PM_OC_Obs_fire', 'PM_OC_Mod_fire', 'PM_OC_Mod_nofire']].plot(kind='line')

# EC now
df_pmEC_all = df_improve_merged_all.ix[:, ['SiteId', 'Time On', 'PM_EC_Obs_fire', 'PM_EC_Obs_nofire', 'PM_EC_Mod_fire', 'PM_EC_Mod_nofire', 'PM_EC_bias_fire', 'PM_EC_bias_nofire', 'PM_EC_fb_fire', 'PM_EC_fb_nofire', 'PM_EC_fe_fire', 'PM_EC_fe_nofire']]
df_pmEC_all = df_pmEC_all.set_index('SiteId')
pmEC_site = df_pmEC_all.loc[df_pmEC_all.index == 60893003] #530390010, 60893003
pmEC_site['datetime'] = pd.to_datetime(pmEC_site['Time On'])
pmEC_site = pmEC_site.set_index('datetime')
pmEC_site.ix[:, ['PM_EC_Obs_fire', 'PM_EC_Mod_fire', 'PM_EC_Mod_nofire']].plot(kind='line')

#%%
#----------PLOTTING FOR IMPROVE--------------
# some plotting

# dectionary for names
spc_name = {'PM25': '$PM_{2.5}$', 'PM_OC':'OC', 'PM_EC':'EC', 'PM_SO4':'$SO_{4}^{2-}$', 'PM_NH4':'$NH_{4}^{+}$', 'PM_NO3':'$NO_{3}^{-}$'}
# plot mfb now
for species in ['PM25', 'PM_OC', 'PM_EC', 'PM_SO4', 'PM_NH4', 'PM_NO3']:
    x = np.arange(0, df_fire_avg[species+'_mean_conc'].max()+1, 0.1)
    
    # define the parameters for mfb goals and criteris functions
    # the coefficients
    y1 = +170*np.exp(-0.5*(2*x)/0.5) + 30  # MFB goal
    y2 = -170*np.exp(-0.5*(2*x)/0.5) - 30  # MFB goal
    y3 = +140*np.exp(-0.5*(2*x)/0.5) + 60  # MFB criteria
    y4 = -140*np.exp(-0.5*(2*x)/0.5) - 60  # MFB criteria
    
    fig, ax = plt.subplots(figsize=(12,10))
    
    # plot the mfb goal/criteria lines first
    line, = plt.plot(x, y1, '-', linewidth=1.75, color='k', label='MFB Goal', zorder=1)
    line, = plt.plot(x, y2, '-', linewidth=1.75, color='k', zorder=1)
    dashes = [10, 5, 10, 5]
    line, = plt.plot(x, y3, '--', linewidth=2.75, color='k', label='MFB Criteria', zorder=1)
    line, = plt.plot(x, y4, '--', linewidth=2.75, color='k', zorder=1) 
        
    # now plot the datapoints
    df_nofire_avg.plot(x=species+'_mean_conc', y=species+'_fb', kind='scatter', zorder=2, marker = 'o', s=50.0, color = 'b', edgecolor='b', label='No Fire', ax = fig.gca())
    df_fire_avg.plot(x=species+'_mean_conc', y=species+'_fb', kind='scatter', zorder=2, marker = 'D', s=50.0, color = 'r', edgecolor='r', label='With Fire', ax = fig.gca())
    ax.legend(fontsize=20, scatterpoints=1, frameon=True, framealpha=1, fancybox=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.ylim(-220,220)
    plt.xlim(xmin=-0.25, xmax=int(max(x))+0.5)
    #plt.xlim(xmin=-0.25, xmax=5)
    plt.grid(True)
    plt.ylabel('Fractional Bias, %', fontsize=20)
    plt.xlabel('Mean Concentration ($\mu g/m^{3}$)', fontsize=20)
    plt.title(' Fractional Bias for %s at all IMPROVE sites'%(spc_name[species]), fontsize=20)
    plt.savefig(plotDir + 'improve_fb_bugel_plot_'+species, pad_inches=0.1, bbox_inches='tight')
    
# plot mfe now
for species in ['PM25', 'PM_OC', 'PM_EC', 'PM_SO4', 'PM_NH4', 'PM_NO3']:
    x = np.arange(0, df_fire_avg[species+'_mean_conc'].max()+1, 0.1)
    
    # define the parameters for mfb goals and criteris functions
    # the coefficients
    y1 = +150*np.exp(-0.5*(2*x)/0.75) + 50  # MFE goal
#    y3 = +140*np.exp(-0.5*(2*x)/0.5) + 60  # MFE criteria
    y3 = +125*np.exp(-0.5*(2*x)/0.75) + 75  # MFE criteria
    
    fig, ax = plt.subplots(figsize=(12,10))
    
    # plot the mfb goal/criteria lines first
    line, = plt.plot(x, y1, '-', linewidth=1.75, color='k', zorder=1)
    dashes = [10, 5, 10, 5]
    line, = plt.plot(x, y3, '--', linewidth=2.75, color='k', zorder=1)
        
    # now plot the datapoints
    df_nofire_avg.plot(x=species+'_mean_conc', y=species+'_fe', kind='scatter', zorder=2, marker= 'o', s=50.0, color = 'b', edgecolor='b', ax = fig.gca())
    df_fire_avg.plot(x=species+'_mean_conc', y=species+'_fe', kind='scatter', zorder=2, marker='D', s=50.0, color = 'r', edgecolor='r', ax = fig.gca())
    ax.legend(fontsize=20, scatterpoints=1, frameon=True, framealpha=1, fancybox=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.ylim(-20,220)
    plt.xlim(xmin=-0.25, xmax=int(max(x))+0.5)
    #plt.xlim(xmin=-0.25, xmax=5)
    plt.grid(True)
    plt.ylabel('Fractional Error, %', fontsize=20)
    plt.xlabel('Mean Concentration ($\mu g/m^{3}$)', fontsize=20)
    plt.title(' Fractional Error for %s at all IMPROVE sites'%(spc_name[species]), fontsize=20)
    plt.savefig(plotDir + 'improve_fe_bugel_plot_new_eq_'+species, pad_inches=0.1, bbox_inches='tight')
    
#%%
#----------PLOTTING FOR AQS--------------

# dectionary for names
spc_name = {'pm25': '$PM_{2.5}$'}
# plot mfb now
for species in ['pm25']:
    x = np.arange(0, df_fire_aqs_avg[species+'_mean_conc'].max()+1, 0.1)
    
    # define the parameters for mfb goals and criteris functions
    # the coefficients
    y1 = +170*np.exp(-0.5*(2*x)/0.5) + 30  # MFB goal
    y2 = -170*np.exp(-0.5*(2*x)/0.5) - 30  # MFB goal
    y3 = +140*np.exp(-0.5*(2*x)/0.5) + 60  # MFB criteria
    y4 = -140*np.exp(-0.5*(2*x)/0.5) - 60  # MFB criteria
    
    fig, ax = plt.subplots(figsize=(12,10))
    
    # plot the mfb goal/criteria lines first
    line, = plt.plot(x, y1, '-', linewidth=1.75, color='k', zorder=1)
    line, = plt.plot(x, y2, '-', linewidth=1.75, color='k', zorder=1)
    dashes = [10, 5, 10, 5]
    line, = plt.plot(x, y3, '--', linewidth=2.75, color='k', zorder=1)
    line, = plt.plot(x, y4, '--', linewidth=2.75, color='k', zorder=1) 
        
    # now plot the datapoints
    df_nofire_aqs_avg.plot(x=species+'_mean_conc', y=species+'_fb', kind='scatter', zorder=2, marker = 'o', s=50.0, color = 'b', edgecolor='b', label='no fire', ax = fig.gca())
    df_fire_aqs_avg.plot(x=species+'_mean_conc', y=species+'_fb', kind='scatter', zorder=2, marker = 'D', s=50.0, color = 'r', edgecolor='r', label='with fire', ax = fig.gca())
    ax.legend(fontsize=20, scatterpoints=1, frameon=True, framealpha=1, fancybox=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.ylim(-220,220)
    plt.xlim(xmin=-0.25, xmax=int(max(x))+0.5)
    #plt.xlim(xmin=-0.25, xmax=5)
    plt.grid(True)
    plt.ylabel('Fractional Bias, %', fontsize=20)
    plt.xlabel('Mean Concentration ($\mu g/m^{3}$)', fontsize=20)
    plt.title(' Fractional Bias for %s at all AQS sites'%(spc_name[species]), fontsize=20)
    plt.savefig(plotDir + 'aqs_fb_bugel_plot_'+species, pad_inches=0.1, bbox_inches='tight')
    
# plot mfe now
for species in ['pm25']:
    x = np.arange(0, df_fire_aqs_avg[species+'_mean_conc'].max()+1, 0.1)
    
    # define the parameters for mfb goals and criteris functions
    # the coefficients
    y1 = +150*np.exp(-0.5*(2*x)/0.75) + 50  # MFE goal
    #y3 = +140*np.exp(-0.5*(2*x)/0.5) + 60  # MFE criteria
    y3 = +125*np.exp(-0.5*(2*x)/0.75) + 75  # MFE criteria
    
    fig, ax = plt.subplots(figsize=(12,10))
    
    # plot the mfb goal/criteria lines first
    line, = plt.plot(x, y1, '-', linewidth=1.75, color='k', zorder=1)
    dashes = [10, 5, 10, 5]
    line, = plt.plot(x, y3, '--', linewidth=2.75, color='k', zorder=1)
        
    # now plot the datapoints
    df_nofire_aqs_avg.plot(x=species+'_mean_conc', y=species+'_fe', kind='scatter', zorder=2, color = 'b', marker='o', s=50.0, edgecolor='b', label='no fire', ax = fig.gca())
    df_fire_aqs_avg.plot(x=species+'_mean_conc', y=species+'_fe', kind='scatter' , zorder=2, marker = 'D', s=50.0, color = 'r', edgecolor='r',label='with fire',  ax = fig.gca())
    ax.legend(fontsize=20, scatterpoints=1, frameon=True, framealpha=1, fancybox=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.ylim(-20,220)
    plt.xlim(xmin=-0.25, xmax=int(max(x))+0.5)
    #plt.xlim(xmin=-0.25, xmax=5)
    plt.grid(True)
    plt.ylabel('Fractional Error, %', fontsize=20)
    plt.xlabel('Mean Concentration ($\mu g/m^{3}$)', fontsize=20)
    plt.title(' Fractional Error for %s at all AQS sites'%(spc_name[species]), fontsize=20)
    plt.savefig(plotDir + 'aqs_fe_bugel_plot_'+species, pad_inches=0.1, bbox_inches='tight')
  

#%%
# plot for paper
spc_name = {'PM25': '$PM_{2.5}$'}
# plot mfb now
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,20))
for species in ['PM25']:
    x = np.arange(0, df_fire_avg[species+'_mean_conc'].max()+1, 0.1)
    
    # define the parameters for mfb goals and criteris functions
    # the coefficients
    y1 = +170*np.exp(-0.5*(2*x)/0.5) + 30  # MFB goal
    y2 = -170*np.exp(-0.5*(2*x)/0.5) - 30  # MFB goal
    y3 = +140*np.exp(-0.5*(2*x)/0.5) + 60  # MFB criteria
    y4 = -140*np.exp(-0.5*(2*x)/0.5) - 60  # MFB criteria
    
    # plot the mfb goal/criteria lines first
    line, = ax[0,0].plot(x, y1, '-', linewidth=1.75, color='k', zorder=1)
    line, = ax[0,0].plot(x, y2, '-', linewidth=1.75, color='k', zorder=1)
    dashes = [10, 5, 10, 5]
    line, = ax[0,0].plot(x, y3, '--', linewidth=2.75, color='k', zorder=1)
    line, = ax[0,0].plot(x, y4, '--', linewidth=2.75, color='k', zorder=1) 
        
    # now plot the datapoints
    df_nofire_avg.plot(x=species+'_mean_conc', y=species+'_fb', kind='scatter', zorder=2, marker = 'o', s=50.0, color = 'b', edgecolor='b',  ax = ax[0,0])
    df_fire_avg.plot(x=species+'_mean_conc', y=species+'_fb', kind='scatter', zorder=2, marker = 'D', s=50.0, color = 'r', edgecolor='r',  ax = ax[0,0])
    ax[0,0].legend(fontsize=15, scatterpoints=1, frameon=True, framealpha=1, fancybox=True)
    ax[0,0].tick_params(axis='both', which='major', labelsize=20)
    ax[0,0].set_ylim(-220,220)
    ax[0,0].set_xlim(xmin=-0.25, xmax=int(max(x)))
    #plt.xlim(xmin=-0.25, xmax=5)
    ax[0,0].grid(True)
    ax[0,0].set_ylabel('Fractional Bias, %', fontsize=20)
    ax[0,0].set_xlabel('Average Concentration ($\mu g/m^{3}$)', fontsize=20)
    ax[0,0].set_title(' IMPROVE sites - FB', fontsize=20)
#    plt.savefig(plotDir + 'improve_fb_bugel_plot_'+species, pad_inches=0.1, bbox_inches='tight')
    
# plot mfe now
for species in ['PM25']:
    x = np.arange(0, df_fire_avg[species+'_mean_conc'].max()+1, 0.1)
    
    # define the parameters for mfb goals and criteris functions
    # the coefficients
    y1 = +150*np.exp(-0.5*(2*x)/0.75) + 50  # MFE goal
#    y3 = +140*np.exp(-0.5*(2*x)/0.5) + 60  # MFE criteria
    y3 = +125*np.exp(-0.5*(2*x)/0.75) + 75  # MFE criteria
    
#    fig, ax = plt.subplots(figsize=(12,10))
    
    # plot the mfb goal/criteria lines first
    line, = ax[0,1].plot(x, y1, '-', linewidth=1.75, color='k', label='MFB Goal', zorder=1)
    dashes = [10, 5, 10, 5]
    line, = ax[0,1].plot(x, y3, '--', linewidth=2.75, color='k', label='MFB Criteria', zorder=1)
        
    # now plot the datapoints
    df_nofire_avg.plot(x=species+'_mean_conc', y=species+'_fe', kind='scatter', zorder=2, marker= 'o', s=50.0, color = 'b', edgecolor='b', label='No Fire', ax = ax[0,1])
    df_fire_avg.plot(x=species+'_mean_conc', y=species+'_fe', kind='scatter', zorder=2, marker='D', s=50.0, color = 'r', edgecolor='r', label='With Fire', ax = ax[0,1])
    ax[0,1].legend(fontsize=20, scatterpoints=1, frameon=True, framealpha=1, fancybox=True)
    ax[0,1].tick_params(axis='both', which='major', labelsize=20)
    ax[0,1].set_ylim(-20,220)
    ax[0,1].set_xlim(xmin=-0.25, xmax=int(max(x)))
    #plt.xlim(xmin=-0.25, xmax=5)
    ax[0,1].grid(True)
    ax[0,1].set_ylabel('Fractional Error, %', fontsize=20)
    ax[0,1].set_xlabel('Average Concentration ($\mu g/m^{3}$)', fontsize=20)
    ax[0,1].set_title(' IMPROVE sites - FE', fontsize=20)
#    plt.savefig(plotDir + 'improve_fe_bugel_plot_new_eq_'+species, pad_inches=0.1, bbox_inches='tight')
    
#----------PLOTTING FOR AQS--------------

# dectionary for names
spc_name = {'pm25': '$PM_{2.5}$'}
# plot mfb now
for species in ['pm25']:
    x = np.arange(0, df_fire_aqs_avg[species+'_mean_conc'].max()+1, 0.1)
    
    # define the parameters for mfb goals and criteris functions
    # the coefficients
    y1 = +170*np.exp(-0.5*(2*x)/0.5) + 30  # MFB goal
    y2 = -170*np.exp(-0.5*(2*x)/0.5) - 30  # MFB goal
    y3 = +140*np.exp(-0.5*(2*x)/0.5) + 60  # MFB criteria
    y4 = -140*np.exp(-0.5*(2*x)/0.5) - 60  # MFB criteria
    
#    fig, ax = plt.subplots(figsize=(12,10))
    
    # plot the mfb goal/criteria lines first
    line, = ax[1,0].plot(x, y1, '-', linewidth=1.75, color='k', zorder=1)
    line, = ax[1,0].plot(x, y2, '-', linewidth=1.75, color='k', zorder=1)
    dashes = [10, 5, 10, 5]
    line, = ax[1,0].plot(x, y3, '--', linewidth=2.75, color='k', zorder=1)
    line, = ax[1,0].plot(x, y4, '--', linewidth=2.75, color='k', zorder=1) 
        
    # now plot the datapoints
    df_nofire_aqs_avg.plot(x=species+'_mean_conc', y=species+'_fb', kind='scatter', zorder=2, marker = 'o', s=50.0, color = 'b', edgecolor='b' , ax = ax[1,0])
    df_fire_aqs_avg.plot(x=species+'_mean_conc', y=species+'_fb', kind='scatter', zorder=2, marker = 'D', s=50.0, color = 'r', edgecolor='r', ax = ax[1,0])
    ax[1,0].legend(fontsize=20, scatterpoints=1, frameon=True, framealpha=1, fancybox=True)
    ax[1,0].tick_params(axis='both', which='major', labelsize=20)
    ax[1,0].set_ylim(-220,220)
    ax[1,0].set_xlim(xmin=-0.25, xmax=int(max(x))+0.5)
    #plt.xlim(xmin=-0.25, xmax=5)
    ax[1,0].grid(True)
    ax[1,0].set_ylabel('Fractional Bias, %', fontsize=20)
    ax[1,0].set_xlabel('Average Concentration ($\mu g/m^{3}$)', fontsize=20)
    ax[1,0].set_title(' AQS sites - FB', fontsize=20)
#    plt.savefig(plotDir + 'aqs_fb_bugel_plot_'+species, pad_inches=0.1, bbox_inches='tight')
    
# plot mfe now
for species in ['pm25']:
    x = np.arange(0, df_fire_aqs_avg[species+'_mean_conc'].max()+1, 0.1)
    
    # define the parameters for mfb goals and criteris functions
    # the coefficients
    y1 = +150*np.exp(-0.5*(2*x)/0.75) + 50  # MFE goal
    #y3 = +140*np.exp(-0.5*(2*x)/0.5) + 60  # MFE criteria
    y3 = +125*np.exp(-0.5*(2*x)/0.75) + 75  # MFE criteria
    
#    fig, ax = plt.subplots(figsize=(12,10))
    
    # plot the mfb goal/criteria lines first
    line, = ax[1,1].plot(x, y1, '-', linewidth=1.75, color='k', zorder=1)
    dashes = [10, 5, 10, 5]
    line, = ax[1,1].plot(x, y3, '--', linewidth=2.75, color='k', zorder=1)
        
    # now plot the datapoints
    df_nofire_aqs_avg.plot(x=species+'_mean_conc', y=species+'_fe', kind='scatter', zorder=2, color = 'b', marker='o', s=50.0, edgecolor='b',  ax = ax[1,1])
    df_fire_aqs_avg.plot(x=species+'_mean_conc', y=species+'_fe', kind='scatter' , zorder=2, marker = 'D', s=50.0, color = 'r', edgecolor='r', ax = ax[1,1])
    ax[1,1].legend(fontsize=20, scatterpoints=1, frameon=True, framealpha=1, fancybox=True)
    ax[1,1].tick_params(axis='both', which='major', labelsize=20)
    ax[1,1].set_ylim(-20,220)
    ax[1,1].set_xlim(xmin=-0.25, xmax=int(max(x))+0.5)
    #plt.xlim(xmin=-0.25, xmax=5)
    ax[1,1].grid(True)
    ax[1,1].set_ylabel('Fractional Error, %', fontsize=20)
    ax[1,1].set_xlabel('Average Concentration ($\mu g/m^{3}$)', fontsize=20)
    ax[1,1].set_title(' AQS sites - FE', fontsize=20)
plt.savefig(plotDir + 'improve_aqs_fbfe_bugel_plot_'+species+'.png', pad_inches=0.1, bbox_inches='tight')
  



#%%

#----PLOTTING OF IMPROVE RESULTS ON MAP NOW-------
# now let's plot FB on map
# open the netCDF files
grd_file  = 'C:/Users/vik/Documents/MCIP_data/2011111100/MCIP/GRIDCRO2D'
grd = Dataset(grd_file,'r')
lat = grd.variables['LAT'][0,0,:,:]
lon = grd.variables['LON'][0,0,:,:]
ht  = grd.variables['HT'][0,0,:,:]
    
w = (grd.NCOLS)*(grd.XCELL)
h = (grd.NROWS)*(grd.YCELL)
lat_0 = grd.YCENT
lon_0 = grd.XCENT

# Set up matplotlib basemap.
map = Basemap(projection='lcc', width=w, height=h, lat_0=lat_0, lon_0=lon_0,
              llcrnrlon = lon[0,0], urcrnrlon = lon[258-1,285-1], 
              llcrnrlat = lat[0,0], urcrnrlat = lat[258-1,285-1], resolution='h', area_thresh=1000)# setting area_thresh doesn't plot lakes/coastlines smaller than threshold

x,y = map(lon,lat)

spc_name = {'PM25': '$PM_{2.5}$', 'PM_OC':'OC', 'PM_EC':'EC', 'PM_SO4':'$SO_{4}^{2-}$', 'PM_NH4':'$NH_{4}^{+}$', 'PM_NO3':'$NO_{3}^{-}$'}
for species in ['PM25', 'PM_OC', 'PM_EC', 'PM_SO4', 'PM_NH4', 'PM_NO3']:
    #Create a  map
    plt.figure(figsize=(14,10))
    ax=plt.gca()
    
    map.drawcoastlines()
    map.drawcountries()
    map.drawstates(linewidth=0.5)
    map.drawmapboundary(linewidth=3.0)
    
    # store lat longs and other things are a numpy array
    lat_loc   = []
    lon_loc   = []
    conc_site = []
    frac_site = []
    
    for i in df_fire_avg.index:
        lat_loc.append(df_fire_avg.at[i, 'Latitude']) 
        lon_loc.append(df_fire_avg.at[i, 'Longitude'])
        conc_site.append(df_fire_avg.at[i, species+'_Obs'])
        frac_site.append(df_fire_avg.at[i, species+'_fb'])
    
    lon  = np.asarray(lon_loc)
    lat  = np.asarray(lat_loc)
    conc = np.asarray(conc_site)
    frac = np.asarray(frac_site)
    
    scale_factor = lambda c: (50 if np.max(c) > 5 else 150)
    # now map
    x,y = map(lon,lat)
    map.scatter(x,y,s=scale_factor(conc)*conc,c=frac, alpha=1) # we will scale the dots by 15 time the concentrations
    c = plt.colorbar(orientation='vertical', shrink=0.8)
    c.set_label("FB (%)", fontsize=15)
    plt.clim(-200,200)
    plt.title('Fractional Bias at IMPROVE sites: with fires - %s'%(spc_name[species]), fontsize=15, weight='bold')
    plt.savefig(plotDir + 'improve_fb_all_sites_fire'+species+'.png', pad_inches=0.1, bbox_inches='tight')
    plt.show()
#%%
#----PLOTTING OF AQS RESULTS ON MAP NOW-------
# now let's plot FB on map
# open the netCDF files
grd_file  = 'C:/Users/vik/Documents/MCIP_data/2011111100/MCIP/GRIDCRO2D'
grd = Dataset(grd_file,'r')
lat = grd.variables['LAT'][0,0,:,:]
lon = grd.variables['LON'][0,0,:,:]
ht  = grd.variables['HT'][0,0,:,:]
    
#met_file = 'C:/Users/vik/Documents/MCIP_data/2011111100/MCIP/METCRO2D'
#met = Dataset(met_file,'r')
#
#prs = met.variables['PRSFC'][:,0,:,:]
#pbl = met.variables['PBL'][:,0,:,:]
#tmp = met.variables['TEMP2'][:,0,:,:]                
#wsp = met.variables['WSPD10'][:,0,:,:]

# Determine map parameters from GRID file.
#w = len(lon)*grd.XCELL
#h = len(lat)*grd.YCELL
w = (grd.NCOLS)*(grd.XCELL)
h = (grd.NROWS)*(grd.YCELL)
lat_0 = grd.YCENT
lon_0 = grd.XCENT

# Set up matplotlib basemap.
#m = Basemap(projection='lcc', width=w, height=h, lat_0=lat_0, lon_0=lon_0,
#            llcrnrlon  = -125., urcrnrlon  = -108., llcrnrlat  =   40., urcrnrlat  =   50., resolution='i')
#map = Basemap(projection='lcc', width=w, height=h, lat_0=lat_0, lon_0=lon_0,
#            llcrnrlon  = -126., urcrnrlon  = -108.25, llcrnrlat  =   39.5, urcrnrlat  =   49.85, resolution='i')
map = Basemap(projection='lcc', width=w, height=h, lat_0=lat_0, lon_0=lon_0,
              llcrnrlon = lon[0,0], urcrnrlon = lon[258-1,285-1], 
              llcrnrlat = lat[0,0], urcrnrlat = lat[258-1,285-1], resolution='h', area_thresh=1000)
# setting area_thresh doesn't plot lakes/coastlines smaller than threshold
x,y = map(lon,lat)

#Create a contour map of convective rain
plt.figure(figsize=(14,10))
ax=plt.gca()

#as of now, not plotting height
#c1=map.contourf(x,y,ht[:, :],cmap=plt.cm.terrain)
#
#cb=plt.colorbar(c1, shrink=0.8)
#cb.set_label('Height (msl)')

map.drawcoastlines()
map.drawcountries()
map.drawstates(linewidth=0.5)
map.drawmapboundary(linewidth=3.0)
#map.shadedrelief()
#map.etopo()
#map.bluemarble()

# store lat longs and other things are a numpy array
lat_loc   = []
lon_loc   = []
conc_site = []
frac_site = []

for i in df_nofire_aqs_avg.index:
    lat_loc.append(df_nofire_aqs_avg.at[i, 'Latitude']) 
    lon_loc.append(df_nofire_aqs_avg.at[i, 'Longitude'])
    conc_site.append(df_nofire_aqs_avg.at[i, 'conc_obs'])
    frac_site.append(df_nofire_aqs_avg.at[i, 'pm25_fb'])

lon  = np.asarray(lon_loc)
lat  = np.asarray(lat_loc)
conc = np.asarray(conc_site)
frac = np.asarray(frac_site)

# now map
x,y = map(lon,lat)
map.scatter(x,y,s=15*conc,c=frac, alpha=1, edgecolors='None') # we will scale the dots by 15 time the concentrations
c = plt.colorbar(orientation='vertical', shrink=0.8)
c.set_label("FB (%)", fontsize=15)
plt.clim(-200,200)
plt.title('Fractional Bias at AQS sites: no fires', fontsize=15, weight='bold')
#plt.savefig(plotDir + 'aqs_fb_all_sites_nofire.png', pad_inches=0.1, bbox_inches='tight')
plt.show()
