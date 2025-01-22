"""
TAMSAT-ALERT API Version 2 (soil moisture monitoring and forecasting)

The revised TAMSAT-ALERT API (V2) is based on Vicky Boult's original API
that only consided end of season forecasts of WRSI and has been updated
to provide the following:
    1. Uses the most recent version of TAMSAT soil moisture developed 
       within EOCIS.
    2. Can handle spatially variable start and end dates for the growing 
       season. These need to be defined in advance.
    3. Weight the TAMSAT-ALERT soil moisture forecasts using spatially
       variable ECMWF-S2S tercile precipitation forecasts.
    4. Additional metrics to provide WRSI from season start to current
       date.

The code should be executed as follows (example arguments provided):
python TAMSAT-ALERT_SM_API.V2.py -poi_start=2024-03-01 -poi_end=2024-07-31 -current_date=2024-05-15 -clim_years=1991,2020 -coords=6,-5,32,43 -weights=0.33,0.34,0.33

Authors: R. Maidment, V. Boult
"""

import sys
import wget
from datetime import datetime as dt
from datetime import timedelta as td
import datetime
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os.path
import scipy.stats
import pandas as pd
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib
import matplotlib.colors as mcolors
import argparse
import warnings
warnings.filterwarnings("ignore")


def makedir(path):
        if not os.path.exists(path):
            os.makedirs(path)


def parse_poi_start(value):
    """
    Custom parser to handle poi_start as either a date in the format YYYY-MM-DD
    or a file path ending in '.nc'.
    """
    try:
        # Check if the value is a valid date in the format YYYY-MM-DD
        #dt.strptime(value, "%Y-%m-%d")
        return dt.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        # If not a date, check if it's a valid file path ending with '.nc'
        if value.endswith('.nc') and os.path.isfile(value):
            return value
        raise argparse.ArgumentTypeError(
            "Invalid format. poi_start must be a date in the format YYYY-MM-DD (e.g., 2025-01-16) "
            "or a valid file path ending with '.nc'.")


def parse_current_date(value):
    """
    Custom parser to handle current_date as either a string in the format YYYY-MM-DD
    or the exact string 'LATEST'.
    """
    cutoff_date = dt(2024, 1, 1).date()  # Define the cutoff date
    if value == "LATEST":
        return value  # Return the valid string directly
    try:
        # Try to parse the date string in the format YYYY-MM-DD
        parsed_date = dt.strptime(value, "%Y-%m-%d").date()
        if parsed_date <= cutoff_date:
            raise argparse.ArgumentTypeError(
                f"Invalid date. Dates must be after {cutoff_date}."
            )
        return parsed_date
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. current_date must be 'LATEST' or a date in the format YYYY-MM-DD (e.g., 2025-01-16).")


def parse_clim_years(value):
    """
    Custom parser to handle clim_years as a comma-separated list of two integers
    representing years (e.g., 1991,2020).
    """
    try:
        # Parse as a list of numbers from a comma-separated string
        years = [int(year) for year in value.split(',')]
        if len(years) == 2 and all(1983 <= year <= 2100 for year in years):
            return years
        raise ValueError("Clim_years must be exactly two valid years between 1983 and 2100 (e.g., 1991,2020)")
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Clim_years must be two comma-separated years (e.g., 1991,2020)"
        )


def parse_coordinates(value):
    """
    Custom parser to handle coordinates as a comma-separated list of four numbers
    (e.g., 10,20,30,40).
    """
    try:
        # Parse as a list of numbers from a comma-separated string
        coordinates = [float(coord) for coord in value.split(',')]
        if len(coordinates) == 4:
            return coordinates
        raise ValueError("Coordinates must be exactly four numbers separated by commas (e.g., 10,20,30,40)")
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Coordinates must be four comma-separated numbers (e.g., 10,20,30,40)")


def parse_weights(value):
    """
    Custom parser to handle weights as either a comma-separated list of three numbers
    that sum to 1.0 (e.g., 0.3,0.4,0.3) or the exact string 'ECMWF_S2S'.
    """
    if value == "ECMWF_S2S":
        return value  # Return the valid string directly
    try:
        # Attempt to parse as a list of numbers from a comma-separated string
        weights = [float(w) for w in value.split(',')]
        if len(weights) == 3:
            if abs(sum(weights) - 1.0) < 1e-6:  # Allow for floating-point precision issues
                return weights
            else:
                raise ValueError("Weights must sum to 1.0")
        raise ValueError("Weights must be exactly three numbers separated by commas (e.g., 0.3,0.4,0.3)")
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Weights must be three comma-separated numbers that sum to 1.0 (e.g., 0.3,0.4,0.3) or 'ECMWF_S2S'")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-poi_start", type=parse_poi_start, help="Start date for period of interest, usually the start of the growing season - can be either a fixed date in format YYYY-MM-DD (e.g., 2025-01-16) or a file path of a gridded file (netCDF format with filename ending in .nc) giving spatially varying start dates.", required=True)
    parser.add_argument("-poi_end", type=parse_poi_start, help="End date for period of interest, usually the end of the growing season - can be either a fixed date (in format YYYY-MM-DD (e.g., 2025-01-16) or a file path of a gridded file (netCDF format with filname ending in .nc) giving spatially varying end dates.", required=True)
    parser.add_argument("-current_date", type=parse_current_date, help="Date in the season up to which soil moisture estimates are considered and soil moisture forecasts are considered from the day afterwards - must be in format YYYY-MM-DD (e.g. 2024-03-01 for 1st March 2024) or 'LATEST' if wanting to use the most recent soil moisture estimates", required=True)
    parser.add_argument('-clim_years', type=parse_clim_years, help="The start and end year of the climatological period over which anomalies are computed - must be two comma-separated years (e.g. 1991,2020).", required=True)
    parser.add_argument('-coords', type=parse_coordinates, help="Domain coordinates (N,S,W,E) must be a comma-separated list of four numbers (e.g. 10,20,30,40).", required=True)
    parser.add_argument('-weights', type=parse_weights, help="Weights can be a list of three numbers if using own tercile weights (e.g. -weights=0.3,0.4,0.3 which corresponds to [above, normal, below]) or 'ECMWF_S2S' if using the S2S tercile forecasts from the European Centre for Medium Range Weather Forecasts.", required=True)
    return(parser)


def check_current_date(current_date, sm_hist_dir, poi_end):
    # Reset current_date if current_date is after the last soil moisture day
    # Get last date of soil moisture
    sm_flist = []
    for root, dirs, files in os.walk(sm_hist_dir):
        for name in files:
            if name.endswith('.nc'):
                sm_flist.append(os.path.join(root, name))
    
    ds_sm = xr.open_dataset(sm_flist[-1])
    sm_last_date = pd.to_datetime(ds_sm.time.values[-1]).date()
    ds_sm.close()
    
    if current_date == 'LATEST':
        current_date = sm_last_date
        print('-> Latest date is %s' % dt.strftime(sm_last_date, '%Y-%m-%d'))
    
    if (poi_end.values.max() - current_date).days > 160:
        print('-> Warning! the season end is beyond the forecast window: cannot forecast beyond 160 days')
        sys.exit()
    
    if current_date > sm_last_date:
        current_date = sm_last_date
        print('-> Specified current date is after the last available day with soil moisture estimates, as such the specified current date (%s) has been reset to %s' % (dt.strftime(current_date, '%Y-%m-%d'), dt.strftime(sm_last_date, '%Y-%m-%d')))
    
    return current_date


def download_historical_data(remoteurl, datatype, version, yearstart, yearend):
    # List required files
    flist_required = []
    for yyyy in np.arange(yearstart, yearend + 1):
        if datatype == 'sm_hist':
            fname = f'sm{yyyy}_daily.v{version}.nc'
            localdir_name = 'soil_moisture_historical'
            comment = '-> Downloading required historical soil moisture files, this might take a few minutes ...'
        elif datatype == 'rfe_hist':
            fname = f'prate_tamsat_{yyyy}_sub.v{version}.nc'
            localdir_name = 'rainfall_historical'
            comment = '-> Downloading required historical rainfall files, this might take a few minutes ...'
        
        fname_full = os.path.join(inputdir, localdir_name, 'v' + version, fname)
        flist_required.append(fname_full)
    
    # Remove any corrupt files ('ending in .tmp') of duplicate files (with '(1)' in filename)
    flist_remove = []
    for root, dirs, files in os.walk(os.path.join(inputdir, localdir_name, 'v' + version)):
        for name in files:
            if name.endswith('tmp') or '(1)' in name:
                flist_remove.append(os.path.join(root, name))
    
    if len(flist_remove):
        for f in flist_remove:
            os.remove(f)    
    
    # List missing files
    flist_missing = []
    for f in flist_required:
        if not os.path.exists(f):
            flist_missing.append(f)
    
    # Add last two years so that any recently updated files are downloaded
    recent_files = [flist_required[-2], flist_required[-1]]
    flist_missing.extend(recent_files)
    flist_missing = list(set(flist_missing))
    flist_missing.sort()
    
    # Remove local version of the last two years to avoid duplicates
    for f in recent_files:
        if os.path.exists(f):
            os.remove(f)
    
    # Download missing files
    if len(flist_missing) > 0:
        print(comment)
        for f in flist_missing:
            fname = os.path.basename(f)
            url_file = os.path.join(remoteurl, datatype, 'v' + version, fname)            
            os.chdir(os.path.dirname(f))
            try:
                filename = wget.download(url_file, bar=None)
            except:
                print('-> Warning! Unable to download file: %s' % url_file)


def download_forecast_data(remoteurl, sm_fcast_dir, version, current_date):
    # Remove current version of the TAMSAT-ALERT ensemble forecast filenames list
    fcast_flist_local = os.path.join(sm_fcast_dir, 'tamsat-alert_fcast_filelist.csv')
    if os.path.exists(fcast_flist_local):
        os.remove(fcast_flist_local)
    
    # Download latest list of TAMSAT-ALERT ensemble forecast filenames
    fcast_flist_remote = os.path.join(remoteurl, 'sm_fcast', 'v' + version, 'tamsat-alert_fcast_filelist.csv')
    os.chdir(sm_fcast_dir)
    try:
        wget.download(fcast_flist_remote, bar=None)
    except:
        print('-> Warning! Unable to download file: %s' % fcast_flist_remote)
    
    # Determine file closest to the current date
    df = pd.read_csv(fcast_flist_local, header=None)
    fcast_flist = [item for sublist in df.values.tolist() for item in sublist]
    fcast_dates = [dt.strptime(x.split('_')[1][0:8], "%Y%m%d").date() for x in fcast_flist]
    fcast_file = min(fcast_dates, key=lambda x: abs(x - current_date))
    fcast_stamp = fcast_file.strftime("%Y%m%d")
    
    # Download required forecast file
    fcast_fname_local = os.path.join(sm_fcast_dir, 'alert_' + fcast_stamp + '_ens.daily.nc')
    if not os.path.exists(fcast_fname_local):
        fcast_fname_remote = os.path.join(remoteurl, 'sm_fcast', 'v' + sm_version, 'alert_' + fcast_stamp + '_ens.daily.nc')
        os.chdir(sm_fcast_dir)
        try:
            '-> Downloading required soil moisture forecast file, this might take a few minutes ...'
            wget.download(fcast_fname_remote, bar=None)
        except:
            print('-> Warning! Unable to download file: %s' % fcast_fname_remote)
    
        # Check file size
        file_size = os.path.getsize(fcast_fname_local)
        if file_size < 150 * 1024 * 1024:
            print('-> Warning! Forecast file size suggests it is corrupt or did not fully download: %s' % fcast_fname_local)
            os.remove(fcast_fname_local)
            try:
                wget.download(fcast_fname_remote, bar=None)
            except:
                print('-> Warning! Unable to download file: %s' % fcast_fname_remote)
    
    # Reset forecast date depending on latest available forecast data
    fcast_date = dt.strptime(fcast_stamp, "%Y%m%d").date()
    
    return fcast_date, fcast_fname_local


def get_year_range(clim_start_year, clim_end_year, poi_start, poi_end):
    yr = [clim_start_year, clim_end_year, 
        poi_start.values.min().year, poi_start.values.max().year,
        poi_end.values.min().year, poi_end.values.max().year]
    return min(yr), max(yr)


def dekad_lookup(year):
    # List of month lengths for a regular year
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Adjust February for leap years
    #if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    #    month_lengths[1] = 29
    
    # Generate dekad end dates
    dekad_end_dates = []
    dekad_number = 1
    for month, days_in_month in enumerate(month_lengths, start=1):
        # Dekad 1: Ends on the 10th
        dekad_end_dates.append((dekad_number, dt(year, month, 10).strftime("%d-%m-%Y")))
        dekad_number += 1
        
        # Dekad 2: Ends on the 20th
        dekad_end_dates.append((dekad_number, dt(year, month, 20).strftime("%d-%m-%Y")))
        dekad_number += 1
        
        # Dekad 3: Ends on the last day of the month
        dekad_end_dates.append((dekad_number, dt(year, month, days_in_month).strftime("%d-%m-%Y")))
        dekad_number += 1
    
    # Convert to a DataFrame for easy lookup
    dekad_df = pd.DataFrame(dekad_end_dates, columns=["Dekad", "End_Date"])
    return dekad_df


def replace_dekad_with_dates(data_array, lookup_table):
    # Convert the lookup table to a dictionary for mapping
    dekad_to_date_dict = lookup_table.set_index('Dekad')["End_Date_dt"].to_dict()
    
    # Vectorize the lookup function to handle arrays of dekad values
    vectorized_lookup = np.vectorize(lambda x: dekad_to_date_dict.get(x, "NaT"))
    
    # Apply the vectorized function to the DataArray
    replaced_data_array = xr.apply_ufunc(vectorized_lookup, data_array, dask="allowed")
    
    return replaced_data_array


def convert_dekad_to_datetime(fname):
    """
    Convert dekad number (1-72) spanning over two years into datetime object that xarray can read
    """
    # Read in SOS/EOS file
    ds_date = xr.open_dataset(fname)
    
    # Fill NaT (over the ocean) with max dekad - this is ok as there is no soil moisture over the ocean
    ds_date['dekad_capped_filled'] = ds_date['SOS_capped_filled'].fillna(ds_date['SOS_capped_filled'].max())
    
    # Next, create dekad LUT for given year/next year of dekad number and corresponding end of dekad date 
    # e.g. dekad '4' for 2024 equates to '2024-02-10'
    yyyy = pd.to_datetime(ds_date.time.values[0]).date().year
    dekad_LUT = pd.concat([dekad_lookup(yyyy), dekad_lookup(yyyy + 1)]).reset_index(drop=True)
    dekad_LUT['Dekad'] = np.arange(1, 73)
    dekad_LUT['End_Date_dt'] = [pd.to_datetime(x, format='%d-%m-%Y').date() for x in dekad_LUT.End_Date]
    
    # Finally, convert dekad number to actual datetime object
    ds_date['dekad_datetime'] = replace_dekad_with_dates(ds_date['dekad_capped_filled'], dekad_LUT)
    
    return (ds_date['dekad_datetime'])


def create_dataarray(poi_in, lon_min, lon_max, lat_min, lat_max):
    time = [poi_in.replace(month=1, day=1)]
    lon = np.arange(-17.875, 51.375 + 0.25, 0.25) 
    lat = np.arange(-35.375, 37.375 + 0.25, 0.25)
    data = np.full((len(time), len(lon), len(lat)), poi_in)
    da = xr.DataArray(data, coords={"time": time, "lon": lon, "lat": lat}, dims=["time", "lon", "lat"], name= "dekad_datetime")
    da = da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    da["time"] = pd.to_datetime(da["time"].values)
    return(da)


def check_poi(poi_in, lon_min, lon_max, lat_min, lat_max):
    """Preprocess poi_start and poi_end

    If supplied as a single date, will convert to a grid of datetime objects.
    If supplied as a file of dekad values, will convert to datetime object.
    """
    if isinstance(poi_in, datetime.date):
        poi = create_dataarray(poi_in, lon_min, lon_max, lat_min, lat_max)
        #print('Dealing with spatially constant date')
    else:
        # Read in start/emd of season (SOS) file and convert dekad number to datetime
        poi = convert_dekad_to_datetime(poi_in)
        #print('Dealing with spatially varying dates')
    
    return(poi.sel(time=poi.time.values[0]))


def process_weights(weights, current_date, fcast_date, poi_end):
    if isinstance(weights, list):
        # Post-process weights so that if single values are given, a map is created
        ds_weights = create_dataarray(poi_end_in, lon_min, lon_max, lat_min, lat_max).to_dataset(name='above')
        ds_weights['above'].values[:] = weights[0]
        ds_weights['normal'] = ds_weights['above'].copy()
        ds_weights['normal'].values[:] = weights[1]
        ds_weights['below'] = ds_weights['above'].copy()
        ds_weights['below'].values[:] = weights[2]
        ds_weights['above'] = ds_weights['above'].astype(float)
        ds_weights['normal'] = ds_weights['normal'].astype(float)
        ds_weights['below'] = ds_weights['below'].astype(float)
        met_forc_start_date = current_date + td(days=1)
        met_forc_end_date = poi_end.values.min()
    elif weights == 'ECMWF_S2S':
        # Download and read in tercile forecast file closest to the current_date/fcast_date - can use filelist method - need to add in check so it forecast period is 7 days away from fcast_date, no file is selected
        # Remove current version of the TAMSAT-ALERT ensemble forecast filenames list
        ecmwfs2s_flist_local = os.path.join(ecmwfs2s_dir, 'ecmwf-s2s_tercile_fcast_filelist.csv')
        if os.path.exists(ecmwfs2s_flist_local):
            os.remove(ecmwfs2s_flist_local)
        
        # Download latest list of TAMSAT-ALERT ensemble forecast filenames
        ecmwfs2s_flist_remote = os.path.join(remoteurl, 'ecmwf_s2s_tercile_prob', 'ecmwf-s2s_tercile_fcast_filelist.csv')
        os.chdir(ecmwfs2s_dir)
        try:
            wget.download(ecmwfs2s_flist_remote, bar=None)
        except:
            print('-> Warning! Unable to download file: %s' % ecmwfs2s_flist_remote)
        
        # Determine file closest to the fcast date
        df = pd.read_csv(ecmwfs2s_flist_local, header=None)
        ecmwfs2s_flist = [item for sublist in df.values.tolist() for item in sublist]
        ecmwfs2s_dates = [dt.strptime(x.split('_')[4][0:8], "%Y%m%d").date() for x in ecmwfs2s_flist]
        ecmwfs2s_file = min(ecmwfs2s_dates, key=lambda x: abs(x - fcast_date))
        ecmwfs2s_stamp = ecmwfs2s_file.strftime("%Y%m%d")
        
        # Download required ECMWF-S2S tercile forecast file
        ecmwfs2s_fname_local = os.path.join(ecmwfs2s_dir, 'ecmwfs2s_tp47_africa_terciles_' + ecmwfs2s_stamp + '.nc')
        if not os.path.exists(ecmwfs2s_fname_local):
            ecmwfs2s_fname_remote = os.path.join(remoteurl, 'ecmwf_s2s_tercile_prob', os.path.basename(ecmwfs2s_fname_local))
            os.chdir(ecmwfs2s_dir)
            try:
                '-> Downloading required ECMWF-S2S forecast file ...'
                wget.download(ecmwfs2s_fname_remote, bar=None)
            except:
                print('-> Warning! Unable to download file: %s' % ecmwfs2s_fname_remote)
        
        #s2s_flist = []
        #for root, dirs, files in os.walk(s2s_dir):
        #    for name in files:
        #        s2s_flist.append(os.path.join(root, name))   
        
        #s2s_dates = [dt.strptime(os.path.basename(x).split('_')[4][0:8], "%Y%m%d").date() for x in s2s_flist]
        #s2s_file = min(s2s_dates, key=lambda x: abs(x - current_date))
        #s2s_stamp = s2s_file.strftime("%Y-%m-%d")
        #s2s_fname = os.path.join(s2s_dir, 'ecmwfs2s_tp47_africa_terciles_' + s2s_stamp.replace('-', '') + '.nc')
        ds_weights = xr.open_dataset(ecmwfs2s_fname_local)
        ds_weights = ds_weights.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        met_forc_start_date = pd.to_datetime(ds_weights.start_time.values[0]).date()
        met_forc_end_date = pd.to_datetime(ds_weights.time.values[0]).date()
        ds_weights = ds_weights.isel(start_time=0).drop("start_time")
    
    return(ds_weights, met_forc_start_date, met_forc_end_date)


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict) 


def leap_remove_gridded(timeseries, datastartyear, timedim):
    """
    This function removes leap days from a time series 
    param timeseries: array containing daily time series
    param datastartyear: start year of the input data
    param timedim: time dimension location
    output data: time series with the leap days removed. 
    """
    data = timeseries
    leaplist=[]
    # system only takes 365 days in each year so we
    # remove leap year values from the long term time series
    if datastartyear % 4 == 1:  # if the start year is not a leap year (Matthew)
        for t in range(1154, data.shape[timedim], 1459):
            leaplist.append(t)
    elif datastartyear % 4 == 2:  # if the start year is not a leap year (Mark)
        for t in range(789, data.shape[timedim], 1459):
            leaplist.append(t)
    elif datastartyear % 4 == 3:  # if the start year is not a leap year (Luke)
        for t in range(424, data.shape[timedim], 1459):
            leaplist.append(t)
    elif datastartyear % 4 == 0:  # if the start year is a leap year (John)
        for t in range(59, data.shape[timedim], 1459):
            leaplist.append(t)
    data=np.delete(data,leaplist,axis=timedim)
    return data


def reshape_hist_data(datain, startyear):
    '''
    This function reorganises a historical daily time series (long,lat,time array) into an array with year per row. 
    It is assumed that the data start on January 1st. Leap days are removed.
    param datain: daily time series
    param startyear: first year in the daily time series
    param dataout: daily time series array reshaped as described above
    '''
    londimlen  = datain.shape[0]
    latdimlen  = datain.shape[1]
    datain = leap_remove_gridded(datain,startyear,2)
    timedimlen = datain.shape[2]
    extra_date = timedimlen % 365
    # add pseudo values to make the reshape work 
    # (i.e. add enough hours to make it an exact number of years worth of hours)
    sudovals = np.nan * np.ones((londimlen, latdimlen, (365 - extra_date)))
    datain = np.concatenate((datain,sudovals),axis=2)
    newtdim=int(datain.shape[2]//365)
    dataout = np.reshape(datain, (londimlen, latdimlen, newtdim, 365)).transpose((0,1,3,2))
    return dataout


def make_two_year_array(datain):
    tmp1=np.append(datain[:,:,:,0:-1],datain[:,:,:,1:],axis=2)
    sudovals = np.nan * np.ones((datain.shape[0], datain.shape[1],365))
    sudovals=np.expand_dims(sudovals,3)
    lastyear=datain[:,:,:,datain.shape[3]-1]
    lastyear=np.expand_dims(lastyear,3)
    lastyear = np.append(lastyear,sudovals,axis=2)
    dataout = np.append(tmp1,lastyear,axis=3)
    return(dataout)

# Import recent and forecast soil moisture data and spatially subset
# sm_recent_roi -> sm for current year(s)
# sm_fcast_roi -> sm forecasts from current_date out to 160 days
# fcast_date -> current_date + 1 day
def import_sm_data(poi_start, poi_end, current_date, sm_hist_dir, sm_fcast_fname, lon_min, lon_max, lat_min, lat_max):
    """ 
    Read in current and forecast soil moisture
    """ 
    # Get earliest date from poi_start map and latest date from poi_end
    poi_start_earliest = poi_start.values.min()
    poi_end_latest = poi_end.values.max()
    
    # Identify which historical files are needed (deals with year boundary)
    year_min = int(np.min([poi_start_earliest.year, current_date.year, poi_end_latest.year]))
    year_max = int(np.max([poi_start_earliest.year, current_date.year, poi_end_latest.year]))
    yr_list = np.arange(year_min, year_max + 1).tolist()
    
    # Read in recent sm file(s)
    sm_recent_list = []
    for root, dirs, files in os.walk(sm_hist_dir):
        for name in files:
            if int(name[2:6]) in yr_list:
                sm_recent_list.append(os.path.join(root, name))
    
    ds_sm_recent = xr.open_mfdataset(sm_recent_list)
    ds_sm_recent = ds_sm_recent.assign_coords(time=[pd.to_datetime(x).date() for x in ds_sm_recent['time'].values])
    da_sm_recent = ds_sm_recent['sm_c4grass']
    da_sm_recent_roi = da_sm_recent.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    
    # Read in relevant forecast sm file
    #sm_fcast_list = []
    #for root, dirs, files in os.walk(sm_fcast_dir):
    #    for name in files:
    #        sm_fcast_list.append(os.path.join(root, name))
    
    #fcast_dates = [dt.strptime(os.path.basename(x)[6:14], "%Y%m%d").date() for x in sm_fcast_list]
    #fcast_file = min(fcast_dates, key=lambda x: abs(x - current_date))
    #fcast_stamp = fcast_file.strftime("%Y-%m-%d")
    #fname = os.path.join(sm_fcast_dir, 'alert_' + fcast_stamp.replace('-', '') + '_ens.daily.nc')
    ds_sm_fcast = xr.open_dataset(sm_fcast_fname)
    ds_sm_fcast = ds_sm_fcast.assign_coords(time=[pd.to_datetime(x).date() for x in ds_sm_fcast['time'].values])
    ds_sm_fcast = ds_sm_fcast.assign_coords({"ens_year": np.arange(2005, 2019 + 1)})
    da_sm_fcast = ds_sm_fcast['sm_c4grass']
    da_sm_fcast = da_sm_fcast.rename({'longitude':'lon', 'latitude':'lat'})
    da_sm_fcast_roi = da_sm_fcast.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    
    # Define forecast date (depending on latest available forecast data)
    #fcast_date = dt.strptime(fcast_stamp, "%Y-%m-%d").date()
    
    return da_sm_recent_roi, da_sm_fcast_roi

# Splice data together recent and forecast soil moisture data for poi
# sm_poi_roi -> sm from poi_start to current_date and then sm forecasts from fcast_date out to 160 days
# sm_full_roi -> same as sm_poi_roi, but if fcast_date is before poi_start
def splice_sm_data(poi_start, poi_end, fcast_date, sm_recent_roi, sm_fcast_roi):
    if fcast_date <= poi_start.values.min():
        mask = (sm_fcast_roi['time'] >= fcast_date) & (sm_fcast_roi['time'] <= poi_end)  # Includes run up to season if forecast before season - for plotting later
        sm_full = sm_fcast_roi.where(mask) 
        sm_full = sm_full.sel(time=slice(fcast_date, poi_end.values.max()))
        sm_full = sm_full.transpose("lon", "lat", "time", "ens_year")
        mask = (sm_fcast_roi['time'] >= poi_start) & (sm_fcast_roi['time'] <= poi_end)  # Includes run up to season if forecast before season - for plotting later
        sm_poi = sm_fcast_roi.where(mask)
        sm_poi = sm_poi.sel(time=slice(poi_start.values.min(), poi_end.values.max()))
        sm_poi = sm_poi.transpose("lon", "lat", "time", "ens_year")
    else:
        mask = (sm_recent_roi['time'] >= poi_start) & (sm_recent_roi['time'] <= fcast_date-td(days = 1))
        sm_recent_splice = sm_recent_roi.where(mask)
        sm_recent_splice = sm_recent_splice.sel(time=slice(poi_start.values.min(), fcast_date-td(days = 1)))
        mask = (sm_fcast_roi['time'] >= fcast_date) & (sm_fcast_roi['time'] <= poi_end)
        sm_fcast_splice = sm_fcast_roi.where(mask)
        sm_fcast_splice = sm_fcast_splice.sel(time=slice(fcast_date, poi_end.values.max()))
        sm_poi = xr.merge(xr.broadcast(sm_recent_splice, sm_fcast_splice))["sm_c4grass"]
        sm_poi = sm_poi.transpose("lon", "lat", "time", "ens_year")
        sm_full = sm_poi # For plotting later
    
    return sm_poi.to_dataset(name='sm_c4grass'), sm_full.to_dataset(name='sm_c4grass')

# Import historical rainfall (for weighting)
# precip_hist_roi -> daily precip from 2005-2019
def import_hist_precip(clim_start_year, clim_end_year, rfe_dir, lon_min, lon_max, lat_min, lat_max):
    """ 
    Read in historical precipitation
    """
    # Read in historical sm file(s)
    rfe_list = []
    for root, dirs, files in os.walk(rfe_dir):
        for name in files:
            if 'sub.v3.1.nc' in name:
                rfe_list.append(os.path.join(root, name))
    
    rfe_list = [x for ind, x in enumerate(rfe_list) if int(os.path.basename(x).split('_')[2]) in np.arange(clim_start_year, clim_end_year + 1)]
    rfe_ds = xr.open_mfdataset(rfe_list)
    rfe_ds['precip'] = rfe_ds['rfe_filled']
    rfe = rfe_ds['precip']
    rfe = rfe.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    
    return rfe

# Import historical soil moisture data (for climatology)
# sm_hist_roi -> daily sm_c4grass from 2005-2019
def import_hist_sm(clim_start_year, clim_end_year, sm_hist_dir, lon_min, lon_max, lat_min, lat_max, remove_leapyear):
    """ 
    Read in historical soil moisture
    """
    # Read in historical sm file(s)
    sm_list = []
    for root, dirs, files in os.walk(sm_hist_dir):
        for name in files:
            if int(name[2:6]) in np.arange(clim_start_year, clim_end_year + 1):
                sm_list.append(os.path.join(root, name))
    
    sm_ds = xr.open_mfdataset(sm_list)
    sm = sm_ds['sm_c4grass']
    sm = sm.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    
    if remove_leapyear:
        sm = sm.sel(time=~((sm["time"].dt.month == 2) & (sm["time"].dt.day == 29)))
    
    return sm

# Weighting (precipitation)    
def weight_forecast(precip_hist_roi, met_forc_start_date, met_forc_end_date, poi_start, clim_start_year, clim_end_year, ds_weights):
    """ 
    Compute weightings for historical precip using forecast tercile weights
    """
    #clim_start_year = 2005
    #clim_end_year = 2019
    #met_forc_start_date = fcast_date
    #met_forc_end_date = fcast_date + td(days=40) # Need to align with the ECMWF S2S forecast period
    
    # Define climatological period
    clim_years = np.arange(clim_start_year, clim_end_year + 1)
    
    # Reshape precip data so that we can splice out the poi (and deal with year boundary)
    precip_hist_roi_reshape = reshape_hist_data(precip_hist_roi.transpose('lon','lat','time').values, clim_start_year)
    precip_hist_roi_reshape = make_two_year_array(precip_hist_roi_reshape)[:, :, :, 0:len(clim_years)]
    
    # Remove any bad precip values < 0
    precip_hist_roi_reshape[precip_hist_roi_reshape < 0] = np.nan
    
    # Assign coordinates for xarray
    lon = precip_hist_roi['lon'].values
    lat = precip_hist_roi['lat'].values
    times = np.arange(0, 730, 1) # just days for now, convert to dates in a min
    
    # Convert back to xarray
    ds_precip_hist_roi_reshape = xr.DataArray(precip_hist_roi_reshape, coords=[lon, lat, times, clim_years], dims=['lon', 'lat', 'time', 'ens_year']).to_dataset(name='rfe')
    
    # Remove leap year (29th Feb) if relevant
    year_tmp = pd.to_datetime(poi_start.time.values).year
    dates = pd.date_range(start=dt(year_tmp, 1, 1), end=dt(year_tmp + 1, 12, 31))
    if len(dates) > 730:
        if year_tmp%4 == 0:
            ind_delete = dates.get_loc(dt(year_tmp, 2, 29))
            dates = dates.delete(ind_delete)
        else:
            ind_delete = dates.get_loc(dt(year_tmp + 1, 2, 29))
            dates = dates.delete(ind_delete)
    
    # Fill dates and splice to period to which weighting applies
    ds_precip_hist_roi_reshape = ds_precip_hist_roi_reshape.assign_coords({"time": dates})
    precip_poi_roi = ds_precip_hist_roi_reshape.sel(time=slice(met_forc_start_date, met_forc_end_date))
    
    # Calculate mean precip for poi (e.g. seasonal mean)
    precip_poi_roi_mean = precip_poi_roi.mean(dim='time', skipna=True)
    
    # Calculate climatological mean and standard deviation for each lon-lat grid cell
    precip_clim_mean = precip_poi_roi_mean.mean(dim='ens_year', skipna=True)
    precip_clim_sd = precip_poi_roi_mean.std(dim='ens_year', skipna=True)
    
    # Calculate tercile boundaries for each lon-lat grid cell
    t1_thres = scipy.stats.norm(precip_clim_mean.rfe.values, precip_clim_sd.rfe.values).ppf(0.33)
    t2_thres = scipy.stats.norm(precip_clim_mean.rfe.values, precip_clim_sd.rfe.values).ppf(0.67)
    t1_thres = np.repeat(np.expand_dims(t1_thres, 2), repeats=precip_poi_roi_mean.rfe.shape[2], axis=2)
    t2_thres = np.repeat(np.expand_dims(t2_thres, 2), repeats=precip_poi_roi_mean.rfe.shape[2], axis=2)
    
    # Fill in weights for each year and lon-lat grid cell
    ds_weights_copy = ds_weights.copy()
    ds_weights_copy = ds_weights_copy.isel(time=0).drop("time")
    precip_weights = precip_poi_roi_mean.where(precip_poi_roi_mean > t1_thres, ds_weights_copy.below * 1e06)
    precip_weights = precip_weights.where(precip_weights > t2_thres, ds_weights_copy.normal * 1e06)
    precip_weights = precip_weights.where(precip_weights > 1000, ds_weights_copy.above * 1e06)
    
    # Tidy up
    precip_weights = precip_weights.rename({'rfe': 'weights'})
    precip_weights['weights'] = precip_weights['weights'] / 1e06
    precip_weights_masked_arr = precip_weights['weights'].to_masked_array()
    da_precip_weights_masked = xr.DataArray(precip_weights_masked_arr.filled(np.nan), dims=['lon', 'lat', 'ens_year'], coords={'lon': lon, 'lat': lat, 'ens_year': clim_years})
    precip_weights['weights_masked'] = da_precip_weights_masked
    
    return precip_weights

# Calculate soil moisture climatology (note: need to check period to crop over when computing the seasonal mean)
def calc_sm_climatology(sm_hist_roi, clim_start_year, clim_end_year, fcast_date, poi_start, poi_end):
    #clim_start_year = 2005
    #clim_end_year = 2019
    
    # Define climatological period
    clim_years = np.arange(clim_start_year, clim_end_year + 1)
    
    # Reshape sm hist data so that we can splice out the poi (and deal with year boundary)
    sm_hist_roi_reshape = reshape_hist_data(sm_hist_roi.transpose('lon', 'lat', 'time').values, clim_start_year)
    sm_hist_roi_reshape = make_two_year_array(sm_hist_roi_reshape)[:,:,:,0:len(clim_years)]
    
    # Assign coordinates for xarray
    lon = sm_hist_roi['lon'].values
    lat = sm_hist_roi['lat'].values
    times = np.arange(0, 730, 1) # just days for now, convert to dates in a min
    
    # Convert back to xarray
    ds_sm_hist_roi_reshape = xr.DataArray(sm_hist_roi_reshape, coords=[lon, lat, times, clim_years], dims=['lon', 'lat', 'time', 'ens_year']).to_dataset(name='sm_c4grass')
    
    # Remove leap year (29th Feb) if relevant
    year_tmp = pd.to_datetime(poi_start.time.values).year
    dates = pd.date_range(start=dt(year_tmp, 1, 1), end=dt(year_tmp + 1, 12, 31))
    if len(dates) > 730:
        if year_tmp%4 == 0:
            ind_delete = dates.get_loc(dt(year_tmp, 2, 29))
            dates = dates.delete(ind_delete)
        else:
            ind_delete = dates.get_loc(dt(year_tmp + 1, 2, 29))
            dates = dates.delete(ind_delete)
    
    dates = [pd.to_datetime(x).date() for x in dates]
    
    # Fill dates and splice to period to which weighting applies
    ds_sm_hist_roi_reshape = ds_sm_hist_roi_reshape.assign_coords({"time": dates})
    
    mask = (ds_sm_hist_roi_reshape['time'] >= poi_start) & (ds_sm_hist_roi_reshape['time'] <= poi_end)
    sm_hist_poi_roi = ds_sm_hist_roi_reshape.where(mask)
    sm_hist_poi_roi = sm_hist_poi_roi.sel(time=slice(poi_start.values.min(), poi_end.values.max()))
    sm_hist_poi_roi = sm_hist_poi_roi.assign_coords(time=sm_hist_poi_roi.time.values.astype("datetime64[s]"))
    #sm_hist_poi_roi = ds_sm_hist_roi_reshape.sel(time=slice(poi_start.min(), poi_end.max()))
    
    # Splice to full period - if forecast date before poi start
    if fcast_date <= poi_start.min().values:
        mask = (ds_sm_hist_roi_reshape['time'] >= fcast_date) & (ds_sm_hist_roi_reshape['time'] <= poi_end)
        sm_hist_full_roi = ds_sm_hist_roi_reshape.where(mask)
        #sm_hist_full_roi = ds_sm_hist_roi_reshape.sel(time=slice(fcast_date, poi_end.max()))
    else:
        sm_hist_full_roi = sm_hist_poi_roi
    
    # Calculate mean sm for poi (e.g. seasonal mean)
    sm_hist_poi_roi_mean = sm_hist_poi_roi.mean(dim='time', skipna=True)
    
    # Added by RM for EOCIS AIP
    # Compute sm clim from poi_start to current_date across clim years (also known as WRSI current)
    mask = (ds_sm_hist_roi_reshape['time'] >= poi_start) & (ds_sm_hist_roi_reshape['time'] <= fcast_date - td(days=1))
    sm_hist_current_roi = ds_sm_hist_roi_reshape.where(mask)
    sm_hist_current_roi = sm_hist_current_roi.assign_coords(time=sm_hist_current_roi.time.values.astype("datetime64[s]"))
    #for yyyy in np.arange(2005, 2020):
    #    sm_hist_current_roi.sm_c4grass.sel(lon=38, lat=4, method='nearest', ens_year=yyyy).plot()
    #
    #plt.show()
    sm_hist_current_roi_mean = sm_hist_current_roi.mean(dim='time', skipna=True).mean(dim='ens_year', skipna=True)
    
    return sm_hist_full_roi.sm_c4grass.sel(ens_year=slice(ens_clim_start_year, ens_clim_end_year)).to_dataset(), sm_hist_poi_roi_mean, sm_hist_current_roi, sm_hist_current_roi_mean

# Summary statistics
def summary_stats(sm_hist_poi_roi_mean, weights, sm_poi_roi, sm_full_roi, ens_clim_start_year, ens_clim_end_year):
    lon = sm_hist_poi_roi_mean['lon']
    lat = sm_hist_poi_roi_mean['lat']
    sm_hist_poi_roi_mean_ensyears = sm_hist_poi_roi_mean.sel(ens_year=slice(ens_clim_start_year, ens_clim_end_year)) # Using only 2005-2019 as these are the ensemble forecast years
    
    # Calculate climatological mean and standard deviation from historical data for poi
    clim_mean_wrsi = np.average(sm_hist_poi_roi_mean_ensyears.sm_c4grass.values, axis = 2)
    av = np.repeat(np.expand_dims(clim_mean_wrsi, axis = 2), repeats=sm_hist_poi_roi_mean_ensyears.sm_c4grass.shape[2], axis=2)
    clim_sd_wrsi = np.sqrt(np.average(np.abs(sm_hist_poi_roi_mean_ensyears.sm_c4grass.values - av), axis = 2)) 
    
    # Calculate forecast mean beta for each year
    beta_fcast_poi_roi_mean = np.nanmean(sm_poi_roi.sm_c4grass.values, axis=2)
    
    # Calculate ensemble mean and standard deviation for forecasts
    ens_mean_wrsi = np.average(beta_fcast_poi_roi_mean, weights=weights.weights_masked.values, axis=-1) 
    av = np.repeat(np.expand_dims(ens_mean_wrsi, axis=-1), repeats = beta_fcast_poi_roi_mean.shape[-1], axis=-1)
    ens_sd_wrsi = np.sqrt(np.average(np.abs(beta_fcast_poi_roi_mean - av), weights=weights.weights_masked.values, axis=-1))
    
    # Ensemble forecast - already an xarray
    ensemble_forecast = sm_full_roi # includes run up to season if forecast before season - for plotting
    
    # Convert to xarrays
    ens_mean_wrsi_xr = xr.DataArray(ens_mean_wrsi, coords=[lon, lat], dims=['lon', 'lat'])
    ens_sd_wrsi_xr = xr.DataArray(ens_sd_wrsi, coords=[lon, lat], dims=['lon', 'lat'])
    clim_mean_wrsi_xr = xr.DataArray(clim_mean_wrsi, coords=[lon, lat], dims=['lon', 'lat'])
    clim_sd_wrsi_xr = xr.DataArray(clim_sd_wrsi, coords=[lon, lat], dims=['lon', 'lat'])
    
    return ens_mean_wrsi_xr, ens_sd_wrsi_xr, clim_mean_wrsi_xr, clim_sd_wrsi_xr, ensemble_forecast

# Date stamps for output files
def date_stamps(fcast_date, poi_start, poi_end, lon_min, lon_max, lat_min, lat_max):
    # Forecast date stamps
    currentdate_stamp = current_date.strftime("%Y-%m-%d")
    fcast_stamp = fcast_date.strftime("%Y-%m-%d")
    
    # POI stamps
    start_month = poi_start.values.min().month
    end_month = poi_end.values.min().month
    poi_months = np.arange(start_month, end_month + 1, 1)
    poi_year = poi_start.values.min().year
    
    poi_str = ""
    for mo in np.arange(0, len(poi_months)):
        tmp_date = datetime.datetime(2020 ,poi_months[mo],1).strftime("%b")[0]
        poi_str += tmp_date
    
    poi_stamp = poi_str + '-' + str(poi_year)
    
    #if lon_point != "NA":
    #    loc_stamp = str(lon_point) + "_" + str(lat_point)
    #else:
    loc_stamp = str(lon_min) + "_" + str(lon_max) + "_" + str(lat_min) + "_" + str(lat_max)
    
    return fcast_stamp, poi_stamp, poi_str, loc_stamp, currentdate_stamp

# Outputs
def output_forecasts(ens_mean_wrsi_xr, ens_sd_wrsi_xr, clim_mean_wrsi_xr, clim_sd_wrsi_xr, ensemble_forecast, sm_recent_roi, sm_hist_full_roi, sm_hist_current_roi_mean, poi_stamp, forecast_stamp, clim_start_year, clim_end_year, poi_start, poi_end, poi_str, fcast_date, loc_stamp, currentdate_stamp):    
    # Save output files
    ds_ens_mean_wrsi = ens_mean_wrsi_xr.to_dataset(name='ens_mean_wrsi')
    ds_ens_sd_wrsi = ens_sd_wrsi_xr.to_dataset(name='ens_sd_wrsi')
    ds_clim_mean_wrsi = clim_mean_wrsi_xr.to_dataset(name='clim_mean_wrsi')
    ds_clim_sd_wrsi = clim_sd_wrsi_xr.to_dataset(name='clim_sd_wrsi')
    ds_out = xr.combine_by_coords([ds_ens_mean_wrsi, ds_ens_sd_wrsi, ds_clim_mean_wrsi, ds_clim_sd_wrsi])
    fname = os.path.join(datadir, 'wrsi_' + poi_stamp + '_' + forecast_stamp + '_' + loc_stamp + '.nc')
    ds_out.to_netcdf(fname)
    fname = os.path.join(datadir, 'ensemble_forecast_wrsi_' + poi_stamp + '_' + forecast_stamp + '_' + loc_stamp + '.nc')
    ensemble_forecast = ensemble_forecast.assign_coords(time=ensemble_forecast.time.values.astype("datetime64[s]"))
    ensemble_forecast.to_netcdf(fname)
    
    terciles_text(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp)
    prob_dist_plot(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp)
    ensemble_timeseries_plot(ensemble_forecast, fcast_date, poi_start, poi_end, sm_hist_full_roi, poi_stamp, forecast_stamp, loc_stamp)    
    
    #if sys.argv[7] == "region":
    wrsi_current_plot(sm_recent_roi, sm_hist_current_roi_mean, clim_mean_wrsi_xr, ens_mean_wrsi_xr, poi_start, current_date, poi_stamp, clim_start_year, clim_end_year, loc_stamp, currentdate_stamp, forecast_stamp)
    wrsi_forecast_plot(clim_mean_wrsi_xr, ens_mean_wrsi_xr, poi_stamp, forecast_stamp, clim_start_year, clim_end_year, poi_str, loc_stamp)
    prob_map_plot(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp)

# Calculate tercile probabilities
def terciles_text(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp):
    # Calculate probability of lower tercile soil moisture
    a = scipy.stats.norm(clim_mean_wrsi_xr, clim_sd_wrsi_xr).ppf(0.33)
    b_lower = scipy.stats.norm(ens_mean_wrsi_xr, ens_sd_wrsi_xr).cdf(a)
    # Calculate probability of mid tercile soil moisture
    a = scipy.stats.norm(clim_mean_wrsi_xr, clim_sd_wrsi_xr).ppf(0.67)
    b_upper = scipy.stats.norm(ens_mean_wrsi_xr, ens_sd_wrsi_xr).cdf(a)
    # Calculate all terciles
    lower_terc = b_lower
    middle_terc = b_upper - b_lower
    upper_terc = 1 - b_upper
    # Calculate mean of all terciles
    lower_terc = np.nanmean(lower_terc)
    middle_terc = np.nanmean(middle_terc)
    upper_terc = np.nanmean(upper_terc)
    
    df = pd.DataFrame({'Category': ['Probability of seasonal mean soil moisture (sm_c4grass/beta) falling into lower tercile: %s' % lower_terc.round(3), 
                            'Probability of seasonal mean soil moisture (sm_c4grass/beta) falling into middle tercile: %s' % middle_terc.round(3), 
                            'Probability of seasonal mean soil moisture (sm_c4grass/beta) falling into upper tercile: %s' % upper_terc.round(3)]})
    fname = os.path.join(datadir, 'terciles_' + poi_stamp + '_' + forecast_stamp + '_' + loc_stamp + '.csv')
    df.to_csv(fname, header=False, index=False)
    print('-> Tercile probabilities for end of season WRSI:')
    print("    Lower : %s" % str(round(lower_terc, 3)))
    print("    Middle: %s" % str(round(middle_terc, 3)))
    print("    Upper : %s" % str(round(upper_terc, 3)))

# Plot probability distributions
def prob_dist_plot(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp):
    # Define tercile boundaries
    lower_thresh = 0.33
    upper_thresh = 0.67
    # Create probability distribution of climatology
    clim_dist = np.random.normal(loc = np.nanmean(clim_mean_wrsi_xr), scale = np.nanmean(clim_sd_wrsi_xr), size = 10000)
    # Create probability distribution of ensemble forecast
    ens_dist = np.random.normal(loc = np.nanmean(ens_mean_wrsi_xr), scale = np.nanmean(ens_sd_wrsi_xr), size = 10000)
    # Calculate tercile thresholds
    low_a = np.nanmean(scipy.stats.norm(np.nanmean(clim_mean_wrsi_xr), np.nanmean(clim_sd_wrsi_xr)).ppf(lower_thresh))
    up_a = np.nanmean(scipy.stats.norm(np.nanmean(clim_mean_wrsi_xr), np.nanmean(clim_sd_wrsi_xr)).ppf(upper_thresh))
    # Convert to pandas series - for plotting purposes
    clim_dist_pd = pd.Series(clim_dist)
    ens_dist_pd = pd.Series(ens_dist)
    # Get plotting parameters - x axis limits
    clim_dist_pd.plot.hist()
    ens_dist_pd.plot.hist()
    ax = plt.gca()
    xlims = ax.get_xlim()
    plt.close()
    # Build plot
    plt.figure(figsize = (6,4))
    clim_dist_pd.plot.density(color = "black", linewidth = 2, xlim = (xlims), label = "Climatology")
    ens_dist_pd.plot.density(color = "red", linewidth = 2, label = "Forecast")
    plt.xlabel("Seasonal mean SM (beta)", fontweight = "bold")
    plt.ylabel("Probability", fontweight = "bold")
    plt.axvline(low_a, color = "grey", linestyle = "--", label = "Tercile boundaries")
    plt.axvline(up_a, color = "grey", linestyle = "--")
    plt.legend(loc = 2)
    # Save plot
    fname = os.path.join(plotsdir, 'probdist_' + poi_stamp + '_' + forecast_stamp + '_' + loc_stamp + '.png')
    plt.savefig(fname, dpi=300)
    plt.close()

# Plot ensemble forecast compared to climatology
def ensemble_timeseries_plot(ensemble_forecast, fcast_date, poi_start, poi_end, sm_hist_full_roi, poi_stamp, forecast_stamp, loc_stamp):
    # Create data frame of dates
    #date_labs = pd.to_datetime(ensemble_forecast['time'].values)
    date_labs = pd.to_datetime(sm_hist_full_roi['time'].values)
    # Setup plot
    plt.figure(figsize = (7,4))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))
    # Depending on positioning of forecast date relative to poi
    if fcast_date <= poi_start.values.min(): # If all forecast
        plt.plot(date_labs, np.nanmean(ensemble_forecast, axis=(0, 1)), color="grey", label="Ensemble forecast")   
        plt.plot(fcast_date, np.nanmean(ensemble_forecast.sel(time = fcast_date)), 
                    marker="o", color="red", markersize=8, label="Forecast date")
    else: # If some observed and some forecast
        obs = ensemble_forecast.sel(time=slice(poi_start.values.min(), fcast_date - datetime.timedelta(days=1)))
        fcast = ensemble_forecast.sel(time=slice(fcast_date - datetime.timedelta(days=1), poi_end.values.min()))
        date_obs = pd.to_datetime(obs["time"].values)
        date_fcast = pd.to_datetime(fcast["time"].values)
        plt.plot(date_obs, np.nanmean(obs.sm_c4grass.values, axis=(0, 1)), color = "black", label="Observed")
        plt.plot(date_fcast, np.nanmean(fcast.sm_c4grass.values, axis=(0, 1)), color = "grey", label="Ensemble forecast")
        plt.plot(fcast_date - datetime.timedelta(days=1), np.nanmean(ensemble_forecast.sm_c4grass.sel(time=np.datetime64(fcast_date, 's'))),
                    marker="o", color="red", markersize=8, label = "Forecast date")    
    # Continue with plotting visuals
    plt.fill_between(date_labs, np.nanpercentile(np.nanmean(sm_hist_full_roi.sm_c4grass.values, axis=(0, 1)), 5, axis = (1)), 
                    np.nanpercentile(np.nanmean(sm_hist_full_roi.sm_c4grass.values, axis=(0, 1)), 95, axis=(1)), 
                        alpha=0.35, color="grey", label="5th-95th percentile")
    plt.axvline(poi_start.values.min(), color="black", linestyle="--", label="POI boundaries")
    plt.axvline(poi_end.values.min(), color="black", linestyle="--")
    plt.ylabel("Soil moisture (beta)", fontweight="bold")
    plt.gcf().autofmt_xdate()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc=2)
    fname = os.path.join(plotsdir, 'timeseries_' + poi_stamp + '_' + forecast_stamp + '_' + loc_stamp + '.png')
    plt.savefig(fname, dpi=300)
    plt.close()

# Plot WRSI forecast maps
def wrsi_forecast_plot(clim_mean_wrsi_xr, ens_mean_wrsi_xr, poi_stamp, forecast_stamp, clim_start_year, clim_end_year, poi_str, loc_stamp):
    # Extract lons and lats for plotting axies
    lons = clim_mean_wrsi_xr['lon'].values
    lats = clim_mean_wrsi_xr['lat'].values
    # Calculate max values to standardised colorbars on both plots
    vmax = np.nanmax([clim_mean_wrsi_xr, ens_mean_wrsi_xr])
    # Calculate percent anomaly    
    percent_anom = (ens_mean_wrsi_xr / clim_mean_wrsi_xr) * 100
    # Save to netCDF - perc_anom
    percent_anom_xr = xr.DataArray(percent_anom, coords = [lons,lats], dims = ['longitude','latitude'])
    fname = os.path.join(datadir, 'wrsi-forecast_' + poi_stamp + '_' + forecast_stamp + '_' + loc_stamp + '.nc')
    percent_anom_xr.to_netcdf(fname)
    # Colormap setup - make 'bad' values grey
    BrBG_cust = matplotlib.cm.get_cmap("BrBG")
    BrBG_cust.set_bad(color = "silver")
    RdBu_cust = matplotlib.cm.get_cmap("RdBu")
    RdBu_cust.set_bad(color = "silver")
    # Build plot
    fig = plt.figure(figsize = (32,10))
    # Plot climatology
    clim_plt = fig.add_subplot(131, projection = ccrs.PlateCarree())
    clim_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    clim_plt.pcolormesh(lons, lats, clim_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    clim_gl = clim_plt.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    clim_gl.top_labels = False
    clim_gl.right_labels = False
    clim_gl.xlabel_style = {'size': 18}
    clim_gl.ylabel_style = {'size': 18}
    clim_gl.xformatter = LONGITUDE_FORMATTER
    clim_gl.yformatter = LATITUDE_FORMATTER
    #clim_plt.set_title('SM (beta) climatology\n' + poi_str + ' ' + str(clim_start_year) + '-' + str(clim_end_year), fontsize = 20)
    clim_plt.set_title('WRSI climatology (' + str(clim_start_year) + '-' + str(clim_end_year) + ')\n' + poi_str, fontsize = 20)
    clim_cb = plt.pcolormesh(lons, lats, clim_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    clim_cb = plt.colorbar(clim_cb)
    clim_cb.ax.tick_params(labelsize=18)
    clim_cb.ax.tick_params(top=False, right=False)
    clim_plt.set_aspect("auto", adjustable = None)
    clim_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    clim_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    clim_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Plot forecast
    ens_plt = fig.add_subplot(132, projection = ccrs.PlateCarree())
    ens_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    ens_plt.pcolormesh(lons, lats, ens_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    ens_gl = ens_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    ens_gl.top_labels = False
    ens_gl.right_labels = False
    ens_gl.xlabel_style = {'size': 18}
    ens_gl.ylabel_style = {'size': 18}
    ens_gl.xformatter = LONGITUDE_FORMATTER
    ens_gl.yformatter = LATITUDE_FORMATTER
    #ens_plt.set_title('SM (beta) forecast for ' + poi_stamp + "\nIssued "+ forecast_stamp, fontsize = 20)
    ens_plt.set_title('WRSI forecast for ' + poi_stamp + "\nIssued "+ forecast_stamp, fontsize = 20)
    ens_cb = plt.pcolormesh(lons, lats, ens_mean_wrsi_xr.T, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    ens_cb = plt.colorbar(ens_cb)
    ens_cb.ax.tick_params(labelsize=18)
    ens_plt.set_aspect("auto", adjustable = None)
    ens_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    ens_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    ens_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Plot anomaly
    anom_plt = fig.add_subplot(133, projection = ccrs.PlateCarree())
    anom_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    anom_plt.pcolormesh(lons, lats, percent_anom.T, vmin = 50, vmax = 150, cmap = RdBu_cust)
    anom_gl = anom_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    anom_gl.top_labels = False
    anom_gl.right_labels = False
    anom_gl.xlabel_style = {'size': 18}
    anom_gl.ylabel_style = {'size': 18}
    anom_gl.xformatter = LONGITUDE_FORMATTER
    anom_gl.yformatter = LATITUDE_FORMATTER
    #anom_plt.set_title('SM (beta) % anomaly for ' + poi_stamp + "\nIssued "+ forecast_stamp, fontsize = 20)
    anom_plt.set_title('WRSI % anomaly for ' + poi_stamp + "\nIssued "+ forecast_stamp, fontsize = 20)
    anom_cb = plt.pcolormesh(lons, lats, percent_anom.T, vmin = 50, vmax = 150, cmap = RdBu_cust)
    anom_cb = plt.colorbar(anom_cb)
    anom_cb.ax.tick_params(labelsize=18)
    anom_plt.set_aspect("auto", adjustable = None)
    anom_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    anom_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    anom_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Save and show
    fname = os.path.join(plotsdir, 'wrsi-forecast_map_' + poi_stamp + '_' + forecast_stamp + '_' + loc_stamp + '.png')
    plt.savefig(fname, dpi=300)
    plt.close()

# Plot WRSI current maps
def wrsi_current_plot(sm_recent_roi, sm_hist_current_roi_mean, clim_mean_wrsi_xr, ens_mean_wrsi_xr, poi_start, current_date, poi_stamp, clim_start_year, clim_end_year, loc_stamp, currentdate_stamp, forecast_stamp):
    # WRSI current (from season start to current date)
    mask = (sm_recent_roi['time'] >= poi_start) & (sm_recent_roi['time'] <= current_date)
    sm_current = sm_recent_roi.where(mask)
    sm_wrsi_current = sm_current.mean(dim='time')
    #sm_current.sel(lon=38, lat=4, method='nearest').plot(); plt.show()
    
    # WRSI current climatology
    sm_wrsi_current_clim = sm_hist_current_roi_mean.sm_c4grass.transpose()
    
    # WRSI current anomaly
    sm_wrsi_current_anom = sm_wrsi_current - sm_wrsi_current_clim
    
    # WRSI current % anomaly
    sm_wrsi_current_percent_anom = (sm_wrsi_current / sm_wrsi_current_clim) * 100
    
    # Save WRSI current to netCDF file
    ds_wrsi_current = xr.combine_by_coords([sm_wrsi_current.to_dataset(name='wrsi_current'),
                                            sm_wrsi_current_clim.to_dataset(name='wrsi_current_clim'),
                                            sm_wrsi_current_anom.to_dataset(name='wrsi_current_anom'),
                                            sm_wrsi_current_percent_anom.to_dataset(name='wrsi_current_percent_anom')])
    
    fname = os.path.join(datadir, 'wrsi-current_' + poi_stamp + '_' + forecast_stamp + '_' + loc_stamp + '.nc')
    ds_wrsi_current.to_netcdf(fname)
    
    # Checks
    #for yyyy in np.arange(2005, 2020):
    #    sm_hist_current_roi.sm_c4grass.sel(lon=38, lat=4, method='nearest', ens_year=yyyy).plot(color='gray')
    # 
    #sm_hist_current_roi.sm_c4grass.mean(dim='ens_year').sel(lon=38, lat=4, method='nearest').plot(color='black')
    #sm_current.sel(lon=38, lat=4, method='nearest').plot(color='red')
    #plt.show()
    
    # Extract lons and lats for plotting axies
    lons = sm_wrsi_current_clim['lon'].values
    lats = sm_wrsi_current_clim['lat'].values
    # Calculate max values to standardised colorbars on both plots
    vmax = np.nanmax([sm_wrsi_current, sm_wrsi_current_clim, clim_mean_wrsi_xr, ens_mean_wrsi_xr]) 
    
    # Colormap setup - make 'bad' values grey
    BrBG_cust = matplotlib.cm.get_cmap("BrBG")
    BrBG_cust.set_bad(color = "silver")
    RdBu_cust = matplotlib.cm.get_cmap("RdBu")
    RdBu_cust.set_bad(color = "silver")
    # Build plot
    fig = plt.figure(figsize = (32,10))
    # Plot WRSI current climatology
    clim_plt = fig.add_subplot(131, projection = ccrs.PlateCarree())
    clim_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    clim_plt.pcolormesh(lons, lats, sm_wrsi_current_clim, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    clim_gl = clim_plt.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    clim_gl.top_labels = False
    clim_gl.right_labels = False
    clim_gl.xlabel_style = {'size': 18}
    clim_gl.ylabel_style = {'size': 18}
    clim_gl.xformatter = LONGITUDE_FORMATTER
    clim_gl.yformatter = LATITUDE_FORMATTER
    clim_plt.set_title('WRSI climatology ' + '(' + str(clim_start_year) + '-' + str(clim_end_year) + ')\n from season start until %s' % currentdate_stamp, fontsize = 20)
    clim_cb = plt.pcolormesh(lons, lats, sm_wrsi_current_clim, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    clim_cb = plt.colorbar(clim_cb)
    clim_cb.ax.tick_params(labelsize=18)
    clim_cb.ax.tick_params(top=False, right=False)
    clim_plt.set_aspect("auto", adjustable = None)
    clim_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    clim_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    clim_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Plot WRSI current
    ens_plt = fig.add_subplot(132, projection = ccrs.PlateCarree())
    ens_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    ens_plt.pcolormesh(lons, lats, sm_wrsi_current, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    ens_gl = ens_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    ens_gl.top_labels = False
    ens_gl.right_labels = False
    ens_gl.xlabel_style = {'size': 18}
    ens_gl.ylabel_style = {'size': 18}
    ens_gl.xformatter = LONGITUDE_FORMATTER
    ens_gl.yformatter = LATITUDE_FORMATTER
    ens_plt.set_title('WRSI from season start until %s' % currentdate_stamp , fontsize = 20)
    ens_cb = plt.pcolormesh(lons, lats, sm_wrsi_current, vmin = 0, vmax = vmax, cmap = BrBG_cust)
    ens_cb = plt.colorbar(ens_cb)
    ens_cb.ax.tick_params(labelsize=18)
    ens_plt.set_aspect("auto", adjustable = None)
    ens_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    ens_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    ens_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Plot anomaly
    anom_plt = fig.add_subplot(133, projection = ccrs.PlateCarree())
    anom_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    anom_plt.pcolormesh(lons, lats, sm_wrsi_current_percent_anom, vmin = 50, vmax = 150, cmap = RdBu_cust)
    anom_gl = anom_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    anom_gl.top_labels = False
    anom_gl.right_labels = False
    anom_gl.xlabel_style = {'size': 18}
    anom_gl.ylabel_style = {'size': 18}
    anom_gl.xformatter = LONGITUDE_FORMATTER
    anom_gl.yformatter = LATITUDE_FORMATTER
    anom_plt.set_title('WRSI % anomaly \nfrom season start until ' + currentdate_stamp, fontsize = 20)
    anom_cb = plt.pcolormesh(lons, lats, sm_wrsi_current_percent_anom, vmin = 50, vmax = 150, cmap = RdBu_cust)
    anom_cb = plt.colorbar(anom_cb)
    anom_cb.ax.tick_params(labelsize=18)
    anom_plt.set_aspect("auto", adjustable = None)
    anom_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    anom_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    anom_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Save and show
    fname = os.path.join(plotsdir, 'wrsi-current_map_' + poi_stamp + '_' + currentdate_stamp + '_' + loc_stamp + '.png')
    plt.savefig(fname, dpi=300)
    plt.close()

# Plot probability of lower tercile map
def prob_map_plot(clim_mean_wrsi_xr, clim_sd_wrsi_xr, ens_mean_wrsi_xr, ens_sd_wrsi_xr, poi_stamp, forecast_stamp, loc_stamp):
# Extract lons and lats for plotting axies
    lons = clim_mean_wrsi_xr['lon'].values
    lats = clim_mean_wrsi_xr['lat'].values
    lower_thresh = 0.33
    # Calculate probability of lower tercile soil moisture
    a = scipy.stats.norm(clim_mean_wrsi_xr, clim_sd_wrsi_xr).ppf(0.33)
    b_lower = scipy.stats.norm(ens_mean_wrsi_xr, ens_sd_wrsi_xr).cdf(a)
    # Save to netCDF - prob_lower_terc
    b_lower_xr = xr.DataArray(b_lower, coords = [lons,lats], dims = ['longitude','latitude'])
    fname = os.path.join(datadir, 'prob_lower_tercile_' + poi_stamp + '_' + forecast_stamp + '_' + loc_stamp + '.nc')
    b_lower_xr.to_netcdf(fname)
    # Colormap setup - make 'bad' values grey
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap([c('green'), c('palegreen'), lower_thresh - 0.05, c('white'), c('white'), lower_thresh + 0.05, c('yellow'), c('brown')])
    rvb_cust = matplotlib.cm.get_cmap(rvb)
    rvb_cust.set_bad(color = "silver")
    # Extract lons and lats for plotting axies
    lons = clim_mean_wrsi_xr['lon'].values
    lats = clim_mean_wrsi_xr['lat'].values    
    # Build plot
    fig = plt.figure(figsize = (10,10))
    # Plot climatology
    prob_plt = fig.add_subplot(111, projection = ccrs.PlateCarree())
    prob_plt.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    prob_plt.pcolormesh(lons, lats, b_lower.T, vmin = 0, vmax = 1, cmap = rvb_cust)
    prob_gl = prob_plt.gridlines(crs = ccrs.PlateCarree(), draw_labels = True, linewidth = 0.7, color = "black", alpha = 1, linestyle = "--")
    prob_gl.top_labels = False
    prob_gl.right_labels = False
    prob_gl.xlabel_style = {'size': 18}
    prob_gl.ylabel_style = {'size': 18}
    prob_gl.xformatter = LONGITUDE_FORMATTER
    prob_gl.yformatter = LATITUDE_FORMATTER
    prob_plt.set_title('Probability of lower tercile SM\n' + poi_stamp + " Issued "+ forecast_stamp, fontsize = 20)
    prob_cb = plt.pcolormesh(lons, lats, b_lower.T, vmin = 0, vmax = 1, cmap = rvb_cust)
    prob_cb = plt.colorbar(prob_cb)
    prob_cb.ax.tick_params(labelsize=18)
    prob_plt.set_aspect("auto", adjustable = None)
    prob_plt.add_feature(cfeature.OCEAN, facecolor = "white", zorder = 1)
    prob_plt.add_feature(cfeature.COASTLINE, linewidth = 2)
    prob_plt.add_feature(cfeature.BORDERS, linewidth = 2)
    # Save and show
    fname = os.path.join(plotsdir, 'prob_map_plot_' + poi_stamp + '_' + forecast_stamp + '_' + loc_stamp + '.png')
    plt.savefig(fname)
    plt.close()

# Create summary of API input arguments
def create_inputs_summary_csv(csv_fname):
    # Create text file summarising input arguments
    data = [
        ["Season Start", poi_start_in],
        ["Season End", poi_end_in],
        ["Current Date", current_date],
        ["Climatological Period", f"{clim_start_year}-{clim_end_year}"],
        ["Domain Coordinates", f"N:{lat_max}, S:{lat_min}, W:{lon_min}, E:{lon_max}"],
        [
            "Tercile Weights",
            "Derived from ECMWF S2S precipitation forecasts"
            if weights == "ECMWF_S2S"
            else f"Upper:{weights[0]}, Middle:{weights[1]}, Lower:{weights[2]}"
        ],
    ]
    
    # Convert to a DataFrame
    df = pd.DataFrame(data, columns=["Attribute", "Value"])
    
    # Save to a CSV file
    df.to_csv(csv_fname, index=False)

# Wrapper for everything
def wrapper(current_date):
    # 1. Post-process poi_start and poi_end so that if spatially varying, dates are in datetime format and if fixed, a map of datetime objects is created
    poi_start = check_poi(poi_start_in, lon_min, lon_max, lat_min, lat_max)
    poi_end = check_poi(poi_end_in, lon_min, lon_max, lat_min, lat_max)
    # 2. Does some checks on the current_date before proceeding 
    current_date = check_current_date(current_date, sm_hist_dir, poi_end)
    # 3. Given inputs, get years needed
    yearstart, yearend = get_year_range(clim_start_year, clim_end_year, poi_start, poi_end)
    # 4. Download soil moisture and rainfall (for weighting)
    download_historical_data(remoteurl, 'sm_hist', sm_version, yearstart, yearend)
    download_historical_data(remoteurl, 'rfe_hist', rfe_version, yearstart, yearend)
    # 5. Download TAMSAT-ALERT forecasts
    fcast_date, sm_fcast_fname = download_forecast_data(remoteurl, sm_fcast_dir, sm_version, current_date)
    # 6. Post-process weights so that they are in an xarray dataset retrives start/end dates to apply the tercile weights over
    ds_weights, met_forc_start_date, met_forc_end_date = process_weights(weights, current_date, fcast_date, poi_end)
    # 7. Import recent soil moisture estimates and forecasts and subset
    sm_recent_roi, sm_fcast_roi = import_sm_data(poi_start, poi_end, current_date, sm_hist_dir, sm_fcast_fname, lon_min, lon_max, lat_min, lat_max)
    # 8. Splice together recent soil moisture estimates and forecasts
    sm_poi_roi, sm_full_roi = splice_sm_data(poi_start, poi_end, fcast_date, sm_recent_roi, sm_fcast_roi)
    # Sanity check
    #for yyyy in np.arange(2005, 2020):
    #    sm_poi_roi.sm_c4grass.sel(lon=38, lat=3, method='nearest', ens_year=yyyy).plot()
    #
    #plt.show()
    # 9. Read in rainfall and subset
    precip_hist_roi = import_hist_precip(ens_clim_start_year, ens_clim_end_year, rfe_hist_dir, lon_min, lon_max, lat_min, lat_max)
    # 10. Read in historical soil moisture estimates
    sm_hist_roi = import_hist_sm(clim_start_year, clim_end_year, sm_hist_dir, lon_min, lon_max, lat_min, lat_max, True)
    # 11. Given supplied weights (probabilites), produce rainfall-based weights (e.g. a weight for each ensemble year)
    precip_weights = weight_forecast(precip_hist_roi, met_forc_start_date, met_forc_end_date, poi_start, ens_clim_start_year, ens_clim_end_year, ds_weights)
    # 12. Compute soil moisture climatology
    sm_hist_full_roi, sm_hist_poi_roi_mean, sm_hist_current_roi, sm_hist_current_roi_mean = calc_sm_climatology(sm_hist_roi, clim_start_year, clim_end_year, fcast_date, poi_start, poi_end)
    #sm_hist_full_roi.sm_c4grass.sel(lon=39, lat=-3, method='nearest', ens_year=2005).plot()
    # 13. Produce summmary fields and output text
    ens_mean_wrsi_xr, ens_sd_wrsi_xr, clim_mean_wrsi_xr, clim_sd_wrsi_xr, ensemble_forecast = summary_stats(sm_hist_poi_roi_mean, precip_weights, sm_poi_roi, sm_full_roi, ens_clim_start_year, ens_clim_end_year)
    forecast_stamp, poi_stamp, poi_str, loc_stamp, currentdate_stamp = date_stamps(fcast_date, poi_start, poi_end, lon_min, lon_max, lat_min, lat_max)
    # 14. Produce API outputs (data and plots)
    output_forecasts(ens_mean_wrsi_xr, ens_sd_wrsi_xr, clim_mean_wrsi_xr, clim_sd_wrsi_xr, ensemble_forecast, sm_recent_roi, sm_hist_full_roi, sm_hist_current_roi_mean, poi_stamp, forecast_stamp, clim_start_year, clim_end_year, poi_start, poi_end, poi_str, fcast_date, loc_stamp, currentdate_stamp)


# Auto-run
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    poi_start_in = args.poi_start
    poi_end_in = args.poi_end
    current_date = args.current_date
    clim_start_year, clim_end_year = args.clim_years
    lat_max, lat_min, lon_min, lon_max = args.coords
    weights = args.weights
    
    """   
    #poi_start_in = '/gws/nopw/j04/tamsat/rmaidment/KMD/T-A_API_KMD/data/kenya_current_lr_sos.nc'
    poi_start_in = dt.strptime('2024-03-01', '%Y-%m-%d').date()
    poi_end_in = dt.strptime('2024-08-31', '%Y-%m-%d').date()
    current_date = dt.strptime('2024-04-10', '%Y-%m-%d').date()
    clim_start_year = 1991
    clim_end_year = 2020
    lon_min = 32.0
    lon_max = 43.0
    lat_min = -5.0
    lat_max = 6.0
    weights = [float(0.33), float(0.34), float(0.33)]
    #weights = 'ECMWF_S2S'
    """
    
    print('==================================================') 
    print('--- Executing the TAMSAT-ALERT API (Version 2) ---')
    print('==================================================') 
    print('Season start: %s' % poi_start_in)
    print('Season end: %s' % poi_end_in)
    print('Current date: %s' % current_date)
    print('Climatological period: %s-%s' % (clim_start_year, clim_end_year))
    print('Domain coordinates: N:%s, S:%s, W:%s, E:%s' % (lat_max, lat_min, lon_min, lon_max))
    if weights == 'ECMWF_S2S':
        print('Tercile weights: Derived from ECMWF S2S precipitation forecasts')
    else: 
        print('Tercile weights: Upper:%s, Middle:%s, Lower:%s' % (weights[0], weights[1], weights[2]))
    
    # Clim years for soil moisture forecasts
    ens_clim_start_year = 2005
    ens_clim_end_year = 2019
    
    # Dataset versions to use
    sm_version = '2.3.0'
    rfe_version = '3.1'
    
    # TAMSAT data URL
    remoteurl = 'https://gws-access.jasmin.ac.uk/public/tamsat/tamsat-alert-api_forcing_data'
    
    # Path of current .py file (all data and outputs will be saved here)
    cwd = os.getcwd()
    inputdir = os.path.join(cwd, 'input_data')
    outputdir = os.path.join(cwd, 'outputs', dt.strftime(current_date, format='%Y-%m-%d') + '_' + dt.strftime(dt.now(), format='%Y-%m-%dT%H:%M:%S'))
    datadir = os.path.join(outputdir, 'data')
    plotsdir = os.path.join(outputdir, 'plots')
    sm_hist_dir = os.path.join(inputdir, 'soil_moisture_historical', 'v' + sm_version)
    sm_fcast_dir = os.path.join(inputdir, 'soil_moisture_forecasts', 'v' + sm_version)
    rfe_hist_dir = os.path.join(inputdir, 'rainfall_historical', 'v' + rfe_version)
    ecmwfs2s_dir = os.path.join(inputdir, 'ecmwfs2s_tercile_forecasts')
    
    # Create directories
    makedir(datadir)
    makedir(plotsdir)
    makedir(sm_hist_dir)
    makedir(sm_fcast_dir)
    makedir(rfe_hist_dir)
    makedir(ecmwfs2s_dir)
    
    # Create API inputs summary file
    create_inputs_summary_csv(os.path.join(outputdir, 'API_input_arguments.csv'))
    
    # Execute API
    wrapper(current_date)
