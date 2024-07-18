import numpy as np
import pandas as pd
import netCDF4 as nc
import gan_utils

faulty_stations = ['C059A2225266', 'D63DFE9B164B', 'D07769DF208C', 'DF15D23E4B15', 'E2A0DF1A4941', 'E437CB2AF225', 'F5C16A4B6340',
                   'F033A8C6BB79', 'F4683D808CFB', 'D3FE8EEF188C', 'D883D89E6A24', 'EC032D8260EB', 'C3FD36A6C1BC', 'D083B9FD07FB', 
                   'EB90524D4F3E', 'FCBBD3B1DB2C']



def generate_GAN_dataset(qrfpredictions, palmfile, measurementpath, savepath, scaling_factor):
    '''
    Generates HR and LR images for the SR-GAN model. The HR images are generated from the PALM simulation, while the LR images are
    generated using the measurement data.

    PROCESS:
    1. HR images
    2. LR images
         2.1. Extract stations within the PALM boundary
         2.2. Extract measurements for each station
         2.3. Generate LR images with measurements
    3. Generate tfrecords with HR and LR arrays
    '''

    hr_array = load_hr(qrfpredictions)
    lr_array = load_lr(palmfile, measurementpath, hr_array.shape, scaling_factor)
    tfrecordpath = 


def load_hr(path):
    hr = np.load(path)
    hr = hr[:, :, :, 1]
    return hr


def load_lr(palmfile, measurementpath, hr_shape, scaling_factor):
    info = palm_data(palmfile)
    stationsdata = stations_loc(info)
    station_temps = extract_station_data(stationsdata, info['times'], measurementpath)
    lr = fill_lr_map(hr_shape, scaling_factor, station_temps)
    return lr


def palm_data(palmpath):
    file = nc.Dataset(palmpath)
    palminfo = palm_info(file)
    return palminfo


def fill_lr_map(hrshape, sf, stationsdata):
    lr = np.zeros((hrshape[0], hrshape[1] // sf, hrshape[2] // sf))
    stationsdata = transform_idxs(stationsdata, sf)
    for station in stationsdata.keys():
        lr[stationsdata[station]['lat_idx_lr'], stationsdata[station]['lon_idx_lr']] = stationsdata[station]['temps']
    return lr


def transform_idxs(stationsdata, sf):
    for station in stationsdata.keys():
        stationsdata[station]['lat_idx_lr'] = stationsdata[station]['lat_idx'] // sf
        stationsdata[station]['lon_idx_lr'] = stationsdata[station]['lon_idx'] // sf
    return stationsdata


def extract_times(origintime: np.datetime64, times_list: list):
    """
    Extracts the time vector, formatting it as a datetime. The time contained within the PALM file is given as
    minutes since origin. Additionally, a boolean vector is generated, indicating the start of the useable time
    series (certain observations are required to create the moving average).
    """
    times = []
    for _, time in enumerate(times_list):
        times.append(origintime + pd.Timedelta(minutes=np.round(time * 24 * 60)))

    #* lost observations fixed, for most PALM files it is 2/56 which are lost
    # lost observations for the one hour moving average: 60 min / timedelta
    td_minutes = (times[2]-times[1]).total_seconds() / 60
    lost_obs = int(60/td_minutes)
    t_bool = [True] * len(times)
    t_bool[0:lost_obs] = [False] * lost_obs
    if not times:
        raise ValueError
    return times


def lv03_to_lv95(lv03_lat: float, lv03_lon: float):
    return lv03_lat + 1000000, lv03_lon + 2000000


def coordinates(palmfile, res=16):
    CH_S, CH_W = lv03_to_lv95(palmfile.origin_y, palmfile.origin_x)
    # CH_S, CH_W, _ = wgs84_to_lv(palmfile.origin_lat, palmfile.origin_lon, 'lv95') #type: ignore
    CH_N = CH_S + palmfile.dimensions['y'].size * res
    CH_E = CH_W + palmfile.dimensions['x'].size * res
    return CH_N, CH_E, CH_S, CH_W


def palm_info(palmfile):
    info = {}
    info['times'] = extract_times(pd.to_datetime(palmfile.origin_time), palmfile['time'])
    info['CH_N'], info['CH_E'], info['CH_S'], info['CH_W'] = coordinates(palmfile)
    return info


def stations_loc(boundary):
    stationscsv = pd.read_csv('S:/pools/t/T-IDP-Projekte-u-Vorlesungen/Meteoblue/Data/Messdaten/stations_new.csv', delimiter=';')
    stationsloc = {}
    print(f'Boundary: N{boundary["CH_N"]} - E{boundary["CH_E"]}')
    stations = stationscsv['stationid_new'].unique()
    for station in stations:
        row = stationscsv[stationscsv['stationid_new'] == station]
        if not row.empty:
            if boundary['CH_W'] <= int(row["CH_E"]) <= boundary['CH_E'] and \
               boundary['CH_S'] <= int(row["CH_N"]) <= boundary['CH_N']:
                stationsloc[station] = {'lat': int(row["CH_N"]), 'lon': int(row["CH_E"]), 
                                        'lat_idx': int((boundary['CH_N'] - row['CH_N']) / 16), 
                                        'lon_idx': int((row['CH_E'] - boundary['CH_W']) / 16)}
    return stationsloc

def extract_measurements(measurementpath, stationid, times):
    try:
        measurementfile= pd.read_csv(f'{measurementpath}/temp_{stationid}.csv', delimiter=';')
    except FileNotFoundError:
        return [np.nan] * len(times)
    
    true_temps = []
    measurementfile['datetime_round'] = pd.to_datetime(measurementfile['datetime']).dt.round('30min')
    times = pd.to_datetime(times)
    for t in times:
        t = t.tz_localize(None)
        try:
            temp = np.mean(measurementfile[measurementfile['datetime_round'] == t]['temp'])
            true_temps.append(temp)
        except IndexError as e:
            print(measurementfile['datetime'])
            print(t)
            raise e
                
    return true_temps


def extract_station_data(stationsdata, times, measurementpath):
    '''
    Extracts the temperature data for each station within the PALM boundary and extends the stationsloc dictionary.
    '''
    for station in stationsdata.keys():
        if station not in faulty_stations:
            station_temps = extract_measurements(measurementpath, station, times)
            stationsdata[station]['temps'] = station_temps
    return stationsdata

