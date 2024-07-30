import numpy as np
import pandas as pd
import netCDF4 as nc


def palm_times(palmfile: nc.Dataset) -> list:
    """
    Extracts the time vector, formatting it as a datetime. The time contained within the PALM file is given as
    minutes since origin. Additionally, a boolean vector is generated, indicating the start of the useable time
    series (certain observations are required to create the moving average).
    """
    origintime = pd.to_datetime(palmfile.origin_time)
    times_list = palmfile['time']
    times = []
    for _, time in enumerate(times_list):
        times.append(origintime + pd.Timedelta(minutes=np.round(time * 24 * 60)))
    if not times:
        raise ValueError
    return times


def empty_array(palmdim):
    """
    Generates an empty array of the same dimensions as the PALM file.
    """
    empty_array = np.empty((palmdim[0], palmdim[1], palmdim[2]), dtype=np.float32)
    #! NaN is used as a placeholder for missing data - alternative is 0 or -9999
    empty_array[:] = np.nan
    return empty_array


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
    info['CH_N'], info['CH_E'], info['CH_S'], info['CH_W'] = coordinates(palmfile)
    return info


def stations_loc(boundary, stationinfo):
    stationscsv = pd.read_csv(stationinfo, delimiter=';')
    stationsloc = {}
    stations = stationscsv['stationid_new'].unique()
    for station in stations:
        row = stationscsv[stationscsv['stationid_new'] == station]
        if not row.empty:
            if boundary['CH_W'] <= int(row["CH_E"]) <= boundary['CH_E'] and boundary['CH_S'] <= int(row["CH_N"]) <= boundary['CH_N']:
                stationsloc[station] = {'lat': int(row["CH_N"]), 'lon': int(row["CH_E"]), 
                                        'lat_idx': int((boundary['CH_N'] - row['CH_N']) / 16), 
                                        'lon_idx': int((row['CH_E'] - boundary['CH_W']) / 16)}
    return stationsloc


def extract_measurements(measurementpath, stationid, times):
    try:
        measurementfile= pd.read_csv(f'{measurementpath}/temp_{stationid}.csv', delimiter=';')
    except FileNotFoundError:
        print(f'File not found: {measurementpath}/temp_{stationid}.csv')
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


def fill_maps(mapshape, stationsloc, measurementpath, times):
    temparray = empty_array(mapshape)
    for station in stationsloc.keys():
        temps = extract_measurements(measurementpath, station, times)
        if not np.isnan(temps).all():
            try:
                lat_idx = stationsloc[station]['lat_idx']
                lon_idx = stationsloc[station]['lon_idx']
                temparray[:, lat_idx, lon_idx] = temps
            except IndexError as e:
                print(f'Station {station} is outside of the reduced boundary ({lat_idx},{lon_idx}) vs. {mapshape})')
        else:
            print(f'Station {station} has no measurements')
    temparray = np.flip(temparray, axis=1)
    return temparray


def highres_maps(palmpath, measurementpath, stationinfo, times, shape) -> np.ndarray:
    """
    Generates a high resolution map of the PALM file, using only measurements from the stations within the boundary.
    """
    palmfile = nc.Dataset(palmpath)
    boundary = palm_info(palmfile)
    datetimes = palm_times(palmfile)
    stationsloc = stations_loc(boundary, stationinfo)
    hr_filled_map = fill_maps(shape[:-1], stationsloc, measurementpath, times)
    return hr_filled_map