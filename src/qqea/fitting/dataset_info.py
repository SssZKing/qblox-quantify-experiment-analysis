"""Metadata read from quantify datasets and snapshots (timing, source settings)."""

from datetime import datetime

from quantify_core.data.handling import load_snapshot


def experiment_duration(tuid_arr):
    # Define the format
    time_format = '%Y%m%d-%H%M%S'
    
    # Parse the strings into datetime objects
    dt1 = datetime.strptime(tuid_arr[0], time_format)
    dt2 = datetime.strptime(tuid_arr[1], time_format)
    dt3 = datetime.strptime(tuid_arr[-1], time_format)

    dt_mins = round((dt2 - dt1).total_seconds()/60, 2)
    T_hours = round((dt3 - dt1).total_seconds()/3600, 2)

    print("Dataset mins:", dt_mins)
    print("Total hours:", T_hours)

    return dt_mins, T_hours


def dataset_current(tuid, source = 'GS210_1'):
    if not isinstance(tuid, str):
        tuid = tuid.tuid
    snapshot = load_snapshot(tuid)
    return round(snapshot['instruments'][source]['parameters']['output_level']['value']*1e3, 3)

def dataset_power(tuid, source = 'SGS100A_1'):
    if not isinstance(tuid, str):
        tuid = tuid.tuid
    snapshot = load_snapshot(tuid)
    return round(snapshot['instruments'][source]['parameters']['power']['value'], 3)
