"""Save a raw ``retrieve_acquisition()`` dataset as a quantify experiment folder.

This is data handling rather than fitting; it is due to move to ``qqea.experiments``.
``qqea.fitting.fits`` keeps re-exporting it so old imports keep working.
"""

import json
import re
from pathlib import Path

import numpy as np
from quantify_core.data import handling as dh


# Helpers: sanitize dataset for netCDF write (for retrieve_acquisition dataset)
# -----------------------------------------------------------------------------
_NETCDF_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _netcdf_safe_name(name: str) -> bool:
    return bool(_NETCDF_SAFE.match(name))

def _netcdf_safe_attr_value(v):
    """
    Coerce attrs to netCDF-safe scalar/list-ish types.
    """
    if v is None:
        return "None"
    if isinstance(v, (str, int, float, bool, bytes)):
        return v
    if isinstance(v, np.number):
        return v.item()
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (list, tuple)):
        return type(v)(_netcdf_safe_attr_value(x) for x in v)
    if isinstance(v, dict):
        # netCDF attrs cannot be dicts; store as JSON string
        return json.dumps(
            {str(k): _netcdf_safe_attr_value(val) for k, val in v.items()},
            default=str,
        )
    return str(v)

def _sanitize_dataset_for_netcdf(ds):
    """
    Make an xarray.Dataset safe for dh.write_dataset():
      - ensure variable/coord names are strings and netCDF-safe
        (e.g. rename integer keys 0, 1, ... to 'acq_ch0', 'acq_ch1', ...)
      - ensure attrs are JSON / netCDF serializable
    """
    ds2 = ds.copy()

    # Rename data variables with unsafe names (e.g. integer 0 -> "y0")
    rename_map = {}
    for k in list(ds2.data_vars.keys()):
        k_str = str(k)
        if not isinstance(k, str) or not _netcdf_safe_name(k_str):
            rename_map[k] = f"y{k_str}"

    # Rename coords if needed as well
    for k in list(ds2.coords.keys()):
        k_str = str(k)
        if not isinstance(k, str) or not _netcdf_safe_name(k_str):
            base = f"x{k_str}"
            new_name = base
            i = 0
            while new_name in ds2.coords or new_name in ds2.data_vars:
                i += 1
                new_name = f"{base}_{i}"
            rename_map[k] = new_name

    if rename_map:
        ds2 = ds2.rename(rename_map)

    # Sanitize dataset attrs
    ds2.attrs = {str(k): _netcdf_safe_attr_value(v) for k, v in ds2.attrs.items()}

    # Sanitize coord and data_var attrs
    for cname in list(ds2.coords):
        ds2.coords[cname].attrs = {
            str(k): _netcdf_safe_attr_value(v)
            for k, v in ds2.coords[cname].attrs.items()
        }
    for vname in list(ds2.data_vars):
        ds2.data_vars[vname].attrs = {
            str(k): _netcdf_safe_attr_value(v)
            for k, v in ds2.data_vars[vname].attrs.items()
        }

    return ds2

def save_retrieve_acquisition_dataset(ds, name: str) -> str:
    """
    Save an xarray.Dataset returned by InstrumentCoordinator.retrieve_acquisition()
    into a Quantify-style experiment folder:

      <datadir>/<YYYY-mm-dd>/<tuid>-<name>/
        dataset.hdf5 (or dataset.nc, depending on quantify-core)
        snapshot.json

    Returns
    -------
    tuid : str
        The unique identifier of the experiment.
    """
    tuid = dh.gen_tuid()
    exp_dir = Path(dh.create_exp_folder(tuid=tuid, name=name))

    ds_to_write = ds.copy()
    ds_to_write.attrs["tuid"] = tuid
    ds_to_write.attrs["name"] = name

    # Sanitize for netCDF (handles integer keys etc.)
    ds_to_write = _sanitize_dataset_for_netcdf(ds_to_write)

    dataset_name = getattr(dh, "DATASET_NAME", "dataset.hdf5")
    dh.write_dataset(exp_dir / dataset_name, ds_to_write)

    # Store a snapshot of the current instrument state,
    # similar to MeasurementControl.run()
    with open(exp_dir / "snapshot.json", "w", encoding="utf-8") as f:
        json.dump(dh.snapshot(), f, indent=2, default=str)

    return tuid
