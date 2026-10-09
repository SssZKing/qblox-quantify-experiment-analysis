import warnings

import numpy as np
import pytest
import xarray as xr

from qqea.tomography import (
    concat_gst_datasets, ds_from_dataset_txt, gst_ds_datasets_homogeneity, gst_ds_to_pygsti_dataset_txt,
)


class FakeCircuit:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return f"Circuit({self.s})"


CIRCUITS = [FakeCircuit("{}@(0,1)"), FakeCircuit("Gxpi2:0@(0,1)"), FakeCircuit("Giswap:0:1@(0,1)")]


def _gst_ds(rng, p1=(0.1, 0.5, 0.9), n_shots=400):
    dims = ("repetition", "acq_index_0")
    p1 = np.asarray(p1)
    return xr.Dataset({
        "y0": (dims, (rng.uniform(size=(n_shots, len(p1))) < p1).astype(float)),
        "y1": (dims, (rng.uniform(size=(n_shots, len(p1))) < p1[::-1]).astype(float)),
    })


def test_dataset_txt_matches_legacy(legacy):
    ds = _gst_ds(np.random.default_rng(9))
    txt = gst_ds_to_pygsti_dataset_txt(ds, CIRCUITS)
    assert txt == legacy.gst_ds_to_pygsti_dataset_txt(ds, CIRCUITS)
    lines = txt.splitlines()
    assert lines[0] == "## Columns = 00 count, 01 count, 10 count, 11 count"
    assert lines[2].startswith("Gxpi2:0@(0,1) ")
    assert all(sum(map(int, ln.split()[1:])) == 400 for ln in lines[1:])

    txt_1q = gst_ds_to_pygsti_dataset_txt(ds, CIRCUITS, mode="1q")
    assert txt_1q == legacy.gst_ds_to_pygsti_dataset_txt(ds, CIRCUITS, mode="1q")


def test_concat_and_homogeneity(legacy):
    rng = np.random.default_rng(10)
    p1 = np.linspace(0.05, 0.95, 30)            # enough circuits for chi2/dof to be ~1 +- 0.1
    same = [_gst_ds(rng, p1) for _ in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # identical devices: no warning
        pooled = concat_gst_datasets(same)
    assert pooled["y0"].shape == (1200, 30)
    assert gst_ds_datasets_homogeneity(same) == pytest.approx(legacy.gst_ds_datasets_homogeneity(same))
    assert gst_ds_datasets_homogeneity(same)[2] < 1.5

    drifted = [_gst_ds(rng, p1), _gst_ds(rng, np.clip(p1 + 0.2, 0, 1))]
    with pytest.warns(UserWarning, match="not statistically identical"):
        concat_gst_datasets(drifted)


def test_ds_from_dataset_txt_round_trip():
    pytest.importorskip("pygsti")
    ds = _gst_ds(np.random.default_rng(11))
    pds = ds_from_dataset_txt(gst_ds_to_pygsti_dataset_txt(ds, CIRCUITS))
    assert len(pds) == len(CIRCUITS)
