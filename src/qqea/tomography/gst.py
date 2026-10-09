"""Gate set tomography (GST) dataset helpers: Quantify GST datasets to and from pyGSTi.

pyGSTi is only imported by ``ds_from_dataset_txt``, so the other helpers work without it.
"""
import warnings

import numpy as np


def ds_from_dataset_txt(dataset_txt):
    """Build a pyGSTi DataSet from dataset.txt text (as returned by gst_ds_to_pygsti_dataset_txt)."""
    import pygsti

    lines = [ln.strip() for ln in dataset_txt.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("dataset_txt is empty")

    header = lines[0]
    if not header.startswith("## Columns ="):
        raise ValueError("First line must start with '## Columns ='")

    col_part = header.split("=", 1)[1].strip()
    outcome_labels = [tok.strip().split()[0] for tok in col_part.split(",")]
    if not outcome_labels:
        raise ValueError("No outcome labels found in header")

    ds = pygsti.data.DataSet(outcome_labels=outcome_labels)

    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 1 + len(outcome_labels):
            raise ValueError(
                f"Bad row: expected {1 + len(outcome_labels)} columns, got {len(parts)} in '{ln}'"
            )
        circuit_label = parts[0]
        counts = [int(x) for x in parts[1:]]
        ds.add_count_dict(circuit_label, dict(zip(outcome_labels, counts)))

    ds.done_adding_data()
    return ds


def _gst_ds_counts(gst_ds):
    """(n_circuits, 4) outcome counts in 00, 01, 10, 11 order."""
    y0 = np.rint(np.asarray(gst_ds["y0"].values)).astype(int)
    y1 = np.rint(np.asarray(gst_ds["y1"].values)).astype(int)
    c = np.empty((y0.shape[1], 4), dtype=int)
    c[:, 0] = ((y0 == 0) & (y1 == 0)).sum(axis=0)
    c[:, 1] = ((y0 == 0) & (y1 == 1)).sum(axis=0)
    c[:, 2] = ((y0 == 1) & (y1 == 0)).sum(axis=0)
    c[:, 3] = ((y0 == 1) & (y1 == 1)).sum(axis=0)
    return c


def gst_ds_datasets_homogeneity(ds_list):
    """chi^2/dof for "did these repeat GST runs measure the same device?".

    Repeat runs of one GST experiment can be pooled only if they are i.i.d. samples of the
    same gate set. This is a chi^2 test of homogeneity on the per-circuit outcome counts:
    each circuit contributes (n_runs - 1) * 3 degrees of freedom, and the pooled outcome
    frequencies supply the expected counts.

    Returns (chi2, dof, ratio). ratio ~= 1 means the runs are statistically identical and
    pooling is sound. ratio >> 1 means the device moved between runs -- pooling then averages
    over different gate sets, and since an average of unitaries is not unitary, the drift is
    converted into apparent depolarization: the fit reports a lower fidelity with the loss
    showing up in the non-unitary column, indistinguishable from real decoherence.

    Measured on the five 20260909/10 runs: ratio 2.76, with the pairwise total-variation
    distance growing monotonically with time separation (0.016 adjacent, 0.024 across
    4.4 hours, against a 0.011 shot-noise floor) -- i.e. genuine drift, not scatter.
    """
    C = np.stack([_gst_ds_counts(d) for d in ds_list])       # (n_runs, n_circuits, 4)
    N = C.sum(axis=2)
    tot = C.sum(axis=0)
    p = tot / np.maximum(tot.sum(axis=1, keepdims=True), 1)
    exp = p[None, :, :] * N[:, :, None]
    chi2 = float(np.where(exp > 5, (C - exp) ** 2 / np.maximum(exp, 1e-9), 0.0).sum())
    dof = (len(ds_list) - 1) * 3 * C.shape[1]
    return chi2, dof, chi2 / dof


def concat_gst_datasets(ds_list, check_homogeneity=True, warn_ratio=1.5):
    """Stack repeat GST runs along the shot axis into one dataset.

    Combining happens at the SHOT level, before any counting, so the result feeds
    ``gst_ds_to_pygsti_dataset_txt`` unchanged and the pooled counts are exactly the sum of
    the parts. All runs must share the same circuit list in the same order -- the acq_index
    axis is positional, so a differing experiment_list would silently mix circuits.

    Pooling is a statistics-vs-systematics trade: n runs cut the shot-noise floor by sqrt(n)
    but fold any drift between them into the fit as apparent decoherence. Unless you
    specifically want the time-averaged gate, prefer fitting each run separately and
    averaging the extracted parameters, using the run-to-run spread as the error bar.
    ``check_homogeneity`` issues a UserWarning when the runs fail that test (chi2/dof above
    ``warn_ratio``); see ``gst_ds_datasets_homogeneity``.
    """
    import xarray as xr

    ds_list = list(ds_list)
    if not ds_list:
        raise ValueError("ds_list is empty.")
    shapes = {np.asarray(d["y0"].values).shape[1] for d in ds_list}
    if len(shapes) != 1:
        raise ValueError(f"datasets cover different circuit counts: {sorted(shapes)}")

    if check_homogeneity and len(ds_list) > 1:
        chi2, dof, ratio = gst_ds_datasets_homogeneity(ds_list)
        if ratio > warn_ratio:
            warnings.warn(
                f"these {len(ds_list)} runs are not statistically identical "
                f"(homogeneity chi2/dof = {ratio:.2f}). Pooling will convert the drift "
                f"between them into apparent decoherence. Consider fitting each run "
                f"separately and averaging the parameters instead.",
                stacklevel=2,
            )

    data = {
        v: (("repetition", "acq_index_0"),
            np.concatenate([np.asarray(d[v].values) for d in ds_list], axis=0))
        for v in ("y0", "y1")
    }
    return xr.Dataset(data)


def gst_ds_to_pygsti_dataset_txt(
    gst_ds,
    listOfExperiments,
    mode="auto",
    single_qubit_var="y0",
):
    """Convert a Quantify GST dataset to pyGSTi dataset.txt text.

    - First column is a compact pyGSTi circuit string, e.g. ({})Gxpi2:0Gxpi2:0, keeping any
      trailing line-label annotation such as @(0).
    - mode: "1q" counts outcomes of ``single_qubit_var``; "2q" counts the joint y0/y1
      outcomes 00, 01, 10, 11; "auto" picks "2q" when y0 and y1 are both present.
    - No file is written; this function only returns dataset_txt.

    Example usage
    dataset_txt = gst_ds_to_pygsti_dataset_txt(gst_ds, listOfExperiments, mode="auto")
    print("\\n".join(dataset_txt.splitlines()[:6]))
    """
    data_vars = list(gst_ds.data_vars)
    data_var_set = set(data_vars)

    if mode == "auto":
        mode = "2q" if {"y0", "y1"}.issubset(data_var_set) else "1q"

    n_exp = len(listOfExperiments)

    def _extract_shots(var_name, exp_idx):
        if var_name not in gst_ds.data_vars:
            raise KeyError(f"'{var_name}' not found in gst_ds.data_vars: {list(gst_ds.data_vars)}")
        arr = np.asarray(gst_ds[var_name].values)
        if arr.ndim == 0:
            raise ValueError(f"{var_name} is scalar; expected shot/experiment dimensions.")
        if arr.ndim == 1:
            if n_exp != 1:
                raise ValueError(
                    f"{var_name} is 1D but listOfExperiments has {n_exp} experiments."
                )
            return arr
        if arr.shape[1] == n_exp:
            return arr[:, exp_idx]
        if arr.shape[0] == n_exp:
            return arr[exp_idx, :]
        raise ValueError(
            f"Cannot align {var_name} shape {arr.shape} with {n_exp} experiments."
        )

    def _circuit_to_dataset_label(circuit):
        # pyGSTi's compact "Circuit(({})Gxpi2:0...@(0))" repr, without the Circuit( ) wrapper
        s = repr(circuit)
        if s.startswith("Circuit(") and s.endswith(")"):
            s = s[len("Circuit("):-1]
        return s

    if mode == "1q":
        if single_qubit_var not in data_var_set:
            if len(data_vars) == 1:
                single_qubit_var = data_vars[0]
            else:
                raise KeyError(
                    f"single_qubit_var '{single_qubit_var}' missing; available vars: {data_vars}"
                )
        header = "## Columns = 0 count, 1 count"
    elif mode == "2q":
        if not {"y0", "y1"}.issubset(data_var_set):
            raise KeyError(f"2q mode requires y0 and y1; available vars: {data_vars}")
        header = "## Columns = 00 count, 01 count, 10 count, 11 count"
    else:
        raise ValueError("mode must be 'auto', '1q', or '2q'.")

    lines = [header]
    for exp_idx, circuit in enumerate(listOfExperiments):
        cstr = _circuit_to_dataset_label(circuit)

        if mode == "1q":
            y = _extract_shots(single_qubit_var, exp_idx)
            y = np.rint(np.asarray(y)).astype(int).ravel()
            if not np.isin(y, [0, 1]).all():
                raise ValueError(f"{single_qubit_var} contains non-binary values for experiment {exp_idx}.")
            c1 = int(np.sum(y == 1))
            c0 = int(np.sum(y == 0))
            lines.append(f"{cstr} {c0} {c1}")
            continue

        y0 = _extract_shots("y0", exp_idx)
        y1 = _extract_shots("y1", exp_idx)
        y0 = np.rint(np.asarray(y0)).astype(int).ravel()
        y1 = np.rint(np.asarray(y1)).astype(int).ravel()
        nshots = min(y0.size, y1.size)
        if nshots == 0:
            raise ValueError(f"No shots found for experiment {exp_idx}.")
        y0 = y0[:nshots]
        y1 = y1[:nshots]
        if (not np.isin(y0, [0, 1]).all()) or (not np.isin(y1, [0, 1]).all()):
            raise ValueError(f"y0/y1 contain non-binary values for experiment {exp_idx}.")

        c00 = int(np.sum((y0 == 0) & (y1 == 0)))
        c01 = int(np.sum((y0 == 0) & (y1 == 1)))
        c10 = int(np.sum((y0 == 1) & (y1 == 0)))
        c11 = int(np.sum((y0 == 1) & (y1 == 1)))
        lines.append(f"{cstr} {c00} {c01} {c10} {c11}")

    return "\n".join(lines)
