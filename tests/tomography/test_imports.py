import subprocess
import sys

import pytest


def _modules_after(statement):
    code = f"import sys; {statement}; print(','.join(sorted(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return set(out.stdout.strip().split(","))


def test_package_import_is_light():
    mods = _modules_after("import qqea.tomography")
    for heavy in ("pygsti", "quantify_core", "qqea.tomography.plotting"):
        assert heavy not in mods, heavy


def test_plotting_names_load_on_demand():
    import qqea.tomography as tomo
    assert callable(tomo.plot_process_tomography2)
    assert callable(tomo.qpt_plot_cmap)
    with pytest.raises(AttributeError):
        tomo.not_a_function


def test_star_import_exports_everything():
    import qqea.tomography as tomo
    ns = {}
    exec("from qqea.tomography import *", ns)
    assert set(tomo.__all__) <= set(ns)


def test_legacy_namespace_is_complete(legacy):
    import qqea.tomography.tomography as flat
    removed = {"get_spam_error_superoperator", "file"}
    missing = {n for n in dir(legacy) if not n.startswith("_")} - set(dir(flat)) - removed
    assert not missing
    for n in ("_mle_forward_matrix", "_tomo_rotations", "_project_eigs_to_simplex"):
        assert hasattr(flat, n)
