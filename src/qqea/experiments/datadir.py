"""Point quantify's data directory at ``QBLOX_DATADIR``."""

import os
from pathlib import Path

ENV_VAR = "QBLOX_DATADIR"


def set_datadir_from_env(env_var: str = ENV_VAR) -> Path:
    """Call quantify's ``set_datadir`` with the folder named by ``env_var``.

    Returns the data directory. Raises ``RuntimeError`` if the variable is
    unset or empty, and ``FileNotFoundError`` if the folder does not exist
    (quantify would otherwise create an empty one silently).
    """
    from quantify_core.data.handling import set_datadir

    value = os.environ.get(env_var, "").strip()
    if not value:
        raise RuntimeError(
            f"{env_var} is not set. Set it to the quantify data folder for this "
            f"session, e.g. `set {env_var}=C:\\path\\to\\data` in cmd before starting Jupyter."
        )
    datadir = Path(value).expanduser()
    if not datadir.is_dir():
        raise FileNotFoundError(f"{env_var}={value} is not an existing folder.")
    set_datadir(str(datadir))
    return datadir
