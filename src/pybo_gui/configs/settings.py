"""Where the campaign being analysed lives.

Every plot script reads `data_path` for two things: the experiment_map.json it loads, and
the directory a figure would be written to.

The value comes from the environment, because a plot runs as its own process: the GUI sets
the variable and every plot it launches afterwards inherits it. Nothing is written back to
this file, so choosing a campaign leaves no diff behind - and a script run straight from a
terminal can be pointed at a saved map the same way:

    PYBO_CAMPAIGN_DIR=/path/to/saved/map python -m pybo_gui.modules...plot_pareto_2d --x ...
"""
import os
from pathlib import Path

ENV_VAR = "PYBO_CAMPAIGN_DIR"

# Absolute path to the campaign directory. Defaults to the working directory, so a plot run
# with nothing set reads the map sitting there.
data_path = os.environ.get(ENV_VAR) or str(Path.cwd())


def set_data_path(path) -> None:
    """Point at `path`, for this process and for anything it launches next."""
    global data_path
    data_path = str(Path(path).resolve())
    os.environ[ENV_VAR] = data_path
