"""Figure settings package: generic defaults + publisher styles + the fig_cfg assembler.

- ``store``  — read/write the YAML files (matplotlib-free; safe to import from a GUI).
  Its API is re-exported here for convenience. The host app's domain settings live in
  ``store.APP_DIR`` (the seam), which for pyBO is the plotters package itself.
- ``config`` — assembles ``fig_cfg`` (package defaults -> app defaults -> publisher) and
  applies rcParams (imports matplotlib; imported by the plotters as
  ``from pybo.plotters.figure_settings.config import fig_cfg``).
"""
from pybo.plotters.figure_settings.store import (  # noqa: F401
    APP_DEFAULTS_PATH,
    APP_DIR,
    PACKAGE_DEFAULTS_PATH,
    PUBLISHER_DIR,
    app_defaults_text,
    list_publisher_styles,
    load_app_defaults,
    load_package_defaults,
    load_style,
    style_text,
    validate,
    write_app_defaults,
    write_style,
)
