"""Figure settings package: generic defaults + publisher styles + the fig_cfg assembler.

- ``store``  — read/write the YAML files and active selections (matplotlib-free; safe
  to import from the GUI). Its API is re-exported here for convenience. The host app's
  domain settings, user styles and state live under ``store.APP_DIR`` (the seam).
- ``config`` — assembles ``fig_cfg`` (package defaults -> app defaults -> publisher ->
  user) and applies rcParams (imports matplotlib; imported by the plotters as
  ``from pybo.plotters.figure_settings.config import fig_cfg``).
"""
from pybo.plotters.figure_settings.store import (  # noqa: F401
    APP_DEFAULTS_PATH,
    APP_DIR,
    PACKAGE_DEFAULTS_PATH,
    PUBLISHER_DIR,
    STATE_PATH,
    USER_DIR,
    USER_STYLE_TEMPLATE,
    app_defaults_text,
    delete_user_style,
    get_active,
    list_publisher_styles,
    list_user_styles,
    load_active_styles,
    load_app_defaults,
    load_package_defaults,
    load_style,
    save_user_style,
    set_active_publisher,
    set_active_user,
    style_text,
    validate,
    write_app_defaults,
    write_style,
)
