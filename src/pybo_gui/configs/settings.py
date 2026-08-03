"""Where the campaign being analysed lives.

Every plot script reads `data_path` for two things: the experiment_map.json it loads, and
the directory a figure would be written to. The GUI rewrites it when the selection
changes, so a plot launched afterwards reads the campaign the user is looking at rather
than the one loaded at startup.
"""
from pathlib import Path

# Absolute path to the campaign directory. Defaults to the working directory so a plot run
# straight from a terminal reads the map sitting there.
data_path = r"C:\Users\BerettaDavide\AppData\Local\Temp\pybo_campaign_ja4zu31o"


def set_data_path(path) -> None:
    """Point at `path`, in memory and in this file.

    Written back because a plot runs as its own process and re-imports this module: an
    in-memory change alone would be invisible to it.
    """
    global data_path
    data_path = str(Path(path).resolve())
    source = Path(__file__)
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("data_path = "):
            lines[i] = f'data_path = r"{data_path}"\n'
            break
    source.write_text("".join(lines), encoding="utf-8")
