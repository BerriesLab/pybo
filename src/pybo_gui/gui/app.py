"""Main window for the campaign-analysis GUI.

    python -m pybo_gui.gui.app [data directory]

Owns the tabs and the Steps selector window they read from.
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QStatusBar, QTabWidget

from pybo_gui.gui import tab_campaign, tab_settings
from pybo_gui.gui.launchers import stop_all
from pybo_gui.gui.message_log import show_log
from pybo_gui.gui.settings import Settings
from pybo_gui.gui.step_list import StepListWindow

# Relative to where the GUI was started, not to where pybo is installed: the data being
# analysed belongs to the user's working directory, not to the package. The first of these
# that exists wins; falling back to the working directory itself means the selector always
# opens on something real, and the tree is browsable from there anyway.
DEFAULT_ROOTS = (Path.cwd() / "data", Path.cwd() / "studies" / "data")
DEFAULT_ROOT = next((p for p in DEFAULT_ROOTS if p.is_dir()), Path.cwd())


class MainWindow(QMainWindow):
    def __init__(self, root: str = ""):
        super().__init__()
        self.setWindowTitle("pyBO — campaign analysis")
        self.resize(760, 520)

        self.step_list = StepListWindow(parent=self, initial_root=root)
        # One object shared by every tab: Settings writes to it, the campaign tab reads
        # it when assembling a command line.
        self.settings = Settings()

        tabs = QTabWidget()
        # One call, two pages: assembling a campaign and plotting it are separate tabs but
        # not separate state - see tab_campaign.build.
        constructor, plots = tab_campaign.build(self.step_list, self.settings)
        tabs.addTab(constructor, "Bayesian campaign constructor")
        tabs.addTab(plots, "Plot")
        tabs.addTab(tab_settings.build(self.settings), "Settings")
        self.setCentralWidget(tabs)

        bar = QStatusBar()
        self.setStatusBar(bar)
        # Where every message the tabs produce ends up. Nothing is printed next to the
        # button that produced it any more, so this is the only way to read them.
        log = QPushButton("Log")
        log.setToolTip("Everything the GUI has reported, newest last")
        log.clicked.connect(lambda: show_log(self))
        bar.addPermanentWidget(log)
        stop = QPushButton("Stop plots")
        stop.clicked.connect(stop_all)
        bar.addPermanentWidget(stop)

        self.step_list.scan()
        self.step_list.show()
        # Open from the start rather than on demand: it is the only place a message goes
        # now, so a run that reports something before anyone thinks to open the log would
        # otherwise look like it reported nothing. main() shows the main window after
        # this, which leaves that in front.
        show_log(self)

    def closeEvent(self, event):
        # The selector refuses to close on its own, so release it with us.
        self.step_list.force_close()
        stop_all()
        event.accept()


def main():
    requested = sys.argv[1] if len(sys.argv) > 1 else ""
    if not requested and DEFAULT_ROOT.is_dir():
        requested = str(DEFAULT_ROOT)
    app = QApplication(sys.argv)
    window = MainWindow(root=requested)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
