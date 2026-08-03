"""Main window for the campaign-analysis GUI.

    python -m pybo_gui.gui.app [data directory]

Owns the tab and the Steps selector window it reads from. One tab for now; the
QTabWidget is kept so a second one costs an addTab.
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QStatusBar, QTabWidget

from pybo_gui.gui import tab_campaign, tab_settings
from pybo_gui.gui.launchers import stop_all
from pybo_gui.gui.settings import Settings
from pybo_gui.gui.step_list import StepListWindow

# Relative to where the GUI was started, not to where pybo is installed: the data being
# analysed belongs to the user's working directory, not to the package.
DEFAULT_ROOT = Path.cwd() / "studies" / "data"


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
        tabs.addTab(tab_campaign.build(self.step_list, self.settings),
                    "Bayesian campaign analysis")
        tabs.addTab(tab_settings.build(self.settings), "Settings")
        self.setCentralWidget(tabs)

        bar = QStatusBar()
        self.setStatusBar(bar)
        stop = QPushButton("Stop plots")
        stop.clicked.connect(stop_all)
        bar.addPermanentWidget(stop)

        self.step_list.scan()
        self.step_list.show()

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
