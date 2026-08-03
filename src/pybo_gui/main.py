"""PySide6 entry point for the campaign-analysis GUI.

    python -m pybo_gui.main [data directory]

Adds the application stylesheet around gui.app.MainWindow, which is also
runnable on its own for a plain-Qt look.
"""
import multiprocessing
import sys

from PySide6.QtWidgets import QApplication

from pybo_gui.gui.app import DEFAULT_ROOT, MainWindow


if __name__ == "__main__":
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QTabWidget::pane {
            background: palette(window);
            border: 1px solid palette(mid);
            border-top: none;
        }
        QTabWidget > QStackedWidget > QWidget {
            background: palette(window);
        }
        QTabBar::tab {
            background: #e1e1e1;
            border: 1px solid palette(mid);
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 4px 10px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: palette(window);
            margin-bottom: -1px;
        }
        QTabBar::tab:hover:!selected {
            background: #d8e6f6;
        }
        QPushButton {
            background-color: #f0f0f0;
            border: 1px solid #adadad;
            border-radius: 4px;
            padding: 2px 8px;
        }
        QPushButton:hover {
            background-color: #d8e6f6;
            border: 1px solid #6e9bd1;
        }
        QPushButton:pressed {
            background-color: #b8d0ed;
            border: 1px solid #4b7fbf;
        }
        QPushButton:disabled {
            background-color: #f8f8f8;
            color: #a0a0a0;
            border: 1px solid #d0d0d0;
        }
        QCheckBox {
            padding: 2px 4px;
            border-radius: 4px;
        }
        QCheckBox:hover {
            background-color: #d8e6f6;
        }
        QRadioButton {
            padding: 2px 4px;
            border-radius: 4px;
        }
        QRadioButton:hover {
            background-color: #d8e6f6;
        }
    """)
    requested = sys.argv[1] if len(sys.argv) > 1 else ""
    if not requested and DEFAULT_ROOT.is_dir():
        requested = str(DEFAULT_ROOT)
    window = MainWindow(root=requested)
    window.show()
    sys.exit(app.exec())
