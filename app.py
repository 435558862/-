import logging
import os
import sys
import warnings
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from src.gui.i18n import install_live_translation
from src.gui.main import MainWindow
from src.diagnostics import setup_runtime_diagnostics
from src.version import PRODUCT_NAME, __version__


def _set_application_directory():
    """Keep all relative storage paths stable in source and frozen Windows builds."""
    root = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    os.chdir(root)


def main():
    _set_application_directory()
    setup_runtime_diagnostics(Path.cwd())
    # Initializing app window.
    app = QApplication(sys.argv)
    app.setApplicationName(PRODUCT_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(PRODUCT_NAME)

    # Create app window.
    window = MainWindow(app=app)
    window.show()
    window.raise_()
    window.activateWindow()
    app.setActiveWindow(window)
    translation_timer = install_live_translation(app)

    # Python-launched Qt windows can open behind other apps on macOS.
    QTimer.singleShot(500, window.raise_)
    QTimer.singleShot(600, window.activateWindow)

    # Initialize the event loop.
    app.exec()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
