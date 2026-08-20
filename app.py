import logging
import os
import sys
import warnings
from pathlib import Path
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from src.gui.i18n import install_live_translation
from src.gui.main import MainWindow


def _set_application_directory():
    """Keep all relative storage paths stable in source and frozen Windows builds."""
    root = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    os.chdir(root)


def main():
    _set_application_directory()
    # Initializing app window.
    app = QApplication(sys.argv)

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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    warnings.filterwarnings('ignore')
    main()
