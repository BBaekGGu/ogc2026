"""alg_tester entry point.

Usage:
    conda activate ogc2026
    python alg_tester_app.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from alg_tester_ui.main_window import MainWindow

_SETTINGS_FILE = pathlib.Path(__file__).parent / "settings.json"


def main():
    # Retina / high-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("OGC2026 Algorithm Tester")
    app.setOrganizationName("OGC2026")

    win = MainWindow(_SETTINGS_FILE)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
