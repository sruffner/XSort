import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import QApplication, QMainWindow

from xsort.views.manager import ViewManager

# TODO: Python path hack. I cannot figure out how to organize program structure so I can run app.py from PyCharm IDE
#  and also package Xsort for distribution and running the program via python -m xsort.app. I hate this.
p = Path(__file__).parent
sys.path.append(str(p.absolute()))


class XSortMainWindow(QMainWindow):
    """
    The main application window. The XSort UI is built and controlled by the ViewManager singleton. This is merely the
    container for the UI.
    """

    def __init__(self):
        super().__init__()
        self._starting_up = True
        """ Flag set at startup so that all views will be updated to reflect contents of current working directory. """
        self._view_manager = ViewManager(self)
        """ The model-view controller. Constructs the UI and the data model and hooks them together. """

    def closeEvent(self, event: QCloseEvent) -> None:
        """ [QWidget override] Closing the main window quits the application -- unless the user vetoes the quit. """
        self._view_manager.quit()
        event.ignore()

    def showEvent(self, event: QShowEvent) -> None:
        """
        [QWidget override] When the main window is shown at startup, force user to select a working directory if one
        was not already restored from user preferences during initializations.
        """
        if self._starting_up and not self._view_manager.data_analyzer.is_valid_working_directory:
            QTimer.singleShot(10, self._select_working_directory_at_startup)
        self._starting_up = False

    def _select_working_directory_at_startup(self) -> None:
        self._view_manager.select_working_directory(starting_up=True)


if __name__ == "__main__":
    main_app = QApplication(sys.argv)
    main_window = XSortMainWindow()
    main_window.show()
    exit_code = main_app.exec()
    # Any after-exit tasks can go here (should not take too long!)
    sys.exit(exit_code)
