import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCloseEvent, QAction
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QMessageBox, QDockWidget

# TODO: Python path hack. I cannot figure out how to organize program structure so I can run app.py from PyCharm IDE
#  and also package Xsort for distribution and running the program via python -m xsort.app. I hate this.
p = Path(__file__).parent
sys.path.append(str(p.absolute()))

from baseview import BaseView

from constants import APP_NAME


class XSortMainWindow(QMainWindow):
    """
    TODO: UNDER DEV.
    """

    def __init__(self, app: QApplication):
        super().__init__()
        self._app = app
        """ The application instance. """

        # views -- all in dockable and closable except the "Neurons" view, which serves as the central widget
        self._neurons_view = BaseView(0, "Neurons", QColor('red'))
        self._similarity_view = BaseView(1, "Similarity", QColor('blue'))
        self._templates_view = BaseView(2, "Templates", QColor('green'))
        self._statistics_view = BaseView(3, "Statistics", QColor('yellow'))
        self._pca_view = BaseView(4, "PCA", QColor('cyan'))
        self._channels_view = BaseView(5, "Channels", QColor('black'))
        self._umap_view = BaseView(6, "UMAP", QColor('magenta'))

        self.setCentralWidget(self._neurons_view)

        # actions
        self._open_action: Optional[QAction] = None
        self._quit_action: Optional[QAction] = None
        self._about_action: Optional[QAction] = None
        self._about_qt_action: Optional[QAction] = None
        self.create_actions()

        # menus
        self._file_menu: Optional[QMenu] = None
        self._view_menu: Optional[QMenu] = None
        self._help_menu: Optional[QMenu] = None
        self.create_menus()

        self.create_tool_bars()
        self.create_status_bar()
        self.install_docked_views()

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(800, 600)
        self._restore_layout_from_settings()

    def create_actions(self):
        self._open_action = QAction('&Open', self, shortcut="Ctrl+O", statusTip="Open a different folder",
                                    triggered=self.open)
        self._quit_action = QAction("&Quit", self, shortcut="Ctrl+Q", statusTip=f"Quit {APP_NAME}",
                                    triggered=self.quit)

        self._about_action = QAction("&About", self, statusTip=f"About {APP_NAME}", triggered=self.about)

        self._about_qt_action = QAction("About &Qt", self, statusTip="About the Qt library",
                                        triggered=self._app.aboutQt)

    def create_menus(self):
        self._file_menu = self.menuBar().addMenu("&File")
        self._file_menu.addAction(self._open_action)
        self._file_menu.addSeparator()
        self._file_menu.addAction(self._quit_action)

        self._view_menu = self.menuBar().addMenu("&View")
        # populated in create_dock()

        self.menuBar().addSeparator()

        self._help_menu = self.menuBar().addMenu("&Help")
        self._help_menu.addAction(self._about_action)
        self._help_menu.addAction(self._about_qt_action)

    def create_tool_bars(self):
        pass

    def create_status_bar(self):
        self.statusBar().showMessage("Ready")

    def install_docked_views(self):
        for v in [self._similarity_view, self._templates_view, self._statistics_view, self._pca_view,
                  self._channels_view, self._umap_view]:
            dock = QDockWidget(v.title, self)
            dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
            dock.setWidget(v)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            self._view_menu.addAction(dock.toggleViewAction())

        self.setDockNestingEnabled(True)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)

    def closeEvent(self, event: QCloseEvent) -> None:
        """ Closing the main window quits the application -- unless the user vetoes the quit. """
        self.quit()
        event.ignore()

    def open(self) -> None:
        """
        TODO: This is a placeholder for the "open a different folder" handler. It just raises a message box for now.
        """
        QMessageBox.about(self, f"Open", f"This command will raise a file dialog so that you can select a folder"
                                         f"containing spike-sorter files for analysis.")

    def quit(self) -> None:
        """
        Handler for the Exit/Quit menu command. Unless user vetoes the operation, the current GUI layout is
        saved to the user preferences and exit() is called on the main application object.
        """
        # TODO: If there are changes that need saving, ask.
        res = QMessageBox.question(self, "Exit", "Are you sure you want to quit?")
        if res == QMessageBox.StandardButton.Yes:
            self._save_layout_to_settings()
            self._app.exit(0)

    def about(self) -> None:
        """
        Handler for the 'About <application name>' menu command. It raises a modal message dialog describing the
        application.
        """
        QMessageBox.about(self, f"About {APP_NAME}", f"A description will go here!")

    def _save_layout_to_settings(self) -> None:
        """
        Helper method preseerves GUI layout settings in user preferences storage on host system. This method should
        be called prior to application exit, but before the GUI is destroyed.
        """

    def _restore_layout_from_settings(self) -> None:
        """
        Helper method called at startup that restores the GUI layout in effect when the application last shutdown
        normally. This must not be called until after the main window and all views have been realized (but not
        shown). If the user preferences storage is not found on host, the GUI will come up in a default layout.
        """
        pass


if __name__ == "__main__":
    main_app = QApplication(sys.argv)
    main_window = XSortMainWindow(main_app)
    main_window.show()
    exit_code = main_app.exec()
    # Any after-exit tasks can go here (should not take too long!)
    sys.exit(exit_code)
