from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QStandardPaths, QSettings, QByteArray, QTimer
from PySide6.QtGui import QCloseEvent, QAction, QShowEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QMessageBox, QDockWidget, QFileDialog

from xsort.constants import APP_NAME
from xsort.views.manager import ViewManager


class XSortMainWindow(QMainWindow):
    """
    TODO: UNDER DEV.
    """

    def __init__(self, app: QApplication):
        super().__init__()
        self._app = app
        """ The application instance. """
        self._view_manager = ViewManager()
        """ The model-view controller. """
        self._starting_up = True
        """ Flag set at startup so that all views will be updated to reflect contents of current working directory. """

        self.setCentralWidget(self._view_manager.central_view.view_container)

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

        self.setMinimumSize(800, 600)
        self._restore_layout_from_settings()
        self._restore_workstate_from_settings()
        self.setWindowTitle(self._view_manager.main_window_title)

    def create_actions(self):
        self._open_action = QAction('&Open', self, shortcut="Ctrl+O", statusTip="Select working directory",
                                    triggered=self.select_working_directory)
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
        # populated later

        self.menuBar().addSeparator()

        self._help_menu = self.menuBar().addMenu("&Help")
        self._help_menu.addAction(self._about_action)
        self._help_menu.addAction(self._about_qt_action)

    def create_tool_bars(self):
        pass

    def create_status_bar(self):
        self.statusBar().showMessage("Ready")

    def install_docked_views(self):
        """
        Helper method installs each of the dockable views in a dock widget. By default, all views are initially docked
        along the right edge, but the user can dock to the right or bottom edge of the main window. The bottom right
        corner is reserved for the right docking area. Nesting is permitted.
            This method must be called prior to restoring the main window's state from user settings. Each dock widget
        is assigned a unique name '<title>-DOCK', where <title> is the title of the view it contains, so that its
        state can be saved to and restored from user settings.
        """
        for v in self._view_manager.dockable_views:
            dock = QDockWidget(v.title, self)
            dock.setObjectName(f"{v.title}-DOCK")
            dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
            dock.setWidget(v.view_container)
            self.addDockWidget(Qt.RightDockWidgetArea, dock)
            self._view_menu.addAction(dock.toggleViewAction())

        self.setDockNestingEnabled(True)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)

    def closeEvent(self, event: QCloseEvent) -> None:
        """ [QWidget override] Closing the main window quits the application -- unless the user vetoes the quit. """
        self.quit()
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
        self.select_working_directory(starting_up=True)

    def select_working_directory(self, starting_up: bool = False) -> None:
        """
        Raise a modal file dialog by which user can change the current working directory for XSort. On startup, raise
        the dialog only if a valid working directory is not already specified (in user application settings).
            XSort requires a working directory containing the data source files it reads and analyzes; it also writes
        internal cache files to the directory while preprocessing those data files.

        :param starting_up: If True, then application has just launched and main window is not yet shown. If a valid
        working directory is not already specified in the user's application settings, the user must choose such a
        directory before the application can continue. In this scenario, an explanatory message dialog notifies the
        user and offers the option to quit the application if no valid working directory exists.
        """
        curr_dir = self._view_manager.data_analyzer.working_directory
        if starting_up and self._view_manager.data_analyzer.is_valid_working_directory:
            return
        ok = False
        msg_btns = QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Abort
        if self._view_manager.data_analyzer.is_valid_working_directory:
            msg_btns |= QMessageBox.StandardButton.Cancel
        err_msg: Optional[str] = \
            ("You must specify an existing directory that contains the source data files (Omniplex recording, spike "
             "sorter results) that XSort requires.") if starting_up else None
        while not ok:
            if isinstance(err_msg, str):
                res = QMessageBox.warning(self, "Select working directory", err_msg,
                                          buttons=msg_btns, defaultButton=QMessageBox.StandardButton.Ok)
                if res == QMessageBox.StandardButton.Abort:
                    self._save_settings_and_exit()
                    return
                elif res == QMessageBox.StandardButton.Cancel:
                    return
            if isinstance(curr_dir, Path):
                parent_path = curr_dir.parent if isinstance(curr_dir.parent, Path) else curr_dir
            else:
                parent_path = Path(QStandardPaths.writableLocation(QStandardPaths.HomeLocation))
            work_dir = QFileDialog.getExistingDirectory(self, "Select working directory for XSort",
                                                        str(parent_path.absolute()))
            if work_dir == "" and self._view_manager.data_analyzer.is_valid_working_directory:
                # user cancelled and current working directory is valid
                ok = True
            else:
                err_msg = self._view_manager.data_analyzer.change_working_directory(work_dir)
                ok = (err_msg is None)

        self.setWindowTitle(self._view_manager.main_window_title)

    def quit(self) -> None:
        """
        Handler for the Exit/Quit menu command. Unless user vetoes the operation, the current GUI layout and all other
        application settings are saved to the user preferences and exit() is called on the main application object.
        """
        res = QMessageBox.question(self, "Exit", "Are you sure you want to quit?")
        if res == QMessageBox.StandardButton.Yes:
            self._save_settings_and_exit()

    def _save_settings_and_exit(self) -> None:
        """ Save all user settings and exit the application without user prompt. """
        self._save_layout_to_settings()
        self._save_workstate_to_settings()
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
        # TODO: Avoiding native settings format so I can examine INI file during development
        #  settings = QSettings(xs.ORG_DOMAIN, xs.APP_NAME)
        settings_path = Path(QStandardPaths.writableLocation(QStandardPaths.HomeLocation),
                             f".{APP_NAME}.ini")
        settings = QSettings(str(settings_path.absolute()), QSettings.IniFormat)
        settings.setValue('geometry', self.saveGeometry())
        settings.setValue('window_state', self.saveState(version=0))   # increment version as needed

    def _restore_layout_from_settings(self) -> None:
        """
        Helper method called at startup that restores the GUI layout in effect when the application last shutdown
        normally. This must not be called until after the main window and all views have been realized (but not
        shown). If the user preferences storage is not found on host, the GUI will come up in a default layout.
        """
        # TODO: Avoiding native settings format so I can examine INI file during development
        #  settings = QSettings(xs.ORG_DOMAIN, xs.APP_NAME)
        settings_path = Path(QStandardPaths.writableLocation(QStandardPaths.HomeLocation),
                             f".{APP_NAME}.ini")
        settings = QSettings(str(settings_path.absolute()), QSettings.IniFormat)

        geometry = settings.value("geometry", QByteArray())
        if geometry and isinstance(geometry, QByteArray) and not geometry.isEmpty():
            self.restoreGeometry(geometry)
        else:
            self.setGeometry(200, 200, 800, 600)

        window_state = settings.value("window_state", QByteArray())
        if window_state and isinstance(window_state, QByteArray) and not window_state.isEmpty():
            self.restoreState(window_state)

    def _save_workstate_to_settings(self) -> None:
        """
        Helper method preseerves the current working directory and other work state information in user preferences
        storage on host system. This method should be called prior to application exit.
        """
        # TODO: Avoiding native settings format so I can examine INI file during development
        #  settings = QSettings(xs.ORG_DOMAIN, xs.APP_NAME)
        settings_path = Path(QStandardPaths.writableLocation(QStandardPaths.HomeLocation),
                             f".{APP_NAME}.ini")
        settings = QSettings(str(settings_path.absolute()), QSettings.IniFormat)
        settings.setValue('working_dir', str(self._view_manager.data_analyzer.working_directory))

    def _restore_workstate_from_settings(self) -> None:
        """
        Helper method called at startup that restores the current working directory and other work state information in
        effect when the application last shutdown normally. If the user preferences storage is not found on host, or if
        it lacks an existing working directory that contains the source files required by XSort, all work state
        information is reset.
        """
        # TODO: Avoiding native settings format so I can examine INI file during development
        #  settings = QSettings(xs.ORG_DOMAIN, xs.APP_NAME)
        settings_path = Path(QStandardPaths.writableLocation(QStandardPaths.HomeLocation),
                             f".{APP_NAME}.ini")
        settings = QSettings(str(settings_path.absolute()), QSettings.IniFormat)

        str_path: Optional[str] = settings.value("working_dir")
        if isinstance(str_path, str):
            self._view_manager.data_analyzer.change_working_directory(str_path)
