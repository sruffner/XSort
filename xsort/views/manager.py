from importlib import resources as impresources
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Slot, QStandardPaths, QSettings, QCoreApplication, QByteArray, Qt, QObject, QSize
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QMessageBox, QFileDialog, QMenu, QDockWidget, QDialog, QVBoxLayout, \
    QDialogButtonBox, QTextBrowser

import xsort.assets as xsort_assets
from xsort.data.analyzer import Analyzer, DataType
from xsort.constants import APP_NAME
from xsort.views.acgrateview import ACGRateView
from xsort.views.baseview import BaseView
from xsort.views.channelview import ChannelView
from xsort.views.firingrateview import FiringRateView
from xsort.views.isiview import ISIView
from xsort.views.neuronview import NeuronView
from xsort.views.pcaview import PCAView
from xsort.views.correlogramview import CorrelogramView
from xsort.views.templateview import TemplateView


class _UserGuideView(BaseView):
    """
    This view contains a simple user's guide for XSort, maintained in the Markdown file :file:`assets/help.md`. We put
    this in a view so the user can dock the guide like any other real data view.
    """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Help', None, data_manager)
        self._help_browser = QTextBrowser()
        """ The user guide content is displayed entirelyl in this widget. """

        # load the user guide contents
        # noinspection PyTypeChecker
        inp_file = (impresources.files(xsort_assets) / 'help.md')
        with inp_file.open("rt") as f:
            markdown = f.read()
        self._help_browser.setReadOnly(True)
        self._help_browser.setOpenExternalLinks(True)
        self._help_browser.setMarkdown(markdown)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._help_browser)

        self.view_container.setLayout(main_layout)


class ViewManager(QObject):
    """
    TODO: UNDER DEV -- This is essentially the view + controller for XSort. It builds the UI in the main application
        window and responds to signals from the Analyzer, which serves as the master data model.
    """
    def __init__(self, main_window: QMainWindow):
        super().__init__(None)
        self._main_window = main_window
        """ Reference to the main application window -- to update standard UIlike status bar and window title."""
        self.data_analyzer = Analyzer(main_window)
        """
        The master data model. It encapsulates the notion of XSort's current working directory, mediates access to 
        data stored in the various files within that directory, performs analyses triggered by view actions, and so on.
        """
        self._working_dir_path_at_startup: Optional[str] = None
        """ 
        File system path of the XSort working directory as obtained from user settings during startup. Once the
        main application window is visible, we make this the current working directory if it is still valid.
        """
        self._about_dlg = self._create_about_dialog()
        """ The application's 'About' dialog. """

        self._neuron_view = NeuronView(self.data_analyzer)
        self._templates_view = TemplateView(self.data_analyzer)
        self._correlogram_view = CorrelogramView(self.data_analyzer)
        self._isi_view = ISIView(self.data_analyzer)
        self._acg_vs_rate_view = ACGRateView(self.data_analyzer)
        self._firingrate_view = FiringRateView(self.data_analyzer)
        self._pca_view = PCAView(self.data_analyzer)
        self._channels_view = ChannelView(self.data_analyzer)
        self._user_guide_view = _UserGuideView(self.data_analyzer)

        self._all_views = [self._neuron_view, self._templates_view, self._correlogram_view, self._isi_view,
                           self._acg_vs_rate_view, self._firingrate_view, self._pca_view, self._channels_view,
                           self._user_guide_view]
        """ List of all managed views. """

        # actions and menus
        self._open_action: Optional[QAction] = None
        self._quit_action: Optional[QAction] = None
        self._about_action: Optional[QAction] = None
        self._about_qt_action: Optional[QAction] = None
        self._undo_action: Optional[QAction] = None
        self._undo_all_action: Optional[QAction] = None
        self._delete_action: Optional[QAction] = None
        self._merge_action: Optional[QAction] = None
        self._split_action: Optional[QAction] = None

        self._file_menu: Optional[QMenu] = None
        self._edit_menu: Optional[QMenu] = None
        self._view_menu: Optional[QMenu] = None
        self._help_menu: Optional[QMenu] = None

        self._construct_ui()

        # connect to Analyzer signals
        self.data_analyzer.working_directory_changed.connect(self.on_working_directory_changed)
        self.data_analyzer.progress_updated.connect(self.on_background_task_updated)
        self.data_analyzer.data_ready.connect(self.on_data_ready)
        self.data_analyzer.focus_neurons_changed.connect(self.on_focus_neurons_changed)
        self.data_analyzer.channel_seg_start_changed.connect(self.on_channel_seg_start_changed)
        self.data_analyzer.neuron_label_updated.connect(self.on_neuron_label_updated)
        self.data_analyzer.split_lasso_region_updated.connect(self._refresh_menus)

        self._main_window.setMinimumSize(800, 600)
        self._restore_from_settings()
        self._main_window.setWindowTitle(self.main_window_title)

    def _construct_ui(self) -> None:
        """
        Builds and lays outs the UI within the main application window. This method must be called at startup, prior to
        showing the window and prior to restoring its state from user settings. It creates and configures UI actions and
        menus, installs the "Neurons" view as the central widget in the application window, and installs all other views
        in dock widgets.
            By default, all views are initially docked along the right edge, but the user can dock to the right or
        bottom edge of the main window. The bottom right corner is reserved for the right docking area. Nesting is
        permitted. Each dock widget is assigned a unique name '<title>-DOCK', where <title> is the title of the view it
        contains, so that its  state can be saved to and restored from user settings. Hence it is critical to set up
        the dock widgets before restoring the application window's state from those settings!
        """
        # the table of neurons is the central widget in the application window. Everything else is in dock widgets.
        self._main_window.setCentralWidget(self._neuron_view.view_container)

        # actions
        self._open_action = QAction('&Open', self._main_window, shortcut="Ctrl+O", statusTip="Select working directory",
                                    triggered=self.select_working_directory)
        self._quit_action = QAction("&Quit", self._main_window, shortcut="Ctrl+Q", statusTip=f"Quit {APP_NAME}",
                                    triggered=self.quit)

        self._about_action = QAction("&About", self._main_window, statusTip=f"About {APP_NAME}", triggered=self._about)

        self._about_qt_action = QAction("About Qt", self._main_window, statusTip="About the Qt library",
                                        triggered=QCoreApplication.instance().aboutQt)

        self._undo_action = QAction("&Undo", self._main_window, shortcut="Ctrl+Z", enabled=False, triggered=self._undo)
        self._undo_all_action = QAction("Undo &All", self._main_window, enabled=False, triggered=self._undo_all)
        self._delete_action = QAction("&Delete", self._main_window, shortcut="Ctrl+X", enabled=False,
                                      triggered=self._delete)
        self._merge_action = QAction("&Merge", self._main_window, shortcut="Ctrl+M", enabled=False,
                                     triggered=self._merge)
        self._split_action = QAction("S&plit", self._main_window, shortcut="Ctrl+Y", enabled=False,
                                     triggered=self._split)

        # menus - note that I couldn't get tool tips to show for menu actions on MacOS. So, for the Undo action, the
        # action's text reflects the operation to be undone. See _refresh_menus().
        self._file_menu = self._main_window.menuBar().addMenu("&File")
        self._file_menu.addAction(self._open_action)
        self._file_menu.addSeparator()
        self._file_menu.addAction(self._quit_action)
        self._edit_menu = self._main_window.menuBar().addMenu("&Edit")
        self._edit_menu.addAction(self._undo_action)
        self._edit_menu.addAction(self._undo_all_action)
        self._edit_menu.addSeparator()
        self._edit_menu.addAction(self._delete_action)
        self._edit_menu.addAction(self._merge_action)
        self._edit_menu.addAction(self._split_action)

        # the View menu controls the visibility of all dockable views
        self._view_menu = self._main_window.menuBar().addMenu("&View")
        for v in self.dockable_views:
            dock = QDockWidget(v.title, self._main_window)
            dock.setObjectName(f"{v.title}-DOCK")
            dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
            dock.setWidget(v.view_container)
            self._main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
            # dock widget holding the user guide is separated from the other views and hidden by default
            if isinstance(v, _UserGuideView):
                self._view_menu.addSeparator()
                dock.setHidden(True)
            self._view_menu.addAction(dock.toggleViewAction())

        self._main_window.setDockNestingEnabled(True)
        self._main_window.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)

        self._main_window.menuBar().addSeparator()

        self._help_menu = self._main_window.menuBar().addMenu("&Help")
        # under the hood, these are automatically put in the "Apple" menu on Mac OS X
        self._help_menu.addAction(self._about_action)
        self._help_menu.addAction(self._about_qt_action)

        # status bar
        self._main_window.statusBar().showMessage("Ready")

    def _create_about_dialog(self) -> QDialog:
        dlg = QDialog(self._main_window)
        dlg.setWindowTitle("About XSort")

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(dlg.accept)

        # noinspection PyTypeChecker
        inp_file = (impresources.files(xsort_assets) / 'about.md')
        with inp_file.open("rt") as f:
            markdown = f.read()
        about_browser = QTextBrowser()
        about_browser.setReadOnly(True)
        about_browser.setOpenExternalLinks(True)
        about_browser.setMarkdown(markdown)

        layout = QVBoxLayout()
        layout.addWidget(about_browser)
        layout.addWidget(button_box)
        dlg.setLayout(layout)
        dlg.setMinimumSize(QSize(600, 400))
        return dlg

    @property
    def dockable_views(self) -> List[BaseView]:
        """
        List of dockable XSort views -- ie, all views other than the central view.
        """
        return self._all_views[1:]

    @property
    def main_window_title(self) -> str:
        """
        String to be displayed in the title bar of the main application window. This reflects the current working
        directory, if defined.
        """
        if isinstance(self.data_analyzer.working_directory, Path):
            return f"{APP_NAME} ({str(self.data_analyzer.working_directory)})"
        else:
            return APP_NAME

    @Slot()
    def on_working_directory_changed(self) -> None:
        """ Handler updates all views and refreshes menu state when the current working directory has changed. """
        for v in self._all_views:
            v.on_working_directory_changed()
        self._refresh_menus()

    @Slot(str)
    def on_background_task_updated(self, msg: str) -> None:
        """
        Handler for progress updates from background tasks run by :class:`Analyzer`. The progress message is displayed
        in the application window's status bar. If the message is empty, the status bar will ready "Ready".

        :param msg: The progress message.
        """
        self._main_window.statusBar().showMessage(msg if isinstance(msg, str) and (len(msg) > 0) else "Ready")

    @Slot(DataType, str)
    def on_data_ready(self, dt: DataType, uid: str) -> None:
        """
        Handler updates relevant views when the data model signals that some data is ready to be accessed.
        :param dt: The type of data object retrieved or prepared for access.
        :param uid: Object identifier -- the unique label of a neural unit, or the integer index (in string form) of
            the analog channel source for a channel trace segment.
        """
        if dt == DataType.NEURON:
            for v in self._all_views:
                v.on_neuron_metrics_updated(uid)
        elif dt == DataType.CHANNELTRACE:
            for v in self._all_views:
                v.on_channel_trace_segment_updated(int(uid))
        else:
            for v in self._all_views:
                v.on_focus_neurons_stats_updated(dt, uid)
        self._refresh_menus()

    @Slot()
    def on_focus_neurons_changed(self) -> None:
        """
        Handler notifies all views when there's any change in the subset of neurons having the display focus. It also
        refreshes the state of the edit actions, which depend on whether and how many units are selected.
        """
        for v in self._all_views:
            v.on_focus_neurons_changed()
        self._refresh_menus()

    @Slot()
    def on_neuron_label_updated(self, uid: str) -> None:
        """
        Handler notifies all views when a neural unit's label has changed.
        :param uid: The UID identifying the affected neural unit.
        """
        for v in self._all_views:
            v.on_neuron_label_updated(uid)
        self._refresh_menus()

    @Slot()
    def on_channel_seg_start_changed(self) -> None:
        """
        Handler notifices all views when there's a change in the elapsed start time at which all analog channel trace
        segments begin.
        """
        for v in self._all_views:
            v.on_channel_trace_segment_start_changed()

    def select_working_directory(self, starting_up: bool = False) -> None:
        """
        Raise a modal file dialog by which user can change the current working directory for XSort. On startup, raise
        the dialog only if a valid working directory is not already specified (in user application settings).
            XSort requires a working directory containing the data source files it reads and analyzes; it also writes
        internal cache files to the directory while preprocessing those data files.

        :param starting_up: If True, then application has just launched. If a valid working directory is not already
        specified in the user's application settings, the user must choose such a directory before the application can
        continue. In this scenario, an explanatory message dialog notifies the user and offers the option to quit the
        application if no valid working directory exists.
        """
        if starting_up and isinstance(self._working_dir_path_at_startup, str):
            if self.data_analyzer.change_working_directory(self._working_dir_path_at_startup) is None:
                return

        curr_dir = self.data_analyzer.working_directory
        ok = False
        msg_btns = QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Abort
        if self.data_analyzer.is_valid_working_directory:
            msg_btns |= QMessageBox.StandardButton.Cancel
        err_msg: Optional[str] = \
            ("You must specify an existing directory that contains the source data files (Omniplex recording, spike "
             "sorter results) that XSort requires.") if starting_up else None
        while not ok:
            if isinstance(err_msg, str):
                res = QMessageBox.warning(self._main_window, "Select working directory", err_msg,
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
            work_dir = QFileDialog.getExistingDirectory(self._main_window, "Select working directory for XSort",
                                                        str(parent_path.absolute()))
            if work_dir == "" and self.data_analyzer.is_valid_working_directory:
                # user cancelled and current working directory is valid
                ok = True
            else:
                err_msg = self.data_analyzer.change_working_directory(work_dir)
                ok = (err_msg is None)

        self._main_window.setWindowTitle(self.main_window_title)

    def quit(self) -> None:
        """
        Handler for the Exit/Quit menu command. Unless user vetoes the operation, the current GUI layout and all other
        application settings are saved to the user preferences and exit() is called on the main application object.
        """
        res = QMessageBox.question(self._main_window, "Exit", "Are you sure you want to quit?")
        if res == QMessageBox.StandardButton.Yes:
            self.data_analyzer.prepare_for_shutdown()
            self._save_settings_and_exit()

    def _about(self) -> None:
        """
        Handler for the 'About <application name>' menu command. It raises a modal message dialog describing the
        application.
        """
        self._about_dlg.exec()

    def _undo(self) -> None:
        """ Handler for the 'Edit|Undo' menu command. It undos the last change to the current neural unit list. """
        self.data_analyzer.undo_last_edit()
        self._refresh_menus()

    def _undo_all(self) -> None:
        """ Handler for the 'Edit|Undo All' menu command. """
        self.data_analyzer.undo_all_edits()

    def _delete(self) -> None:
        """
        Handler for the 'Edit|Delete' menu command. It deletes the currently selected neuron, if possible, and
        moves the selection to a neighboring unit in the neuron list as displayed in the neuron view.
        """
        try:
            uid_to_delete = self.data_analyzer.primary_neuron.uid
            uid_neighbor = self._neuron_view.uid_of_unit_below(uid_to_delete)
            self.data_analyzer.delete_primary_neuron(uid_neighbor)
        except Exception:
            pass

    def _merge(self) -> None:
        """ Handler for the 'Edit|Merge menu command; invokes the :class:`Analyzer` method that does the merge. """
        self.data_analyzer.merge_focus_neurons()

    def _split(self) -> None:
        """ Handler for the 'Edit|Split menu command; invokes the :class:`Analyzer` method that does the split. """
        self.data_analyzer.split_primary_neuron()

    def _refresh_menus(self) -> None:
        """ Update the enabled state and item text for selected menu items. """
        descriptors = self.data_analyzer.undo_last_edit_description()
        if isinstance(descriptors, tuple):
            self._undo_action.setText(f"&Undo: {descriptors[1]}")
            self._undo_action.setEnabled(True)
        else:
            self._undo_action.setText("&Undo")
            self._undo_action.setEnabled(False)

        self._undo_all_action.setEnabled(self.data_analyzer.can_undo_all_edits())

        can_delete = self.data_analyzer.can_delete_primary_neuron()
        if can_delete:
            self._delete_action.setText(f"&Delete unit {self.data_analyzer.primary_neuron.uid}")
            self._delete_action.setEnabled(True)
        else:
            self._delete_action.setText(f"&Delete")
            self._delete_action.setEnabled(False)

        can_merge = self.data_analyzer.can_merge_focus_neurons()
        if can_merge:
            uids = [u.uid for u in self.data_analyzer.neurons_with_display_focus]
            self._merge_action.setText(f"&Merge units {','.join(uids)}")
            self._merge_action.setEnabled(True)
        else:
            self._merge_action.setText(f"&Merge")
            self._merge_action.setEnabled(False)

        can_split = self.data_analyzer.can_split_primary_neuron()
        if can_split:
            self._split_action.setText(f"&Split unit {self.data_analyzer.primary_neuron.uid}")
            self._split_action.setEnabled(True)
        else:
            self._split_action.setText(f"&Split")
            self._split_action.setEnabled(False)

    def _save_settings_and_exit(self) -> None:
        """ Save all user settings and exit the application without user prompt. """
        # TODO: Avoiding native settings format so I can examine INI file during development
        #  settings = QSettings(xs.ORG_DOMAIN, xs.APP_NAME)
        settings_path = Path(QStandardPaths.writableLocation(QStandardPaths.HomeLocation),
                             f".{APP_NAME}.ini")
        settings = QSettings(str(settings_path.absolute()), QSettings.IniFormat)

        # basic layout
        settings.setValue('geometry', self._main_window.saveGeometry())
        settings.setValue('window_state', self._main_window.saveState(version=0))  # increment version as needed

        # any view-specific settings
        for v in self._all_views:
            v.save_settings(settings)

        # current working directory
        settings.setValue('working_dir', str(self.data_analyzer.working_directory))

        QCoreApplication.instance().exit(0)

    def _restore_from_settings(self) -> None:
        """
        Helper method called at application startup to restore all user settings -- GUI layout, view-specific settings,
        current working directory -- that were saved when the application last exited. This must not be called until
        after the main window and all views have been realized (but not shown).
            If the user preferences storage is not found or could not be read, application defaults are used for GUI
        layout and any view-specific settings. If the working directory persisted in user preferences storage no longer
        exists, the application will immediately ask the user to specify one.
        """
        # TODO: Avoiding native settings format so I can examine INI file during development
        #  settings = QSettings(xs.ORG_DOMAIN, xs.APP_NAME)
        settings_path = Path(QStandardPaths.writableLocation(QStandardPaths.HomeLocation),
                             f".{APP_NAME}.ini")
        settings = QSettings(str(settings_path.absolute()), QSettings.IniFormat)

        # basic layout
        geometry = settings.value("geometry", QByteArray())
        if geometry and isinstance(geometry, QByteArray) and not geometry.isEmpty():
            self._main_window.restoreGeometry(geometry)
        else:
            self._main_window.setGeometry(200, 200, 800, 600)

        window_state = settings.value("window_state", QByteArray())
        if window_state and isinstance(window_state, QByteArray) and not window_state.isEmpty():
            self._main_window.restoreState(window_state)

        # any view-specific settings
        for v in self._all_views:
            v.restore_settings(settings)

        # remember the working directory path if available. Once the application window is shown, we make this the
        # current working directory -- see select_working_directory()
        str_path: Optional[str] = settings.value("working_dir")
        if isinstance(str_path, str):
            self._working_dir_path_at_startup = str_path
