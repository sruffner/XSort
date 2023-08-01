import sys
from pathlib import Path
from typing import List, Dict, Optional

from PySide6.QtCore import Qt, Slot, QSettings, QStandardPaths, QPoint, QSize, QRect
from PySide6.QtGui import QColor, QKeySequence, QIcon, QCloseEvent, QAction, QGuiApplication
from PySide6.QtWidgets import QApplication, QMainWindow, QSplitter, QMenu, QMessageBox

# TODO: Python path hack. I cannot figure out how to organize program structure so I can run app.py from PyCharm IDE
#  and also package Xsort for distribution and running the program via python -m xsort.app. I hate this.
p = Path(__file__).parent
sys.path.append(str(p.absolute()))

from baseview import BaseView

from constants import APP_NAME


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self, app: QApplication):
        super().__init__()
        self.setWindowTitle(APP_NAME)

        self.app = app
        self.view_mgr = ViewManager(self)

        self.left_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.left_splitter.addWidget(self.view_mgr.views[0])
        self.left_splitter.addWidget(self.view_mgr.views[1])
        self.left_splitter.setOpaqueResize(False)
        self.left_splitter.setChildrenCollapsible(False)

        self.mid_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.mid_splitter.addWidget(self.view_mgr.views[2])
        self.mid_splitter.addWidget(self.view_mgr.views[3])
        self.mid_splitter.addWidget(self.view_mgr.views[4])
        self.mid_splitter.setOpaqueResize(False)
        self.mid_splitter.setChildrenCollapsible(False)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.right_splitter.addWidget(self.view_mgr.views[5])
        self.right_splitter.addWidget(self.view_mgr.views[6])
        self.right_splitter.setOpaqueResize(False)
        self.right_splitter.setChildrenCollapsible(False)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.main_splitter.addWidget(self.left_splitter)
        self.main_splitter.addWidget(self.mid_splitter)
        self.main_splitter.addWidget(self.right_splitter)
        self.main_splitter.setOpaqueResize(False)

        # Set the central widget of the Window.
        self.setCentralWidget(self.main_splitter)
        self.setMinimumSize(800, 600)

        about_action = QAction(f"About {APP_NAME}", self)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)
        prefs_action = QAction("Preferences", self)
        prefs_action.setMenuRole(QAction.MenuRole.PreferencesRole)
        quit_action = QAction("Exit", self)
        quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        quit_action.triggered.connect(self.quit)

        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(QAction("&Open", self, QKeySequence.Open))
        file_menu.addSeparator()
        file_menu.addAction(about_action)
        file_menu.addAction(prefs_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = menu.addMenu("&View")
        self.view_mgr.build_view_menu(view_menu)

        self.view_mgr.restore_layout_from_settings()

    def closeEvent(self, event: QCloseEvent) -> None:
        """ Closing the main window quits the application -- unless the user vetoes the quit. """
        self.quit()
        event.ignore()

    def quit(self) -> None:
        """ Handler for the Exit/Quit menu command. Unless user vetoes the operation, the current GUI layout is
        saved to the user preferences and quit() is called on the main application object. """
        # TODO: If there are changes that need saving, ask.
        res = QMessageBox.question(self, "Exit", "Are you sure you want to quit?")
        if res == QMessageBox.StandardButton.Yes:
            self.view_mgr.save_layout_to_settings()
            self.app.exit(0)


_VIEWS: List[Dict] = [
    dict(title="Neurons", uid=0, hideable=False, floatable=False, bkg=QColor('red'), dock='left', dock_pos=0),
    dict(title="Similarity", uid=1, hideable=True, floatable=True, bkg=QColor('blue'), dock='left', dock_pos=1),
    dict(title="Templates", uid=2, hideable=True, floatable=True, bkg=QColor('green'), dock='mid', dock_pos=0),
    dict(title="Statistics", uid=3, hideable=True, floatable=True, bkg=QColor('yellow'), dock='mid', dock_pos=1),
    dict(title="PCA", uid=4, hideable=True, floatable=True, bkg=QColor('cyan'), dock='mid', dock_pos=2),
    dict(title="Channels", uid=5, hideable=True, floatable=True, bkg=QColor('black'), dock='right', dock_pos=0),
    dict(title="UMAP", uid=6, hideable=True, floatable=True, bkg=QColor('magenta'), dock='right', dock_pos=1)
]


class ViewManager(object):
    """
    The application view manager. Creates and manages all application views, all of which are subclasses of
    :class:`BaseView`.
    """
    def __init__(self, w: MainWindow):
        self._main_window = w
        self.views: List[BaseView] = []
        """ The set of all views available in XSort and managed by this object. """
        self._create_views()

    def _create_views(self):
        for v in _VIEWS:
            hide_toggle: Optional[QAction] = None
            dock_toggle: Optional[QAction] = None
            if v['hideable']:
                hide_toggle = QAction(text=v['title'], parent=self._main_window)
                hide_toggle.setCheckable(True)
                hide_toggle.setChecked(True)
                hide_toggle.setAutoRepeat(False)
                hide_toggle.setIconVisibleInMenu(False)
                hide_toggle.triggered.connect(lambda *args, uid=v['uid']: self.toggle_hide(uid))
            if v['floatable']:
                p = Path(__file__).parent / 'assets/float-window.png'
                dock_toggle = QAction(icon=QIcon(str(p.absolute())), text="", parent=self._main_window)
                dock_toggle.setToolTip("Undock")
                dock_toggle.setAutoRepeat(False)
                dock_toggle.setIconVisibleInMenu(False)
                dock_toggle.triggered.connect(lambda *args, uid=v['uid']: self.toggle_dock(uid))

            self.views.append(BaseView(v['uid'], v['title'], v['dock'], v['dock_pos'], hide_toggle, dock_toggle,
                                       background=v['bkg']))

    def hide_all_views(self):
        """
        Hide all views. This should ONLY be called during application shutdown to ensure that any views currently
        floating in a top-level window are hidden -- otherwise the application will continue running even though the
        main window is gone!
        """
        for v in self.views:
            v.setVisible(False)

    def build_view_menu(self, view_menu: QMenu) -> None:
        """
        Helper method populates the application **View** menu with a separate entry for each application view that
        may be shown/hidden by the user. The menu item will be checked whenever the corresponding view is visible.
        Additional entries let the user show all views or dock and show all views.

           This method should only be called during application startup, before the main window is shown.
        """
        for view in self.views:
            if view.hide_action:
                view_menu.addAction(view.hide_action)
        view_menu.addSeparator()
        show_all = QAction("Show all", self._main_window)
        show_all.setAutoRepeat(False)
        show_all.triggered.connect(self.show_all_views)
        view_menu.addAction(show_all)
        dock_all = QAction("Dock all", self._main_window)
        dock_all.setAutoRepeat(False)
        dock_all.triggered.connect(self.dock_and_show_all_views)
        view_menu.addAction(dock_all)
        view_menu.addSeparator()

    @Slot()
    def show_all_views(self) -> None:
        """ Ensures that all application views are visible. """
        for view in self.views:
            if view.is_hidden:
                view.setVisible(True)
                view.hide_action.setChecked(True)

    @Slot()
    def dock_and_show_all_views(self) -> None:
        """ Redocks any floating application views and ensures that all views are visible. """
        for view in self.views:
            if view.is_floated:
                self.toggle_dock(view.uid)
        self.show_all_views()
        self._main_window.repaint()

    @Slot(int)
    def toggle_hide(self, uid: int) -> None:
        """
        Toggles the visibility of the specified application view.
        :param uid: The view ID.
        """
        view = self.views[uid]
        if view.hide_action:
            view.setVisible(True if view.is_hidden else False)

    @Slot(int)
    def toggle_dock(self, uid: int) -> None:
        """
        Toggles the docking state -- either docked within its splitter parent in the main window, or floating as a
        top-level window -- of the specified application view.
        :param uid: The view ID.
        """
        view = self.views[uid]
        splitter = self._main_window.left_splitter if view.dock_panel == 'left' else \
            (self._main_window.mid_splitter if view.dock_panel == 'mid' else self._main_window.right_splitter)

        if view.is_floated:
            was_visible = not view.is_hidden
            splitter.insertWidget(view.dock_position, view)
            view.show_title_bar(True)
            if was_visible:
                view.show()
                splitter.repaint()
        else:
            was_visible = not view.is_hidden
            view.hide()
            # noinspection PyTypeChecker
            view.setParent(None)
            view.show_title_bar(False)
            view.setGeometry(view.window_geometry)
            if was_visible:
                view.show()
                splitter.repaint()

    def restore_layout_from_settings(self) -> None:
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
        settings.beginGroup('layout')

        size: Optional[QSize] = settings.value('w_size')
        if size is None:
            # TODO: What if there are multiple screens?
            screen = QGuiApplication.screens()[0]
            geom = screen.availableGeometry()
            size = QSize(geom.width() * 2 / 3, geom.height() * 2 / 3)
        self._main_window.resize(size)

        pos: Optional[QPoint] = settings.value('w_pos')
        self._main_window.move(pos or QPoint(60, 60))

        num_views = settings.beginReadArray('views')
        if num_views == len(self.views):
            for i in range(num_views):
                settings.setArrayIndex(i)
                floated: bool = settings.value('floated', type=bool)
                hidden: bool = settings.value('hidden', type=bool)
                w_rect: Optional[QRect] = settings.value('w_rect')
                if floated and (self.views[i].dock_action is not None):
                    self.views[i].dock_action.trigger()
                if hidden and (self.views[i].hide_action is not None):
                    self.views[i].hide_action.trigger()
                if w_rect:
                    self.views[i].window_geometry = w_rect
                    if floated:
                        self.views[i].setGeometry(w_rect)
        settings.endArray()
        # split sizes may be returned as stringized integers, so we have to convert back
        for entry in [dict(n='left_split', s=self._main_window.left_splitter),
                      dict(n='mid_split', s=self._main_window.mid_splitter),
                      dict(n='right_split', s=self._main_window.right_splitter),
                      dict(n='main_split', s=self._main_window.main_splitter)]:
            sizes: Optional[List] = settings.value(entry['n'])
            if sizes:
                entry['s'].setSizes([int(k) for k in sizes])
        settings.endGroup()

    def save_layout_to_settings(self) -> None:
        """
        Helper method preseerves GUI layout settings in user preferences storage on host system. This method should
        be called prior to application exit, but before the GUI is destroyed.
        """
        # TODO: Avoiding native settings format so I can examine INI file during development
        #  settings = QSettings(_ORG_DOMAIN, _APP_NAME)
        settings_path = Path(QStandardPaths.writableLocation(QStandardPaths.HomeLocation),
                             f".{APP_NAME}.ini")
        settings = QSettings(str(settings_path.absolute()), QSettings.IniFormat)
        settings.beginGroup('layout')
        settings.setValue('w_pos', self._main_window.pos())
        settings.setValue('w_size', self._main_window.size())
        settings.beginWriteArray('views')
        for i, view in enumerate(self.views):
            settings.setArrayIndex(i)
            settings.setValue('hidden', view.is_hidden)
            settings.setValue('floated', view.is_floated)
            settings.setValue('w_rect', view.window_geometry)
        settings.endArray()
        for entry in [dict(n='left_split', s=self._main_window.left_splitter),
                      dict(n='mid_split', s=self._main_window.mid_splitter),
                      dict(n='right_split', s=self._main_window.right_splitter),
                      dict(n='main_split', s=self._main_window.main_splitter)]:
            settings.setValue(entry['n'], entry['s'].sizes())
        settings.endGroup()


if __name__ == "__main__":
    main_app = QApplication(sys.argv)
    main_window = MainWindow(main_app)
    main_window.show()
    exit_code = main_app.exec()
    # Any after-exit tasks can go here (should not take too long!)
    sys.exit(exit_code)
