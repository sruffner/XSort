from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, QRect
from PySide6.QtGui import QPalette, QColor, QGuiApplication, QCloseEvent, QIcon, QMoveEvent, QResizeEvent, QAction
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QVBoxLayout, QToolButton, QFrame, QHBoxLayout

from constants import APP_NAME


class BaseView(QWidget):
    """
    TODO: UNDER DEV
    """

    def __init__(self, uid: int, title: str, dock: str, dock_pos: int, hide_action: Optional[QAction] = None,
                 dock_action: Optional[QAction] = None, background: Optional[QColor] = None) -> None:
        """
        Create an empty view with the specified title and configuration.

        :param uid: The view's identifier.
        :param title: The view's title.
        :param dock: The splitter within which view is docked in main window: 'left', 'mid', or 'right'.
        :param dock_pos: The view's preferred ordinal position within its splitter.
        :param hide_action: The UI action that toggles the view's visibility. If None, view cannot be hidden.
        :param dock_action: The UI action that toggles the view's docking state. If None, view cannot be "floated".
        """
        super(BaseView, self).__init__()

        self._uid = uid
        """ The view's identifier. """
        self._title = title
        """ The view's title. """
        self._dock = dock
        """ Identifies the splitter within which view is docked in main window. """
        self._dock_pos = dock_pos
        """ The view's preferred ordinal position within its splitter parent (when docked). """
        self._toggle_hide = hide_action
        """ UI action that toggles view's visibility. If None, view cannot be hidden. """
        self._toggle_dock = dock_action
        """ UI action that toggles view's docking state. If None, view cannot be undocked. """
        self._title_bar = QFrame(self)
        """ The title bar for the view, present when the view is docked inside the main application window. """
        self._window_geometry: Optional[QRect] = None
        """ View geometry the last time the view floated as an independent top-level window. """

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, background)
        self.setPalette(palette)
        self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(QSize(100, 100))
        self.setWindowTitle(f"{APP_NAME} - {title}")

        self._title_bar.setLineWidth(1)
        self._title_bar.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self._title_bar.setAutoFillBackground(True)
        palette = self._title_bar.palette()
        default_palette = QGuiApplication.palette()
        palette.setColor(QPalette.Window, default_palette.color(QPalette.Window))
        self._title_bar.setPalette(palette)

        button_grp = QWidget()
        close_btn = QToolButton(self._title_bar)
        if self._toggle_hide:
            # wrap the toggle action in an uncheckable "close" action -- we don't want tool button to look checkable.
            p = Path(__file__).parent/'assets/close-hide.png'
            close_action = QAction(icon=QIcon(str(p.absolute())), text="", parent=self._toggle_hide.parent())
            close_action.setToolTip("Close")
            close_action.setCheckable(False)
            close_action.triggered.connect(lambda *args: self._toggle_hide.trigger())
            close_btn.setDefaultAction(close_action)
        close_btn.setMaximumSize(QSize(16, 16))
        close_btn.setVisible(self._toggle_hide is not None)

        dock_btn = QToolButton(self._title_bar)
        if self._toggle_dock:
            dock_btn.setDefaultAction(self._toggle_dock)
        dock_btn.setMaximumSize(QSize(16, 16))
        dock_btn.setVisible(self._toggle_dock is not None)

        layout = QHBoxLayout()
        layout.addWidget(dock_btn, alignment=Qt.AlignVCenter)
        layout.addWidget(close_btn, alignment=Qt.AlignVCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        button_grp.setLayout(layout)

        label = QLabel(title)
        label.setMargin(1)

        layout = QGridLayout()
        layout.addWidget(label, 0, 0, alignment=Qt.AlignVCenter | Qt.AlignLeading)
        layout.addWidget(button_grp, 0, 1, alignment=Qt.AlignVCenter | Qt.AlignTrailing)
        layout.setContentsMargins(0, 0, 0, 0)
        self._title_bar.setLayout(layout)

        layout = QVBoxLayout()
        layout.addWidget(self._title_bar, alignment=Qt.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    @property
    def uid(self) -> int:
        """ The view's ID. """
        return self._uid

    @property
    def title(self) -> str:
        """ The text appearing in this view's title bar. """
        return self._title

    @property
    def hide_action(self) -> Optional[QAction]:
        """ The UI action that toggles this view's visibility. If None, view cannot be hidden. """
        return self._toggle_hide

    @property
    def is_hidden(self) -> bool:
        """ Is this view currently hidden from the UI? """
        return self.isHidden()

    @property
    def dock_action(self) -> Optional[QAction]:
        """ The UI action that toggles this view's docking status. If None, view cannot be undocked. """
        return self._toggle_dock

    @property
    def is_floated(self) -> bool:
        """ Is this view currently floated as a independent window in the UI? A view could be floated but hidden! """
        return self.parent() is None

    @property
    def dock_panel(self) -> str:
        """
        Identifies which splitter pane in the main window houses this view when docked.
        :return: "left", "mid" or "right"
        """
        return self._dock

    @property
    def dock_position(self) -> int:
        """
        The preferred ordinal position of this view when docked in a splitter pane in the main window.
        :return: The preferred ordinal position.
        """
        return self._dock_pos

    @property
    def window_geometry(self) -> QRect:
        """ Rectangle bounding the view the last time it floated as a top-level window (excluding frame). """
        return self._window_geometry or QRect(60, 60, 600, 400)

    @window_geometry.setter
    def window_geometry(self, r: QRect) -> None:
        self._window_geometry = r

    def moveEvent(self, event: QMoveEvent) -> None:
        """
        When view is floated as a top-level window, this override stores the window geometry (so it can be saved
        in GUI layout preferences at application exit), then forwards the event to the super class.
        """
        if self.is_floated:
            self._window_geometry = self.geometry()
        super().moveEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        When view is floated as a top-level window, this override stores the window geometry (so it can be saved
        in GUI layout preferences at application exit), then forwards the event to the super class.
        """
        if self.is_floated:
            self._window_geometry = self.geometry()
        super().resizeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        When a view is floated as a top-level window, the view's docked title bar containing the tool buttons is hidden,
        and a window title bar contains typical close, minimimize and maximize buttons. This overrides the close button
        action by redocking the view rather than hiding it.

        NOTE: This behavior is problematic at application shutdown, since any floated views will remain because the
        normal re-docking won't work once the main window is gone. It is the application's responsibility to hide all
        floated views during shutdown.
        """
        if self._toggle_dock is not None:
            self._toggle_dock.trigger()
            event.ignore()
        elif self._toggle_hide is not None:
            self._toggle_hide.trigger()
            event.accept()
        else:
            event.accept()

    def show_title_bar(self, show: bool) -> None:
        """
        Show or hide the view's title bar, which includes the tool buttons that control the view's visibility and
        docking status.
        :param show: True to show, False to hide.
        """
        self._title_bar.setVisible(show)
