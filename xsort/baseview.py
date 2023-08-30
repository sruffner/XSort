from typing import Optional

from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QTableView, QHeaderView, QHBoxLayout, QSizePolicy

from xsort.analysis import Analyzer


class BaseView(QWidget):
    """
    TODO: UNDER DEV
    """

    def __init__(self, uid: int, title: str, controller: Analyzer, background: Optional[QColor] = None) -> None:
        """
        Create an empty view with the specified title.

        :param uid: The view's identifier.
        :param title: The view's title.
        """
        super(BaseView, self).__init__()

        self._uid = uid
        """ The view's identifier. """
        self._title = title
        """ The view's title. """
        self._controller = controller
        """ The master view controller. """

        if isinstance(background, QColor):
            self.setAutoFillBackground(True)
            palette = self.palette()
            palette.setColor(QPalette.Window, background)
            self.setPalette(palette)
        self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(QSize(100, 100))
        self.setWindowTitle(title)

        self.layout_view_content()

        self._controller.working_directory_changed.connect(self.on_working_directory_changed)

    @property
    def uid(self) -> int:
        """ The view's ID. """
        return self._uid

    @property
    def title(self) -> str:
        """ The view's title. """
        return self._title

    @property
    def view_controller(self) -> Analyzer:
        """ The master view controller. """
        return self._controller

    def layout_view_content(self) -> None:
        """ Create and layout the widgets comprising this view. """
        label = QLabel(f"{self.title} view NOT YET IMPLEMENTED.")
        label.setMargin(1)

        layout = QVBoxLayout()
        layout.addWidget(label, alignment=Qt.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    @Slot()
    def on_working_directory_changed(self) -> None:
        print(f"{self._title}: Got working_directory_changed signal!")


class NeuronsView(BaseView):
    """ TODO: UNDER DEV """

    def __init__(self, uid: int, controller: Analyzer) -> None:
        super(NeuronsView, self).__init__(uid=uid, title='Neurons', controller=controller, background=QColor("red"))
        self._neuron_table_model = None
        """ The neuron table model. """
        self._table_view = QTableView()
        """ Table view displaying the neuron table (read-only). """
    def layout_view_content(self) -> None:
        print("Laying out NeuronsView....")
        self._neuron_table_model = self.view_controller.neuron_table_model

        self._table_view.setModel(self._neuron_table_model)
        hdr = self._table_view.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table_view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        main_layout = QHBoxLayout()
        size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        size.setHorizontalStretch(1)
        self._table_view.setSizePolicy(size)
        main_layout.addWidget(self._table_view)

        self.setLayout(main_layout)

    @Slot()
    def on_working_directory_changed(self) -> None:
        print(f"{self.title}: Got working_directory_changed_signal, table_view={self._table_view}")
        self._neuron_table_model = self.view_controller.neuron_table_model
        self._table_view.setModel(self._neuron_table_model)
