from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout

from xsort.data.analyzer import Analyzer
from xsort.views.baseview import BaseView


class PCAView(BaseView):
    """ TODO: UNDER DEV """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('PCA', None, data_manager)

        label = QLabel(f"{self.title}: Not yet implemented.")
        label.setMargin(1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.view_container.setLayout(main_layout)

    def on_working_directory_changed(self) -> None:
        print(f"{self.title}: Got working_directory_changed_signal")
