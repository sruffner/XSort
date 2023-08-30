import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# TODO: Python path hack. I cannot figure out how to organize program structure so I can run app.py from PyCharm IDE
#  and also package Xsort for distribution and running the program via python -m xsort.app. I hate this.
p = Path(__file__).parent
sys.path.append(str(p.absolute()))

from xsort.mainwindow import XSortMainWindow

if __name__ == "__main__":
    main_app = QApplication(sys.argv)
    main_window = XSortMainWindow(main_app)
    main_window.show()
    exit_code = main_app.exec()
    # Any after-exit tasks can go here (should not take too long!)
    sys.exit(exit_code)
