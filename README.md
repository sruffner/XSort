# XSort

A Python/Qt application that visualizes and edits the results from spike sorting of Omniplex-recorded
data in the Lisberger lab.

## Background
TODO

## Installation (for MacOS/Linux)
- Ensure that Python 3.9+ is installed on your system. We currently build the package against
version 3.11.4. Contact the developer if you run into any issues running the program on 3.9.x or 3.10.x.
- Download the wheel file `xsort-x.y.z-py3-none-any.whl`, where `x.y.x` is the release version number.
- In a terminal console, navigate to the directory holding the wheel file you downloaded, and install 
the package: `pip install xsort-x.y.z-py3-none-any.whl`.
- To start the app: `python -m xsort.app`.
- You may run into the following error when starting XSort: `qt.qpa.plugin: Could not load the Qt platform 
plugin "xcb" in "" even though it was found. This application failed to start because no Qt platform plugin 
could be initialized. Reinstalling the application may fix this problem. Available platform plugins are: 
minimalegl, minimal, wayland-egl, vkkhrdisplay, offscreen, eglfs, vnc, linuxfb, xcb, wayland.` If so, try
reinstalling XCB platform plugin with `sudo apt-get install '*libxcb*'`

## License
`XSort` was created by [Scott Ruffner](mailto:sruffner@srscicomp.com). It is
licensed under the terms of the MIT license.

## Credits
Based on a program developed by David J Herzfeld, who provided critical guidance and feedback during the
development of XSort. Developed with funding provided by the Stephen G. Lisberger laboratory in the Department of
Neuroscience at Duke University.

In addition to the Python standard library, the XSort user interface is built upon the Qt for Python framework, 
[PySide6](https://doc.qt.io/qtforpython-6/index.html), data analysis routines employ the [Numpy](https://numpy.org/) 
and [SciPy](https://scipy.org/) libraries, and graphical plots are drawn using [PyQtGraph](https://pyqtgraph.readthedocs.io/en/latest/index.html).
