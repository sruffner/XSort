# XSort

A Python/Qt application that visualizes and edits the results from spike sorting of Omniplex-recorded
data in the Lisberger lab.

## Background
Researchers in the Lisberger laboratory use the [Omniplex Neural Recording Data Acquisition 
System](https://plexon.com) to record electrical activity in the brain during experiments. They analyze the Omniplex's 
recorded analog channel data streams to detect and "sort" the electrical activity into distinct "spike trains", each
assigned to a different "neural unit".

The various spike-sorter algorithms employed in the lab often produce spurious results -- "garbage" units triggered by 
system noise, two spike trains that have a very similar spike waveform and really "belong" to the same neural unit,
and so on. With `XSort` you can visually assess and edit the original spike-sorter output.

To use the application, select a _working directory containing the Omniplex PL2 recording (`.pl2` file) and the 
original spike sorter results (a Python pickle file, `.pkl`). After analying the files and building an internal cache 
of channel data streams and unit metrics (in individual files stored in the working directory), 
`XSort` offers a number of different data visualizations.

1. A tabular listing of the neural units, including metrics -- total # of spikes observed, mean firing rate, best
signal-to-noise ratio (across all data channels), the primary data channel (on which best SNR was measured), and others.
2. Channel traces for each Omniplex analog data channel, highlighting the spikes of each of up to 3 units currently 
selected for display in the neural unit list, known as the **_display list_**.
3. The per-data channel spike template waveforms (10ms, 1ms pre-spike time) for each unit in the display list.
4. A firing rate-vs-time histogram, inter-spike interval histogram, and autocorrelogram for each unit in the display 
list, along with the cross-correllogram of each unit in the display list vs another unit in the list.
5. A 3D histogram (rendered as a heatmap) representing the unit's autocorrelogram as a function of instantaneous 
firing rate.
6. A principal component analysis (PCA) scatter plot projecting s a unit's spikes into a 2D space defined by the two 
highest-variance (most information) principal components. When the analysis is performed on multiple units, it can help
to detect distinct units that are really part of the same unit. When performed on a single neural unit, it can help 
detect a unit that should be split into two or more distinct spike populations.

You can edit the list of neural units in a number of ways:
1. Attach a short descriptive label (typically, the putative neuron type) to any unit.
2. Delete a selected unit.
3. Merge two selected units into one.
4. Split the currently selected unit into two distinct units.

`XSort` maintains a complete "edit history" in the current working directory, and you can "undo" any or all of the 
changes in reverse order, or undo them all at once to recover the original state of the neural list. Once happy with 
any changes made, you can save the neural list for use in other applications or your own scripts.


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
development of `XSort`. Developed with funding provided by the Stephen G. Lisberger laboratory in the Department of
Neuroscience at Duke University.

In addition to the Python standard library, the `XSort` user interface is built upon the Qt for Python framework, 
[PySide6](https://doc.qt.io/qtforpython-6/index.html), data analysis routines employ the [Numpy](https://numpy.org/) 
and [SciPy](https://scipy.org/) libraries, and graphical plots are drawn using [PyQtGraph](https://pyqtgraph.readthedocs.io/en/latest/index.html).
