# Changelog

## v0.0.7 (10/11/2023)
- Turned on antialiasing in the `pyqtgraph` library. Eventually, this may become a user setting, especially if it causes
performance issues.
- Enhancements to `FiringRateView`:
  1) User can opt to display the firing rate histograms as raw bin counts or normalized.
  2) The histograms are now rendered more accurately as "staircases".
  3) Added a vertical-line "time cursor" that follow the mouse as it moves within the view. A label near the bottom 
  shows the elapsed time in the format **MM:SS**.
  4) A colored, labeled marker is drawn where the time cursor intersects each displayed histogram, and the label 
  reflects the approximate Y-coordinate value for that histogram at the elapsed time of the recording.
  5) Both X- and Y-axes are not really needed, so they are hidden.
- Bug fix in `NeuronView`: On Mac OS, when the system switched automatically to a night time color scheme, could no 
longer read the rows in the neural unit table because the background color was hard-coded to white -- and the night 
time default foreground color is also white!

## v0.0.6 (10/09/2023)
- Initial implementation of `FiringRateView`, displaying firing rate over the course of the recording, normalized by
dividing by the overall mean firing rate.
- Split `StatisticsView` into 2 separate views, `ISIView` and `CorrelogramView`. The latter renders the ACGs and CCGs
for the N neurons in the current display list as a NxN matrix of subplots, with the ACGs along the major diagonal.

## v0.0.5 (10/09/2023)
- Initial implementation of `StatisticsView`, displaying ISI histograms and ACGs for all neural units in the current 
display list, as well as the CCG of the first neuron in that list vs the others.
- Fixed bug in assigning display colors to neural units in the display list.
- Fixed bug in code that retrieved channel trace segments whenever user changed the segment start time using the 
slider in the `ChannelView`.
- Modified how changes in the slider position are handled in `TemplateView`, `ChannelView`, and `StatisticsView` since
the `QSlider.sliderReleased` signal did not fire in RHEL.
- Known issues: (1) Keyboard interface to slider operation does not work in Mac OS. (2) The background task which 
generates ACGs and CCGS for neural units in the current display list is computationally intensive and causes noticeable
display lags in the GUI under certain situations. May have to run these in a separate process, or compute once and 
cache to file. File IO retrievals do not block the GUI like heavy-duty computations -- a Python GIL issue.

## v0.0.4 (10/02/2023)
- User can select up to 3 neural units in the NeuronView for inclusion in the **_display list_**, which affects what is
rendered in other views. A display color is assigned to each "slot" in the display list: blue, red, yellow. 
- For each neuron in the display list, `ChannelView` now displays 10-ms clips at every spike occurrence on the displayed
trace segment for that neuron's so-called primary channel (analog channel with the highest estimated SNR for that
neuron). The clips are drawn by tracing over the trace segment in the clip interval using the display color assigned
to the neuron (but translucent).
- A slider in the `ChannelView` lets the user look at any 1-second segment within the Omniplex recording. Furthermore,
the user can use the mouse wheel to zoom in/out and pan the view. Minimum time span (x-axis) is 100ms. Minimum voltage
range is enough to show 2 adjacent channel traces (they're arranged in descending order from top to bottom).
- Initial implementation of `TemplateView`. Displays per-channel spike templates computed for each of units in the 
current display list. Template width set by user, between 3 and 10ms -- saved and restored as a **_view-specific_** 
user setting.

## v0.0.3 (09/25/2023)
- Developed critical background tasks for analyzing the required source files (Omniplex PL2, spike sorter results)
in the current working directory, caching analog channel traces in separates files within the directory, computing 
and caching neural unit metrics.
- Completed initial implementations of the `NeuronView` and `ChannelView`.

## v0.0.2 (09/05/2023)
- Internal release after restructuring the application code several times.
- Added notion of the **_current working directory_**, which is also saved to user settings so that XSort can restore
the working directory at app launch.

## v0.0.1 (08/01/2023)
- Replaced tkinter with Qt/PySide for the GUI framework. Docking/floating functionality much improved.
- Saves and restores main window geometry and dock widgets's state via Qt's preferences support (QSettings).

## v0.0.0 (07/26/2023)

- Skeleton application with no functionality except some blank views that can be docked or floated as 
top-level windows. To test feasibility of using Qt for cross-platform development.