# Changelog

## v0.0.4 (10/02/2023)
- User can select up to 3 neural units in the NeuronView for inclusion in the **_display list_**, which affects what is
rendered in other views. A display color is assigned to each "slot" in the display list: blue, red, yellow. 
- For each neuron in the display list, `ChannelView` now displays 10-ms clips at every spike occurrence on the displayed
trace segment for that neuron's so-called primary channel (analog channel with the highest estimated SNR for that
neuron). The clips are drawn by tracing over the trace segment in the clip interval using the display color assigned
to the neuron (but translucent).
- A slider in the ChannelView lets the user look at any 1-second segment within the Omniplex recording. Furthermore,
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