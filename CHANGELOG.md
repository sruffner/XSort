# Changelog

## v0.1.1 (TBD)
- `NeuronView`: Selected columns in the neural units table may be hidden/shown via a popup context menu raised by
clicking anywhere on the table header. The UID column may not be hidden. Column visibility is saved in user settings at
application shutdown and restored from settings at startup.
- Added an "About XSort" dialog.
- Created a skeleton "user guide" which appears in a dock widget like the other XSort views. Contents TBD.
- Added user-editable `label` property to `Neuron`, distinct from its `uid`. The label string is restricted to 25 
characters or less and may not contain a comma.`NeuronView` modified to display the unit labels in a **_Label_** 
column. Users can edit each unit's label "in place".
- Developed infrastructure for editing the neural list. Supported operations include: edit a unit label, delete a unit,
or merge two units. A future release will add support for splitting a unit in two by "lassoing" a subset of that unit's
spikes in the `PCAView`. Each individual edit is saved to an "edit history" for the neural unit list. This history is
persisted in a dedicated CSV file in the current XSort working directory prior to switching directories or exiting the
application. Upon returning to a given working directory, the initial state of the neural list is read from the original
spike sorter file, then that list is updated by applying each of the edits in the saved edit history, in order.
- Units created by merging or splitting will have a UID ending in the letter 'x'. The integer index assigned to a unit
is incremented each time a unit is created by merging or splitting, thereby ensuring that unit UIDs are always unique.

## v0.1.0 (11/28/2023)
- Defined similarity metric: The correlation coefficient of two vectors: the horizontal concatenation of the 
per-channel spike template waveforms for one unit and similarly for the other unit. 
- Updated `NeuronView` to include the similarity metric in neural unit table. The metric always compares each unit to
the so-called **_primary neuron_**, ie, the first neuron in the current display/focus list. If no units are currently
selected, then the similarity metric is undefined, and all entries in the `Similarity` column are blank. As with any
column in the table, you can sort on the new column. When similarity is undefined, sorting on that column is the same 
as sorting on the UID.
- Removed `UMAPView`. UMAP (Uniform Manifold Approximation and Projection) analysis is too slow. May revisit this 
decision in a future release.
- Initial implementation of `ACGRateView`, which plots the autocorrelogram as a function of firing rate for each of
the units in the current display/focus list.
- Project dependencies now maintained in `requirements.txt`.

## v0.0.9 (11/20/2023)
- Modified approach to principal component analysis for the `PCAView`. Instead of using a random sampling of 1000
spike multi-clips (horizontal concatenation of the 2ms spike clips recorded on P analog channels) across all units,
we use the spike template waveform clips (first 2ms), resulting in a `Kx(MxP)` matrix where K is the number of units
and M is the number of analog samples in 2ms. The PCA projections for each unit's spike trains are then computed
as before. This achieves better separation of the unit projections in PCAView. Hoever, if only one unit is in the
focus list, we revert to using the random sampling of individual clips to calculate principal components.

## v0.0.8 (11/15/2023)
- Requires Python 3.9+ instead of 3.11+ (although still building on 3.11).
- A dashed green line in `FiringRateView` indicates the elapsed time at which the channel trace segments start in
the `ChannelView`. 
- Clicking anywhere on the `FiringRateView` changes the segment start time to the time under the cursor; the 
`ChannelView` updates accordingly.
- Developed code to perform principal component analysis on the spike trains of the neural units currently comprising
the **_focus list_**. PCA computations take a while -- particularly when the total number of spikes processed reaches
100K or more.
- Every time the user changes the focus list, a new background task is launched to compute statistics (ISI, ACG, CCG,
PCA) for the units in the list. The ISI, ACG and CCG are only computed once and cached in-memory in the relevant
`Neuron` objects. The PCA projections are redone each time the focus list changes.
- Since the user may change the focus list rapidly and the statistics calculations are time-consuming, XSort will
cancel a running task of a given type before launching a new one. This strategy also applies if the user opens a new
working directory, then after a short while decides to switch again.
- The results of the PCA analysis are now rendered in the `PCAView`. The view includes a pushbutton to toggle the
display order of the scatter plots, and a combox box to set the downsampling factor in order to speed up rendering.
- The `SimilarityView` was eliminated. Once we have a similarity metric defined, that will be included as a column in 
the `NeuronView`.
- Moved `README` and `CHANGELOG` from `$PROJECT_ROOT/dist` to `$PROJECT_ROOT/` so that Gitlab picks up the `README`.

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