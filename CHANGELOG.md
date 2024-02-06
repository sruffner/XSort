# Changelog

## v0.1.4 (TBD)
- Finished implementation of the **Help** view, which offers a small user guide for XSort. The guide is divided into
five chapters. A dropdown combo box selects the chapter displayed in the read-only text browers. The chapter contents
are maintained in Markdown files in the `assets` folder.
- Minor changes to `CorrelogramView`: Added a horizontal line at zero correlation and a translucent white vertical band
spanning the time range -1.5 to +1.5 milliseconds in each ACG or CCG plot. A checkbox at the bottom of the view lets
the user toggle the visibility of the zero correlation lines.
- Modified how user selects units for display in the neural units table. Clicking on any row in the table selects the
corresponding unit for display while clearing any previous selection (single-selection behavior). To select multiple
units for display (up to three), you must hold down the `Control` key (`Command` key on MacOS) while clicking on each 
table row to select.
- Updated `setup.cfg` to use ">=" rather than "==" in the `install_requires` section. This is needed for Numpy, in 
particular, as `pip` won't install Numpy <= 1.25 if the Python version is 3.12, as changes made in the 3.12 elease were 
incompatible with Numpy versions prior to 1.26.
- Introduce support for an alternative working directory configuration. Instead of the Omniplex PL2 file, a flat
binary file (extensions `.bin` or `.dat`) can serve as the analog source file. This file contains the analog channel
streams as raw 16-bit signed integer samples. The streams may be stored one after the other or interleaved, and the
raw data may be prefiltered or not. In addition, XSort now handles situations in which the specified working directory
contains multiple Python pickle files and multiple `.pl2/.bin/.dat` files. Whenever the identity of the neural unit
and analog channel data source files is ambiguous, or the analog source requires additional configuration from the
user, XSort raises a modal dialog to request this information from the user.
- Note that, if the analog source is a pre-filtered flat binary file, there is no need to generate the individual
internal analog channel cache files.

## v0.1.3 (01/22/2024)
- Implemented the **Edit|Split** operation. All editing operations are now available.
- Added modal progress dialog to block user input when waiting on a cancelled background task to stop. The dialog
message reads "Please wait..." The time it takes for a background task to detect the cancel request and stop is highly
variable, but will be less than 5 seconds in most cases. The dialog's progress bar animates 0-99% completion in 5 
seconds, even though the wait time is unknown. Most tasks will respond to the cancel signal within that time. If not, 
the progress bar resets to 0% and continues animating. The same dialog will be raised at application exit if the user 
quits while a background task is in progress.
- When the user changes the XSort working directory, tha `BUILDCACHE` background task is launched to examine the
directory contents, cache all Omniplex analog data channel streams in individual binary files, compute metrics 
(per-channel spike templates, best SNR and the analog channel on which that SNR was measured - aka the _primary 
channel_) for each unit defind in the original spike sorter pickle file, and persist unit spike trains and metrics in 
individual unit cache files. For a long recording session with many units, this task takes quite some time, and the 
user can't really do a lot of useful work until it's done. Therefore, a modal progress dialog now blocks user input 
until the `BUILDCACHE` task has finished.
- Fixed calculation of firing rate-vs-time histogram in `Neuron`. When **not** normalized, each bin value is now
divided by the bin size in seconds, so the histogram reflects spikes/second (aka, firing rate) rather than raw spike
count.
- `FiringRateView` modified so that the Y readout value indicates the firing rate at the corresponding time in Hz when
the **Normalized** box is unchecked.
- **BUG FIXED**: `NeuronView` failed to display correctly when the neural unit list contained only a single unit 
(because the sort-by-column algorithm failed to initialize the list of sort indices in this scenario).

## v0.1.2 (01/18/2024)
- `PCAView`: Implemented "lasso" interaction to define a closed polygonal region by clicking on a series of points 
within the view -- in order to select a subset of spikes prior to "splitting" a neural unit. This interaction is
enabled only when a single unit occupies the display focus list, and only when the unit's PCA projection has been 
fully calculated. The "Split" operation is not yet implemented.
- _BUG FIX_: Release 0.1.1 failed to launch because of a configuration error in packaging the wheel file -- the
markdown files were missing from the `assets` folder.

## v0.1.1 (01/12/2024)
- `NeuronView`: Selected columns in the neural units table may be hidden/shown via a popup context menu raised by
clicking anywhere on the table header. The UID column may not be hidden. Column visibility is saved in user settings at
application shutdown and restored from settings at startup. Fixed issue with sorting on UID column.
- `ACGRateView`: Now renders ACG-vs-firing rate histograms as heatmap images.
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
- Implemented **Edit|Delete** and **Edit|Merge** operations, along with **Edit|Undo** and **Edit|Undo All**. If you
perform a "merge", you should wait until the metrics of the merged unit are calculated and cached in the background
before proceeding with further edits.

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