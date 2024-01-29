## XSort's "Views"

----

### Templates

This view displays the mean spike waveforms, or templates, computed for each neural unit in the current display list
on each recorded Omniplex analog data channel. The templates are laid out in small subplots labeled with the source 
channel index. As in all the data views, the waveform trace color matches the highlight color assigned to the unit:
blue for the first selected unit in the display list, red for the second, and yellow for the third.

All spike templates are 10 milliseconds in duration, starting 1 ms prior to the spike occurrence time. However, in many
cases, the "interesting" part of the spike waveform may only last a few milliseconds. Use the slider control at the
bottom of the view to adjust the visible time span from 10 down to as little as 3 ms. The template span is a user
preference that is saved at application exit and restored the next time **XSort** runs.

### Correlograms

This view displays the autocorrelogram (ACG) of the spike train for each unit in the current display list, as well as 
the crosscorrelogram (CCG) of each unit's spike train with the spike train of a different unit in the display list. If 
only one unit is selected for display, only its autocorrelogram is showwn. If two (or three) units are selected, the 
correlograms appear in a 2x2 (or 3x3) array of subplots, with the autocorrelograms on the main diagonal. Note that
the CCG trace colors are a blend of the colors assigned to the units in the display list (red + blue = magenta, etc).

The correlograms are computed once on a background task but then cached in-memory so they need not be computed again.

As with the **Templates** view, you can "zoom" in on the correlograms; use the slider at the bottom of the view to 
adjust the correlogram span between 20 and 200 milliseconds (+/-10ms to +/-100ms).

The plots include two annoations: a dashed white line at zero correlation and a translucent white vertical
band spanning T=-1.5 to +1.5 milliseconds. These visual guides help the user quickly check whether or not there is 
significant correlation in pairs of spikes occurring 1ms apart -- an indication that the neural unit's spike train
is contaminated in some way. The "Show zero correlation" checkbox toggles the visibility of the zero correlation line.

Both the visible state of the zero correlation marker and the current correlogram span are user preferences that are
saved at application exit and restored the next time **XSort** launches.

### Interspike Interval Histograms

This view plots, on the same axes, the interspike interval (ISI) histogram for each unit in the display list. The 
histogram is an array of normalized bin counts for ISIs between 0 and 200 milliseconds, with each bin count divided by 
the maximum observed bin count. As with the correlograms, the ISI histograms are computed once on a background task, 
then cached in memory until the working directory is changed or the application exits.

Use the slider at the bottom of the view to adjust the visible histogram span between 20 and 200 milliseconds.

### ACG-vs-Firing Rate

This view renders, for each unit in the display list, a 3D histogram representing the unit's autocorrelogram as a 
function of instantaneous firing rate. This can be thought of as a "3D autocorrelogram" that shows firing regularity
when the unit is firing at different rates. The histograms are rendered as 10x201 heatmaps, with observed firing rate 
divided into 10 equal bins along the vertical axis, and time T relative to spike occurrence along the horizontal axis, 
with T between -100 and +100 ms. The number of firing rate bins and the correlogram span are fixed and cannot be 
changed. The subplot title displays the unit UID and the observed range of the instantaneous firing rate for that unit.

Like the ACGs, CCGs, and ISI histograms, the ACG-vs-firing rate histogram is computed once in the background and then
cached in memory.

### Firing Rate

This view displays, for each unit in the current display list, a histogram of firing rate as a function of time over the
duration of the Omniplex recording. It provides a means of quickly checking whether firing rate changed dramatically
at any point during the recording (eg, if the unit was "lost" at some point during the experiment).

The firing rate histograms are drawn on a time scale spanning the entire recording, which may last an hour or more.
Use the combo box at the bottom of the view to select a bin size between 20 and 300 seconds. If the **Normalized** box 
is unchecked, the Y-axis is firing rate in Hz. If checked, the histograms are normalized by dividing each bin by the
unit's mean firing rate.

As you move the cursor across this view, a white vertical line follows the cursor, and a text label near the top of
the line displays the elapsed time of the recording at that point in minutes and seconds -- **MM:SS**. A dot marks
the interaction of that line with the firing rate histogram, and an accompanying label displays the normalized ("1.08")
or unnormalized ("24.3 Hz") firing rate for the corresponding bin.

A dashed green vertical line indicates the elapsed recording time **T** at which the analog data clips start in the 
**Channels** view. If you change the start time **T** in the **Channels** view, this green line is updated accordingly. 
Conversely, you can change the clip start time by clicking anywhere inside this view. This is helpful if the firing
rate histograms unveil some sort of anomaly and you wish to see what's happening at that point in time in the analog 
data recording.

### Principal Component Analysis (PCA)

This view renders the results of a principal component analysis (PCA) on the spike clips of each neural unit in the
current display list. The purpose of the analysis is to "map" each spike in each unit's spike train to a point in a 
two-dimensional space, offering a "scatter plot" to help assess whether the units are truly distinct from each other.
The **PCA** view renders each unit's scatter plot in the assigned color (blue, red, yellow), provides for downsampling
of the scatter plots when the unit(s) contains hundreds of thousands of points, and allows the user to define a _split
region_ in PCA space (via a sequence of mouse clicks inside the view) when only a single unit occupies the display list.

PCA is a time-consuming operation that is always performed on a background thread. It can take many seconds to complete,
especially if the unit spike trains are very long. In the analysis approach, each spike is represented by a 2-ms clip 
"centered" on the spike occurrence time. For each spike, one such clip is extracted from the `M` Omniplex analog 
channels recorded, and the clips are concatenated end-to-end, yielding a `P = 2M` _multi-clip_ for each spike. The goal 
of PCA is to reduce the dimension `P` to 2 -- so that each spike is represented by a single point in 2D space.

If there are 3 units in the focus list, there are a total of `N = N1 + N2 + N3` multi-clips of length `P` samples. The
first step in PCA is to form the `3xP` matrix containing the `P`-millisecond multi-clips formed by concatenating the
first two milliseconds of each unit's `M` per-channels spike templates, then find the eigenvalues/vectors for the 
matrix. The eigenvectors associated with the two highest eigenvalues represent the two directions in `P`-space along 
which there's the most variance in the dataset -- these are the two principal components which preserve the most 
information in the original data. A `Px2` matrix is formed from these two eigenvectors. (In a prior version, we selected
a random sampling of 1000 multi-clips across the 3 units -- in proportion to each unit's contribution to the total 
number of clips, resulting in a 1000xP matrix. But the results were unsatisfactory because many of the clips were mostly
noise.) In the second, longer step, all `N` multi-clips are concatenated to form a `NxP` matrix, which is multiplied by 
the `Px2` principal component matrix (this is done in smaller chunks to conserve memory) to generate the projection of 
each unit's spikes onto the 2D plane defined by the two principal components.

[**NOTE**: From this description of the PCA algorithm used, it should be clear that, whenever the composition of the 
display list changes, the principal component analysis must be redone from scratch.]

When there are a great many points to draw, it takes a noticeable amount of time to render the scatter plots in the 
PCA view. And with a great many points and overlapping projections, one unit's projection can obscure another's. At the 
bottom of the view are two controls to help address these issues:
- Pressing the **Toggle Z Order** button changes the Z-order of the displayed scatter plots when more than one unit
is in the display list.
- The **Downsample** combo box lets the user choose the downsampling factor for the rendered plots. The higher the 
value, the faster the scatter plots will render -- but only because fewer points are drawn. Set this to 1 to disable
downsampling.

#### Defining the split region

The main purpose of the **PCA** view is to identify a neural unit that is really a collection of two populations of 
spikes belonging to two distinct neurons. For such a unit, the PCA scatter plot might consist of two physically distinct 
"blobs". When a single unit occupies the display list and its PCA projection has been fully calculated, you can use the
mouse to define a closed polygonal region around one of the blobs -- the _split region_. Simply click inside the view
to start the polygon path; once you relese the mouse button and move it (don't hold and drag), a thin white line will 
follow the cursor as it moves within the view. Click again to define a second point in the polygon, and repeat this 
process until you've surrounded the "blob" you wish to isolate. Double-click to define the last point and close the 
polygon region. You can always start over if you've missed some points. When satisfied, choose the **Edit | Split** menu
command to split the neural unit into two derived units -- one containing the collection of spikes that project inside 
the polygonal split region, and the other formed from the spikes that lie outside it.

See the **Making Changes** section of the user guide for additional details about splitting a unit, as well as merging
two units, deleting a unit, or undoing those changes.

### Channels

This view displays a one-second clip (or shorter) for each of the analog data channels recorded in the Omniplex file
in the current working directory. For each neural unit in the current display list, the view superimposes -- on the 
trace for that unit's _primary channel_ -- 10-millisecond clips indicating the occurrence of spikes from that unit. In 
keeping with the other views, these spike clips are rendered in the highlight color (blue, red or yellow) assigned to
that unit.

A slider at the bottom of the view lets the user choose any 1-second segment over the entire Omniplex recording. The
companion readouts reflect the elapsed recording time -- in the format **MM:SS.mmm** (minutes, seconds, milliseconds) --
at the start and end of the currently visible portion of the traces. 

With the mouse cursor inside the view, use the mouse's scroll wheel to zoom in or out on the plotted traces both 
horizontally in time and vertically in voltage, then hold down the mouse and drag to pan the view in either directions.
The plot's x- and y-axis range limits are configured so the user can zoom in on any 100ms portion of the 1-second 
segment, and on any 2 adjacent channel traces. To reset the view, click the small icon ("A") located in the lower-left 
corner of the view.

