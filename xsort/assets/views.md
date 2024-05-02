## XSort's "Views"

----

### Templates

This view displays the mean spike waveforms, or _spike templates_, of each neural unit in the current display list
on each of up to 16 analog data channels. The templates are laid out in small subplots labeled with the source 
channel index. As in all the data views, the waveform trace color matches the highlight color assigned to the unit:
blue for the first selected unit in the display list -- the _primary unit_ --, red for the second, and yellow for the 
third.

All spike templates are 10 milliseconds in duration, starting 1 ms prior to the spike occurrence time. However, in many
cases, the "interesting" part of the spike waveform may only last a few milliseconds. Use the slider control at the
bottom left of the view to adjust the visible time span from 10 down to as little as 3 ms. The template span is a user
preference that is saved at application exit and restored the next time **XSort** runs.

Whenever the view is updated, the voltage scale (vertical) is automatically adjusted in accordance with the peak-to-peak
amplitude of the primary unit's largest spike template. You can manually adjust the scale between +/-50 and +/-1000uV 
using the slider control at the bottom right.

**NOTE**: When there are more than 16 recorded analog channels, XSort only computes and caches each neural unit's 
per-channel spike templates on the 16 channels [P-8 .. P+7], where P is the index of the unit's primary channel. If two 
different units do not have the same primary channel, their templates are not computed on the same set of channels. By 
design, this view displays all 16 templates for the primary unit, as well any templates for other units in the display
list that were computed on any channel among the primary unit's 16 template channel indices. This makes sense -- it's 
highly unlikely the user will need to compare units with very different template channel sets.

### Correlograms
This view displays the autocorrelogram (ACG) of the spike train for each unit in the current display list, as well as 
the crosscorrelogram (CCG) of each unit's spike train with the spike train of a different unit in the display list. If 
only one unit is selected for display, only its autocorrelogram is shown. If two (or three) units are selected, the 
correlograms appear in a 2x2 (or 3x3) array of subplots, with the autocorrelograms on the main diagonal. Note that
the CCG trace colors are a blend of the colors assigned to the units in the display list (red + blue = magenta, etc).

The correlograms are computed once in a background task but then cached in-memory so they need not be computed again.

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
duration of the recording session. It provides a means of quickly checking whether firing rate changed dramatically
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

#### Algorithm
By design **XSort** only computes mean spike waveforms -- aka _spike templates_ -- on a maximum of 16 channels "near"
the unit's primary channel; call this the unit's _**template channel set**_. If N<=16 analog channels were recorded,
then all units will have the same templates channel set. **PCA is restricted to a unit's template channel set to keep
the computation time and memory usage reasonable.** This is very important for recording sessions with hundreds of
analog channels.

When a single unit is selected for PCA, only the channels in that unit's template channel set are included in the
analysis. When 2 or 3 units are selected for PCA, only those channels comprising the **intersection** of the units'
template channel sets are considered. **If the intersection is empty, then PCA cannot be  performed**.

Let N = N1 + N2 + N3 represent the total number of spikes recorded across all K units (we're assuming K=3 here).
Let the spike clip size be M analog samples long and the number of analog channels included in the analysis be P.
Then every spike may be represented by a _multi-clip_: a vector of length L=MxP, the concatenation of the clips for 
that spike across the P channels. The goal of PCA analysis is to reduce this L-dimensional space down to 2, which can be
easily visualized as a 2D scatter plot.

The first step is to compute the principal components for the N samples in L-dimensional space. A great many of
these clips will be mostly noise -- since, for every spike, we include the clip from each of up to 16 channels, not
just the primary channel for a given unit. So, instead of using a random sampling of individual clips, we use the
spike templates computed on each channel for each unit. The per-channel spike templates are concatenated to form a KxL 
matrix, and principal component analysis yields an Lx2 matrix in which the two columns represent the first 2 principal
components of the data with the greatest variance and therefore most information. **However, if only 1 unit
is included in the analysis, we revert to using a random sampling of individual clips (because we need at
least two samples of the L-dimensional space in order to compute the covariance matrix).**

Then, to compute the PCA projection of unit 1 onto the 2D space defined by these two PCs, we form the N1xL
matrix representing ALL the individual spike multi-clips for that unit, then multiply that by the Lx2 PCA matrix to
yield the N1x2 projection. Similarly for the other units.

[**NOTE**: From this description of the PCA algorithm used, it should be clear that, whenever the composition of the 
display list changes, the principal component analysis must be redone from scratch.]

PCA is a time-consuming operation that is always performed on a background thread. It can take many seconds to complete,
especially if the unit spike trains are very long, and can impact the responsiveness of the user interface (whenever you 
change the composition of the display list, any statistical computations going on in the background must be cancelled 
before the views are updated). **XSort** will not queue a PCA computation at all if the PCA view is hidden -- so it
is best to hide the view until you actually want to see the PCA projections.

When there are a great many points to draw, it takes a noticeable amount of time to render the PCA projection scatter 
plots . And with a great many points and overlapping projections, one unit's projection can obscure another's. At the 
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

This view displays a one-second clip (or shorter) for each of up to 16 analog data channels. When a working directory
is first opened, traces for the first 16 channels (0-15) are shown. Once a unit is selected for display from the
neural units table, then the channels displayed are those from the unit's template channel set (channels [P-8 .. P+7],
where P is the unit's primary channel index). In addition, for each neural unit in the current display list, the view
superimposes -- on the trace for that unit's _primary channel_ -- 10-millisecond clips indicating the occurrence of 
spikes from that unit. In keeping with the other views, these spike clips are rendered in the highlight color (blue, 
red or yellow) assigned to that unit.

A slider at the bottom of the view lets the user choose any 1-second segment over the entire Omniplex recording. The
companion readouts reflect the elapsed recording time -- in the format **MM:SS.mmm** (minutes, seconds, milliseconds) --
at the start and end of the currently visible portion of the traces. 

With the mouse cursor inside the view, use the mouse's scroll wheel to zoom in or out on the plotted traces both 
horizontally in time and vertically in voltage, then hold down the mouse and drag to pan the view in either direction.
The plot's x- and y-axis range limits are configured so the user can zoom in on any 100ms portion of the 1-second 
segment, and on any 2 adjacent channel traces. To reset the view, click the small icon ("A") located in the lower-left 
corner of the view.

Each time the channel traces are updated, **XSort** adjusts the vertical scale in accordance with the worst-case 
peak-to-peak voltage excursion observed across all the traces.You can manually adjust the scale between +/-50 and 
+/-1000uV using the slider control at the bottom right.

