## XSort User Guide

A Python/Qt application that visualizes and edits the results from spike sorting of neural activity recorded 
extracellularly in the Stephen G. Lisberger lab at Duke University.


### Background
Researchers in the Lisberger and other laboratories use various multi-electrode systems (eg, the [Omniplex Neural 
Recording Data Acquisition System](https://plexon.com)) to record electrical activity in the brain during experiments. They analyze the
recorded analog channel data streams to detect and "sort" the electrical activity into distinct "spike trains", each
assigned to a different "neural unit".

The various spike sorter algorithms employed in the lab often produce spurious output -- "garbage" units triggered by 
system noise, two spike trains that have a very similar spike waveform and really "belong" to the same neural unit,
and so on. With **XSort** you can visually assess and edit the original spike sorter's results.

### Usage
To use the application, select a _working directory_ containing the analog recording source file and the original
spike sorter results file. After analyzing the files and building an internal cache of channel data streams and unit 
metrics (stored in a subfolder within the working directory), **XSort** offers a number of different data 
visualizations.

1. A tabular listing of the neural units, including metrics -- total # of spikes observed, mean firing rate, best
signal-to-noise ratio (across all data channels), the _primary channel_ (on which best SNR was measured), and others.
2. Channel traces for up to 16 analog data channels, highlighting the spikes of each of up to 3 units currently selected
for display in the neural units table, known as the **_display focus list_**.
3. The per-channel mean spike waveforms, or _spike templates_ (10ms, 1ms pre-spike time) for each unit in the display 
focus list.
4. A firing rate-vs-time histogram, inter-spike interval histogram, and autocorrelogram for each unit in the display 
focus list, along with the cross-correllogram of each unit vs another unit in the list.
5. A 3D histogram (rendered as a heatmap) representing the unit's autocorrelogram as a function of instantaneous 
firing rate.
6. A principal component analysis (PCA) scatter plot projecting a unit's spikes into a 2D space defined by the two 
highest-variance (most information) principal components. When the analysis is performed on multiple units, it can help
detect distinct spike trains that really belong to a single unit. When performed on a single neural unit, it can help
determmine whether a unit should be split into two or more distinct spike populations.

You can edit the table of neural units in a number of ways:
1. Attach a short descriptive label (typically, the putative neuron type) to any unit.
2. Delete one or more selected units.
3. Merge two selected units into one.
4. Split the currently selected unit into two distinct units.

**XSort** maintains a complete "edit history" in the current working directory (`.xs.edits.txt`), and you can "undo" any
or all of the changes in reverse order, or undo them all at once to recover the original state of the neural unit table. 
Once happy with any changes made, you can save the neural unit list for use in other applications or your own scripts.



