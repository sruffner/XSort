## Using XSort

**XSort** has a highly customizable user interface. A listing of all neural units found in the current working
directory -- the _neural units table_ -- is anchored to the top-left corner of the main application window, while
various data visualizations are housed in dockable, reconfigurable views. See the relevant sections of this guide for a
full description of the units table and the views:
1. **Templates**
2. **Correlograms**
3. **Interspike Interval Histograms**
4. **ACG-vs-Firing Rate**
5. **Firing Rate**
6. **PCA** (Principal Component Analysis)
7. **Channels**
8. **Help** (houses this user guide!)

Each of the views may be "floated" as a top-level window, docked to the right or bottom edge of the main application 
window, or hidden entirely. In a multi-monitor setup, you should be able to move any floating view onto a second screen.
You can also organize several docked views within a single tabbed panel. **XSort** saves the current layout (size,
docking location, visibility) of the main window and its views in a user preferences file (`.XSort.ini` in the user's 
home directory) on shutdown and restores that layout the next time you launch the application.

When docked, the views include a title bar with two pushbuttons -- to either hide or undock the view. Once hidden,
you can show the view by checking the corresponding item in the **View** menu. To move a docked view, "grab" its title
bar and drag it to a new location. As you drag the view around the main window, transient animations hint at where the
view will end up if you release the mouse (it may take a little practice to get the hang of it). Drag the view on top of
another docked view to create a tabbed panel containing both views. 

Individual docked views are separated from each other and the units table by horizontal or vertical "splitters". Grab 
and drag the splitter to change how much horizontal or vertical space is allocated to the docked components.

### Selecting the working directory

**XSort** maintains the notion of a current _working directory_ which must contain the following files:
1. A **single** Omniplex PL2 file in which the Omniplex system's analog channel data streams are recorded.
2. A **single** Python pickle file containing the original spike sorter results in a particular format

When you launch **XSort** for the first time, you will have to specify a valid working directory. To switch to a 
different directory, use the **File | Open** menu command. **XSort** will scan the selected directory, read the header 
and file structure metadata from the PL2 file, and load all neural units defined in the pickle file. If an error occurs,
you will be asked to specify another directory.

**XSort** stores the working directory's path in your preference file on exit, so the next time you run the application it
will open that directory initially.

When **XSort** opens a working directory for the first time, it must do a lot of work initially:
- Extract each recorded analog data channel stream from the PL2 file and cache it in separate flat binary file. These 
channel cache files are written to the working directory and are named `.xs.ch.N`, where `N` is the data channel index.
- Calculate the mean spike waveform or "template" for each unit on each recorded analog data channel, and cache the
unit's spike train, templates, and other metrics in a "unit cache file" in the working directory: `.xs.unit.uid`, where
`uid` is the unique ID assigned to the neural unit.

This internal cache is essential to **XSort** so that it can quickly load short segments of the analog channel data 
streams at any point along the recorded timeline, quickly display spike templates for any selected unit, and use those
templates for principal component analysis without having to recompute them every time. You cannot really do much "work"
with the application until the cache is generated, so a modal progress dialog blocks further user input and displays
progress messages as the cache generation task proceeds on a background thread. 

Once the internal cache has been generated for a working directory, switching to that directory is much faster -- the
background task merely loads the neural unit metrics and the 1-sec channel trace segments from the individual cache
files, which typically takes less than a second on a reasonably up-to-date system.

### Comparing neural units

Whenever you change the working directory -- and after the cache generation task has finished --, the **Channels** view 
will show one-second segments from each of the recorded Omniplex analog channels and the other views will be essentially 
blank. These views display statistical graphs for one or more units selected from the units table -- the _display list_.

When you select a unit in the table, the corresponding row is highlighted, and the various views are gradually updated
to display per-channel spike templates, an autocorrelogram, firing rate-vs-time histogram, and other statistics for that
unit. You will notice that some statistics are not available immediately but must be computed on a background thread. 
The user interface is not blocked in this case, but you will see progress messages posted in the status bar, and the 
various views will be updated once the relevant statistics are ready for display.

For more information on the units table, the display list, and the various views, see the sections **Neural Units 
Table** and **Views** in this user guide



