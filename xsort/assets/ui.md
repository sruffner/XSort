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
8. **Help** (houses this guide)

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
When you select a _working directory_, **XSort** examines the directory contents to verify it contains the required 
data files.
- Analog source file:
    - An Omniplex recording session (`.pl2` file), OR 
    - A flat binary file (`.bin` or `.dat`) containing the analog channel data streams, interleaved or not, and
      prefiltered or not (`int16` samples).
- Neural unit soure file:
    - A Python pickle file (`.pkl` or `.pickle`) containing the spike times for each neural unit extracted from the 
      analog channel data via spike sorting applications employed in the Lisberger lab.

Future versions of **XSort** may support additional source file formats.

If there are multiple analog source files or multiple unit source files, then **XSort** will ask you to select which 
files to use. If the analog source is a flat binary file, you must also specify the number of channels recorded, the 
sampling rate in Hz, the scale factor converting raw 16-bit samples to microvolts, whether the data is prefiltered, and 
whether the individual analog channel streams are interleaved in the file. **XSort** saves the directory configuration 
information to a file (`.xs.directory.txt`) within the directory itself -- so you'll only need to provide it the first 
time you "open" a working directory.

If the working directory is invalid or if it cannot read the analog or unit data source file, **XSort** will ask you to
select another directory. Note that, when you launch **XSort** for the first time, you must specify a valid working 
directory. To switch to a different directory, use the **File | Open** menu command. **XSort** stores the current 
working directory's path in your preferences file on exit, so the next time you run the application it will open that 
directory initially.

### Internal cache files
**XSort** must process possibly hundreds of neural units recorded across hundreds of analog channels at sampling rates 
of 30KHz or more. The analog source file size will typically be in the tens of gigabytes, and the analog data may need 
to be bandpass-filtered. The **XSort** visualizations require expedient access to the **filtered** analog channel 
streams and to certain neural unit metrics, particularly a unit's primary channel and spike templates on the 16 channels
"in the neighborhood" of that channel.

To that end, when **XSort** first opens a working directory, it will build a set of internal cache files. All of these
cache files are stored in the `.xs-cache` subfolder under the working directory (so it is a simple task to manually
remove the internal cache!):
- `.xs.ch.<N>`: Contains the bandpass-filtered data stream recorded on analog channel `<N>`. If the analog source is a
  prefiltered flat binary file, channel caching is unnecessary.
- `.xs.noise`: Contains the estimated noise level on each analog channel.
- `.xs.unit.<UID>`: Contains the original spike train and computed metrics for neural unit `<UID>`.

Building the internal cache can take several minutes when there are hundreds of analog channels and hundreds of neural 
units. But once it is complete, **XSort** will "load" the directory very quickly the next time you open it -- so long as
you do not delete the cache!

This internal cache is essential to **XSort** so that it can quickly load short segments of the analog channel data 
streams at any point along the recorded timeline, quickly display spike templates for any selected unit, and use those
templates for principal component analysis without having to recompute them every time. You cannot really do much "work"
with the application until the cache is generated, so a modal progress dialog blocks further user input and displays
progress messages as the cache generation task proceeds in the background.

Once the internal cache has been generated for a working directory, switching to that directory is much faster -- the
background task merely loads the neural unit metrics and the 1-sec channel trace segments from the individual cache
files, which typically takes less than a second on a reasonably up-to-date system.

### Comparing neural units
Whenever you change the working directory -- and after the cache generation task has finished --, the **Channels** view 
will show one-second segments from each of the first 16 analog channels and the other views will be essentially 
blank. These views display statistical graphs for one or more units selected from the units table -- the _display focus
list_.

When you select a unit in the table, the corresponding row is highlighted, and the various views are gradually updated
to display per-channel spike templates, an autocorrelogram, firing rate-vs-time histogram, and other statistics for that
unit. You will notice that some statistics are not available immediately but must be computed in the background. The 
user interface is not blocked in this case, but you will see progress messages posted in the status bar, and the 
various views will be updated once the relevant statistics are ready for display.

For more information on the units table, the display focus list, and the various views, see the sections **Neural Units 
Table** and **Views** in this user guide.

### User preferences
**XSort** saves various user-specific application settings in a dedicated file, `.XSort.ini`, in your home directory:
- Settings that specify the layout of the application window: size and position, visibility and docking state of the
various views.
- Various view-specific settings.
- File paths for the most recently visited working directories.

Two additional settings are exposed in a **Preferences** dialog (`File | Preferences`):
1. _Internal cache removal policy_. The internal cache files generated by XSort can take up many gigabytes of disk 
space, particularly if analog channel caching is necessary. If disk space is limited on your system, you can change
the application's cache removal policy to **Always** (every working directory visited is marked for cache removal) or
**LRU** (a working directory is marked for cache removal when it is evicted from your most recently visited list, and 
unmarked if it is put back on that list). The default policy is **Never** (cache removal disabled). Note that the
working directory's edit history must be removed when the internal cache is destroyed, since derived units are only
defined in the internal cache). Cache removal does not happen until application shutdown.
2. A set of suggested label string for auto-completing a neural unit label. You can freely add or remove entries from 
the suggestion list.



