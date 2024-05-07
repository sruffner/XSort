## The Neural Units Table

The _neural units table_, occupying the top left corner of **XSort**'s main window, shows all distinct neural
units defined in the current working directory: all units extracted from the original spike sorter file that you have 
**not** deleted, merged, or split, plus any derived units resulting from merging or splitting (see the **Making 
Changes** section of the user guide).

Each row in the table corresponds to one neural unit, and the columns display numeric metrics or other information:
1. **UID** - The unit ID, typically an integer **N**. For a Purkinje cell spike train extracted from the original spike 
sorter file, two units are created -- '**Nc**' (complex spikes) and '**Ns**' (simple spikes). Any unit derived from a
merge or split operation has the UID '**Nx**', where the 'x' suffix indicates it is a derived unit.
2. **Label** - An optional, short label (typically used to specify the unit's putative neuron type). See **Making
Changes** chapter.
3. **Channel** -- The unit's _primary channel_, ie, the analog channel on which the best signal-to-noise ratio was 
measured.
4. **#Spikes** -- The total number of spikes in the unit's spike train.
5. **Rate (Hz)** -- The unit's mean firing rate.
6. **SNR** -- Highest observed signal-to-noise ratio for the unit across all recorded analog data channels.
7. **Amp (uV)** -- Peak-to-peak amplitude of the unit's mean spike waveform (template) as measured on the primary
channel.
8. **%ISI<1** -- Fraction of interspike intervals (ISI) in this unit's spike train that are less than 1 millisecond. An 
ISI less than the typical refractory period is an indication that some of the spike timestamps attributed to the unit 
are simply noise or should be assigned to another unit.
9. **Similarity** -- The degree of similarity of this unit to the first selected unit in the table -- the _primary 
unit_. For each unit in the table, a 1D sequence is formed by concatenating the unit's per-channel spike templates.
The similarity metric for unit **A** is the cross-correlation cofficient of unit **A's** sequence with the analogous 
sequence for the primary unit. A value of 1 indicates perfect correlation; units may be negatively correlated. Of
course, the similarity metric is unknown if no unit is currently selected in the table.

NOTE: Per-channel spike templates are only computed on the 16 channels "near" a unit's primary channel (**XSort** does 
not yet support probe geometry information, so "near" means channels [P-8 .. P+7], where P is the primary channel
index). This is the unit's _template channel set_. The similarity metric for unit A vs B only includes the templates 
for those channels in the intersection of unit A's template channel set with unit B's. If that intersection is empty,
then the similarity is 0.

### Hiding columns and sorting on a given column

With the exception of the **UID** column, any of the columns in the units table may be hidden. To hide a column, 
right-click anywhere on the table header and uncheck the label of the column you wish to hide. The table updates 
immediately. To unhide a colum, right-click again and select any unchecked column that you wish to show.

Note that **XSort** "remembers" which columns are hidden, storing the column state in the user's settings --
so the same columns will be hidden the next time you run the program.

To sort the table on any column, simply click on that column's header. Click again to sort in reverse order. The 
identity of the sort column is not persisted in user settings; **XSort** always sorts on the **UID** column initially.

### The current display list

The _display list_ is the subset of units currently selected in the neural units table. Up to 3 units may be selected 
for display at any one time for comparison purposes, and a highlight color is assigned to each: blue for the first 
selection, red for the second, and yellow for the third. The same color is used to render the spike templates, 
correlograms, and other statistics for each unit across the various views.

When **XSort** initially loads the content of the current working directory, no units are selected, so most of the views
contain nothing of interest. To select a unit, simply click anywhere on the corresponding row in the table; the previous
selection, if any, is cleared. To select multiple units (up to 3) for display, hold down the `Control` key (the 
`Command` key in MacOS) down while clicking on the relevant rows. You can also use the `Up/Down` arrow keys to change
the identity of the primary unit without using the mouse.

You will notice that the views are updated -- sometimes after a noticeable delay -- to display the relevant metrics or 
statistics for the selected unit(s). Some statistics, once computed, are cached in memory and render quickly -- such as 
the correlograms and spike templates. The principal component analyis, however, must be redone each time the display
list changes. 

Whenever any statistics need to be computed, those calculations happen in the background and the views are updated
as the background task delivers its results. If a background task is still running and you change the display list again 
or actually edit the units table, that task is cancelled and a modal dialog blocks user input until that task stops,
typically in a few seconds or less.

While a recording session may include hundreds of analog channels, the **Templates** view only displays unit spike 
templates on a maximum of 16 channels, and the **Channels** view only displays a maximum of 16 individual analog 
channel traces. The first selection in the display list, the **_primary unit_**, determines the range of channels 
selected: the 16 channels in the neighborhood of the unit's primary channel.
