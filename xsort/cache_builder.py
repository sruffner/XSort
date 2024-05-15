import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from PySide6.QtCore import QThread, Slot

from xsort.data.files import WorkingDirectory
from xsort.data.taskmanager import TaskManager

_USAGE_LINES: List[str] = [
    'USAGE:',
    '   python -m xsort.app P [file1 [file2 [R N V interleaved prefiltered]]]\n',
    'DESCRIPTION:',
    '   When the GUI version of XSort opens a working directory for the first time, it must examine the required',
    '   analog and unit data sources and build various internal cache files to optimize performance. This cache build',
    '   phase can take several minutes for long recording sessions with hundreds of analog channels and hundreds of',
    '   neural units. Alternatively, you can launch this console-only version to prebuild the internal cache for any',
    '   specified directory.\n',
    'ARGUMENTS:',
    '   P - Path to working directory (relative or absolute), "." (current directory), or "?" (help)',
    '   file1, file2 - Files in working directory that serve as the analog and unit data source files. The analog data',
    '      file must be an Omniplex PL2 file or a flat binary file (.bin or .dat; 16-bit samples). The unit source',
    '      must be a Python pickle file.',
    '   R - Analog sampling rate in Hz (int) for flat binary file.',
    '   N - Number of analog channel streams stored in flat binary file (int).',
    '   V - Voltage scaling for raw samples in flat binary file (float). Raw sample * V * 1e-7 = sample in microvolts.',
    '   interleaved - 1 if channel data is interleaved in flat binary file; else non-interleaved.',
    '   prefiltered - 1 if channel data in flat binary file is already bandpass-filtered; else unfiltered. If analog',
    '      source is prefiltered, XSort does not generate individual channel cache files.\n',
    'EXAMPLES:',
    '   python -m xsort.app ? : Prints this usage information.',
    '   python -m xsort.app session : Process session directory, which must contain a single PL2 file as the analog',
    '      source and a single pickle file as the unit data source',
    '   python -m xsort.app session analog.pl2 units.pkl : Process session directory using analog.pl2 as the analog',
    '      source and units.pkl as the unit source',
    '   python -m xsort.app session 30000 200 23.4 1 1 : Process session directory, which contains a single flat',
    '      binary file as the analog source (interleaved and prefiltered, 30KHz, 200 channels, 23.4e-7 scale) and a',
    '      single pickle file as the unit source',
    '   python -m xsort.app session a.dat 30000 200 23.4 0 1 : Process session directory containing a single pickle',
    '      file as the unit source, a.dat as the non-interleaved prefiltered flat binary analog source.'
]


class CacheBuilder(QThread):
    """
    This thread object encapsulates the main thread of execution when XSort is launched as a console-only application,
    with at least one command-line argument.

    This implements a command-line XSort utility that takes a file system directory and possibly other arguments,
    verifies that the arguments identify an existing directory containing the analog and unit data source files
    required by XSort, then runs the background task to build the internal channel cache and unit metrics files for
    that directory.

    The user can see usage information by running 'python -m xsort.app ?'.
    """

    def __init__(self, args: List[str], parent=None):
        """
        Construct XSort's cache builder command-line utility.
        :param args: The command-line arguments (after the app name itself).
        """
        super(CacheBuilder, self).__init__(parent)
        self._args = [a for a in args[1:]]
        """ The original command-line arguments (excluding the first, which is the app itself). """
        self._task_error: Optional[str] = None
        """ Brief error description if cache build task failed, else None. """
    def run(self) -> None:
        job_args = self._parse_command_line()
        if job_args is None:
            return
        emsg, work_dir = WorkingDirectory.load_working_directory_headless(**job_args)
        if len(emsg) > 0:
            sys.stdout.write(f'!!! ERROR: {emsg}\n')
            return

        sys.stdout.write('Initializing cache build task resources...\n')
        sys.stdout.flush()
        task_manager = TaskManager()
        task_manager.progress.connect(self._on_task_progress)
        task_manager.error.connect(self.on_task_failed)

        task_manager.build_internal_cache(work_dir)
        while task_manager.busy:
            time.sleep(0.2)
            task_manager.remove_done_tasks()

        if self._task_error is None:
            if len(work_dir.channels_not_cached()) > 0 or len(work_dir.unit_metrics_not_cached()) > 0:
                sys.stdout.write('!!! ERROR: Cache build task finished, but some cache files are missing.\n')
            else:
                sys.stdout.write(f'OK: Successfully built XSort internal cache for {str(work_dir.path.absolute())}')
        else:
            sys.stdout.write(f'!!! ERROR: Cache build task failed: {self._task_error}')

    @Slot(str, int)
    def _on_task_progress(self, desc: str, pct: int) -> None:
        """
        Handler writes task progress message and percentage complete to the console (STDOUT).
        :param desc: Progress message.
        :param pct: Percent complete. If this lies in [0..100], then "{pct}%" is appended to the progress message.
        """
        msg = f"{desc} - {pct}%\n" if (0 <= pct <= 100) else desc
        sys.stdout.write(msg)
        sys.stdout.flush()

    @Slot(str)
    def on_task_failed(self, emsg: str) -> None:
        """ This slot is the mechanism by which :class:`Analyzer` is notified that a background task has failed. """
        self._task_error = emsg

    def _parse_command_line(self) -> Optional[Dict[str, Any]]:
        """
        Helper method parses command-line arguments and prepares a list of zero or more XSort working directories (with
        configuration parameters if specified) to process.

        If parsing fails, an error description is printed to the console, followed by usage information.
        :return: A dictionary containing the specified command-line arguments converted to the appropriate data types:
            {'folder': Path, 'file1': str, 'file2': str, 'rate': int, 'n_ch': int, 'v_scale': float,
            'interleaved': bool, 'prefiltered': bool}. The only required key is 'folder'. Keys corresponding to
            unspecified command-line arguments are omitted. Returns None on failure.
        """
        n_args = len(self._args)
        if n_args == 1 and self._args[0] == "?":
            CacheBuilder.usage()
            return None
        elif not (n_args in [1, 2, 3, 6, 7, 8]):
            sys.stdout.write('!!! ERROR: Invalid number of arguments specified.\n\n')
            CacheBuilder.usage()
            return None
        else:
            out = dict()
            try:
                dir_path = Path(self._args[0])
                if not dir_path.is_dir():
                    raise Exception('!!! ERROR: Specified directory does not exist.')
                out['folder'] = dir_path
                if n_args in [2, 3, 7, 8]:
                    out['file1'] = self._args[1]
                    if n_args in [3, 8]:
                        out['file2'] = self._args[2]
                if n_args >= 6:
                    try:
                        ofs = 8-n_args
                        out['rate'] = int(self._args[3-ofs])
                        out['n_ch'] = int(self._args[4-ofs])
                        out['v_scale'] = float(self._args[5-ofs])
                        out['interleaved'] = (self._args[6-ofs] == '1')
                        out['prefiltered'] = (self._args[7-ofs] == '1')
                    except ValueError:
                        raise Exception('!!! ERROR: Unable to flat binary file configuration info')
                return out
            except Exception as e:
                sys.stdout.write(str(e)+'\n\n')
                CacheBuilder.usage()
                return None

    @staticmethod
    def usage() -> None:
        """ Print usage information for the command-line version of XSort. """
        for line in _USAGE_LINES:
            sys.stdout.write(line + '\n')
        sys.stdout.flush()
