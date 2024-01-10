import csv
from pathlib import Path
from typing import List, Tuple, Optional, Union

from xsort.data.neuron import Neuron

EDIT_HISTORY_FILE: str = '.xs.edits.txt'
""" 
Small text file containing chronological history of any changes made to the list of neural units defined
in the original spike sorter results file.
"""


class UserEdit:
    """
    A simple container encapsulating an individual user-initiated change to the list of neural units defined in the
    original spike sorter results file in the XSort working directory. The following edit operations are supported:
     - Label a neural unit (typically to specify the putative neuron type).
     - Delete a neural unit. Often an "uncurated" spike sorter results file contains many "garbage" units, and the
       user's first task in XSort is to simply delete all such units.
     - Create a new unit by merging two selected units.
     - Split an existing unit. This is done by specifying a subset of the existing unit's spikes to assign to one unit,
       while the remaining are assigned to another. In practice, this is done on the XSort UI by "lassoing" a population
       of spikes in the principal component analysis (PCA) view.

    The edit history for an XSort working directory is simply a list of :class:`UserEdit` objects, in the order in
    which the edits were applied.
    """
    LABEL: str = 'LABEL'
    """ Edit a unit label: 'LABEL,unit_uid,old_label,new_label'."""
    DELETE: str = 'DELETE'
    """ Delete a unit: 'DELETE,unit_uid'. """
    MERGE: str = 'MERGE'
    """ Merge two selected units into one: 'MERGE,uid1,uid2,uid_merged'. """
    SPLIT: str = 'SPLIT'
    """ Split a unit into two units: 'SPLIT,uid_before,uid_split1,uid_split2'. """
    _EDIT_OPS = {LABEL: 'Change unit label', DELETE: 'Delete unit', MERGE: 'Merge units', SPLIT: 'Split unit'}
    """ All recognized edit operations. """

    def __init__(self, op: str, params: List[str]):
        """
        Construct an edit record. The parameter list depend on the edit operation, as follows:
         - op = 'LABEL': params=[<UID of affected neural unit>, <previous label>, <new label>].
         - op = 'DELETE': params=[<UID of deleted neural unit>].
         - op = 'MERGE': params=[uid1, uid2, uid_merged].
         - op = 'SPLIT': params=[uid_split, uid1, uid2].

        :param op: The edit operation.
        :param params: The operation parameter list, as described above. All parameter values are strings.
        :raises IndexError: If the parameter list is not long enough (extra parameter values are ignored).
        :raises TypeError: If any parameter value is not a string.
        :raises ValueError: If any parameter value contains a comma. Edit operations are stored in the edit history
            file as a comma-separated list of strings.
        """
        if not (op in UserEdit._EDIT_OPS):
            raise ValueError(f"Unrecognized edit op: {op}")
        self._op = op
        """ The edit operation type. """
        self._params: List[str] = list()
        """ 
        The operation parameters.
        """
        n_params = 1 if (op == UserEdit.DELETE) else 3
        for i in range(n_params):
            if not isinstance(params[i], str):
                raise TypeError('Expected string-valued parameter')
            elif params[i].find(',') > -1:
                raise ValueError('Parameter values may not contain a comma')
            self._params.append(params[i])

    def __str__(self):
        return ','.join((self._op, *self._params))

    def __eq__(self, other: 'UserEdit'):
        return isinstance(other, UserEdit) and (self._op == other._op) and (len(self._params) == len(other._params)) \
            and all([self._params[i] == other._params[i] for i in range(len(self._params))])

    @property
    def operation(self) -> str:
        """ The edit operation type: LABEL, DELETE, MERGE or SPLIT. """
        return self._op

    @property
    def short_description(self) -> str:
        """ A short generic description of the edit operation type. """
        return self._EDIT_OPS[self._op]

    @property
    def longer_description(self) -> str:
        """ A longer description of the edit record that includes the UIDs of the affected/created units. """
        if self._op == UserEdit.LABEL:
            desc = f"Changed unit {self.affected_uids} label: '{self.previous_unit_label}' --> '{self.unit_label}"
        elif self._op == UserEdit.DELETE:
            desc = f"Deleted unit '{self.affected_uids}'"
        elif self._op == UserEdit.MERGE:
            desc = f"Merged units {self.affected_uids} into unit '{self.result_uids}'"
        else:   # SPLIT
            desc = f"Split unit '{self.affected_uids}' into units {self.result_uids}"
        return desc

    @property
    def affected_uids(self) -> Union[str, Tuple[str, str]]:
        """
        The UID of the unit that is re-labeled, deleted, or split. For a "merge", a 2-tuple containing the UIDs of
        the two merged units.
        """
        return (self._params[0], self._params[1]) if self._op == UserEdit.MERGE else self._params[0]

    @property
    def result_uids(self) -> Union[None, str, Tuple[str, str]]:
        """ The UID of the new merged unit, or the UIDs of the two units resulting from a split. Else None. """
        if self._op == UserEdit.MERGE:
            return self._params[2]
        elif self._op == UserEdit.SPLIT:
            return self._params[1], self._params[2]
        else:
            return None

    @property
    def previous_unit_label(self) -> Optional[str]:
        """ For the "label" op only, the old label assigned to the unit; else None. """
        return self._params[1] if self._op == UserEdit.LABEL else None

    @property
    def unit_label(self) -> Optional[str]:
        """ For the "label" op only, the new label assigned to the unit; else None. """
        return self._params[2] if self._op == UserEdit.LABEL else None

    def apply_to(self, units: List[Neuron]) -> bool:
        if self._op == UserEdit.LABEL:
            for u in units:
                if (u.uid == self.affected_uids) and (u.label == self.previous_unit_label):
                    try:
                        u.label = self.unit_label
                        return True
                    except ValueError:
                        pass
                    break
        elif self._op == UserEdit.DELETE:
            found_idx = -1
            for i in range(len(units)):
                if units[i].uid == self.affected_uids:
                    found_idx = i
                    break
            if found_idx > -1:
                units.pop(found_idx)
                return True
        elif self._op == UserEdit.MERGE:
            idx1, idx2 = -1, -1
            for i in range(len(units)):
                if units[i].uid in self.affected_uids:
                    if idx1 > -1:
                        idx2 = i
                        break
                    else:
                        idx1 = i
            if (idx1 > -1) and (idx2 > -1):
                unit1 = units.pop(idx1)
                unit2 = units.pop(idx2)
                new_uid = self.result_uids
                new_idx = int(new_uid[0:-1])  # remove the 'x' suffix to get the unit index
                merged_unit = Neuron.merge(unit1, unit2, idx=new_idx)
                units.append(merged_unit)
                return True
        else:   # SPLIT
            return False  # TODO: IMPLEMENT

        return False   # if we get here, we didn't find the unit(s) affected in the list provided

    @staticmethod
    def save_edit_history(working_dir: Path, history: List['UserEdit']) -> Tuple[bool, str]:
        """
        Save the list of user edits to the edit history file within the specifed XSort working directory.

        :param working_dir: The XSort working directory.
        :param history: The edit history to be saved. **If this list is empty, the edit history file (if it exists)
            will be removed from the working directory.**
        :return: A 2-tuple: (True, '') if successful, else (False, <error description>).
        """
        if not working_dir.is_dir():
            return False, "Invalid working directory"

        # if the edit history is empty, obliterate the edit history file
        p = Path(working_dir, EDIT_HISTORY_FILE)
        if len(history) == 0:
            p.unlink(missing_ok=True)
            return True, ""

        try:
            with open(p, 'w', newline='') as f:
                writer = csv.writer(f)
                for ue in history:
                    writer.writerow((ue._op, *ue._params))
        except Exception as e:
            msg = f"Failed to save edit history: {str(e)}"
            return False, msg

        return True, ""

    @staticmethod
    def load_edit_history(working_dir: Path) -> Tuple[str, List['UserEdit']]:
        """
        Load the contents of the edit history file from the specified XSort working directory.

        :param working_dir: The XSort working directory.
        :return: A 2-typle: ('', H) on success or (<error description>, []) on failure, where H is the edit history
            read from the file. If the file is missing from the working directory, it is assumed the edit history is
            empty.
        """
        if not working_dir.is_dir():
            return "Invalid working directory", []

        # if edit history file does not exist, the edit history is empty
        p = Path(working_dir, EDIT_HISTORY_FILE)
        if not p.is_file():
            return "", []

        try:
            history: List[UserEdit] = list()
            with open(p, 'r', newline='') as f:
                reader = csv.reader(f)
                for line in reader:
                    ue = UserEdit(line[0], [line[i] for i in range(1, len(line))])
                    history.append(ue)
            return "", history
        except Exception as e:
            msg = f"Failed to load edit history: {str(e)}"
            return msg, []
