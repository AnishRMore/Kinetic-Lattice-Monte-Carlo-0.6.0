"""
This submodule contains all enums used by the KMC package
"""

# Set module-level dunder names
__author__ = 'Will Ebmeyer'
__all__ = ['TransitionTypes',
           'TransitionColumns',
           'LogParamNames',
           'BoundaryFlags']

# Import modules
from enum import Enum, IntEnum


# Declare log file parameter names
class LogParamNames(str, Enum):
    """Enum of parameter names used in the header of a KMC log file."""
    CONSTANTS = 'Constants'
    """Refers to the 'constants' in a KMC log file. Constants are stored as comma-separated key=value pairs"""
    INITIAL_VACANCY_IDS = 'Initial Vacancy Ids'
    """Refers to the initial atom ids of vacancies before the simulator started running"""
    BARRIERS_SOURCE_FILE = 'Barriers Source File'
    """Refers to the file name that barriers data was loaded from, if applicable."""
    LOG_FORMAT = 'Log Format'
    """Refers to the log format parameter in a KMC log file.  Usually 'History'"""


# Declare transition types
class TransitionTypes(Enum):
    """Enum of integers that represent the type of transition that occurred.  Used in KMC log files."""
    PLACEHOLDER = 0
    """A placeholder value indicating no type has been set.  Should not be used."""
    NO_CHANGE = 1
    """Indicates that no actual change occurred (i.e. there was no transition)."""
    VACANCY_TO_LATTICE_SITE = 2
    """Indicates that a vacancy transitioned to a different lattice site."""

    def __int__(self) -> int:
        return int(self.value)


# Declare column name formats for transitions log file
class TransitionColumns(str, Enum):
    """Enum of column name formats used in KMC log files"""
    TRANSITION_TYPE = 'Transition Type %s'
    """Placeholder string for a vacancy's transition type column"""
    TRANSITION_ARGS = 'Transition Args %s'
    """Placeholder string for a vacancy's transition argument column"""
    VACANCY = 'Vacancy %s'
    """Placeholder string for a vacancy's occupied lattice site column"""


# Declare enum for boundary flags
class BoundaryFlags(IntEnum):
    """
    Flags indicating what boundaries, if any, a transition crossed.  If multiple boundaries were crossed, the resulting
    number is the bitwise OR of the individual boundary flags.  Used in KMC log files.
    """
    NONE = 0
    """All zero flags indicate no boundary was crossed"""
    PX = 1
    """Flag indicates transition crossed +X boundary"""
    NX = 2
    """Flag indicates transition crossed -X boundary"""
    PY = 4
    """Flag indicates transition crossed +Y boundary"""
    NY = 8
    """Flag indicates transition crossed -Y boundary"""
    PZ = 16
    """Flag indicates transition crossed +Z boundary"""
    NZ = 32
    """Flag indicates transition crossed -Z boundary"""
