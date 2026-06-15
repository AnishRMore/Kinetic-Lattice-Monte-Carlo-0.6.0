"""
This module contains a bunch of common exceptions and warnings for the program
"""


class KMCBaseException(Exception):
    """Generic exception for all errors caught and thrown by the KMC"""
    pass


class SimValidationException(KMCBaseException):
    """Exception thrown if a KMC simulation fails its initial checks"""
    pass


class KMCAnalysisExceptions(KMCBaseException):
    """Base exception thrown when the analysis module encounters an error"""
    pass


class InvalidStructureFormat(KMCAnalysisExceptions):
    """Exception thrown when the provided structure file was found to have an invalid format"""


class InvalidTransitionFormat(KMCAnalysisExceptions):
    """Exception thrown when the provided transitions file was found to have an invalid format"""


class KMCWarning(Warning):
    """Base class for warnings thrown by the KMC"""


class PoorTransitionFormat(KMCWarning):
    """Warning shown when user poorly formatted the columns in the transitions log file"""


class InvalidArguments(KMCWarning):
    """Warning thrown when the command-line arguments passed to the program are invalid"""


class KMCAnalysisWarning(KMCWarning):
    """Base class for warning thrown by the KMC analysis module"""


class MissingBoundsWarning(KMCAnalysisWarning):
    """Warning thrown when the user attempts to access structure bounds data that hasn't been defined"""
