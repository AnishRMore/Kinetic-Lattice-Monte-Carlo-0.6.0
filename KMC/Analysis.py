"""
This module contains functions for analyzing the output of KMC simulations
"""

__author__ = "Will Ebmeyer"
__all__ = ["KMCResults",
           "calcDiffusionFromMSD",
           "StructureColumns",
           "Structure"]

# Import modules
import pandas as pd
import numpy as np
from enum import Enum
from typing import Tuple, Dict, Union, List
from .Enums import TransitionTypes, TransitionColumns, LogParamNames, BoundaryFlags
from .Exceptions import InvalidTransitionFormat, InvalidStructureFormat
from dataclasses import dataclass
from re import search
from os import path
import gzip


class StructureColumns(Enum):
    """Contains column names used in the structure object's internal dataframe"""
    ELEMENT = 'Element'
    """Refers to element (atomic symbol) column"""
    X = 'X'
    """Refers to x-position column"""
    Y = 'Y'
    """Refers to y-position column"""
    Z = 'Z'
    """Refers to z-position column"""


@dataclass
class Bounds:
    """
    A simple data class containing bounding box information used by structure objects
    """
    xlo: float = None
    xhi: float = None
    ylo: float = None
    yhi: float = None
    zlo: float = None
    zhi: float = None

    @property
    def lx(self) -> Union[float, None]:
        """X-size of the bounding box"""
        if self.xhi is None or self.xlo is None:
            return None
        else:
            return self.xhi - self.xlo

    @property
    def ly(self) -> Union[float, None]:
        """Y-size of the bounding box"""
        if self.yhi is None or self.ylo is None:
            return None
        else:
            return self.yhi - self.ylo

    @property
    def lz(self) -> Union[float, None]:
        """Z-size of the bounding box"""
        if self.zhi is None or self.zlo is None:
            return None
        else:
            return self.zhi - self.zlo

    @property
    def isValid(self) -> Tuple[bool, bool, bool]:
        """Returns whether the bounding box is valid in each direction"""
        return self.lx is not None, self.ly is not None, self.lz is not None


class Structure:
    """
    An object representing a lattice of atom positions
    """

    def __init__(self):
        """Creates and returns a new structure object.  It is recommended to use Structure.fromFile to load a new
        structure."""
        self._dataframe = pd.DataFrame(columns=[StructureColumns.ELEMENT.value,
                                                StructureColumns.X.value,
                                                StructureColumns.Y.value,
                                                StructureColumns.Z.value])
        self.bounds = Bounds()

    @property
    def lx(self):
        """Returns the z-size of the bounding box"""
        return self.bounds.lx

    @property
    def ly(self):
        """Returns the y-size of the bounding box"""
        return self.bounds.ly

    @property
    def lz(self):
        """Returns the z-size of the bounding box"""
        return self.bounds.lz

    def __len__(self):
        """Returns the number of atoms in the structure"""
        return len(self._dataframe)

    def areBoundsValid(self) -> Tuple[bool, bool, bool]:
        """Returns a tuple of booleans indicating whether each dimension of bounding box is valid for use"""
        return self.bounds.isValid

    def rawPositions(self, atom_ids: Union[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts the provided atom ids to raw position data

        Arguments:
            atom_ids: Either an integer or an array of atom ids

        Returns:
            A tuple of the x, y, and z positions
        """
        # Look up positions
        positions = self._dataframe.loc[atom_ids]

        # Convert to returned format
        x = positions[StructureColumns.X.value].values
        y = positions[StructureColumns.Y.value].values
        z = positions[StructureColumns.Z.value].values

        return x, y, z

    @classmethod
    def fromLAMMPS(cls, file: str):
        """
        Loads structure data directly from a LAMMPS .dat file.
        This guarantees that the actual Atom IDs are perfectly mapped to the coordinates,
        regardless of row sorting.
        """
        bounds_dict = {}
        data = []
        
        with open(file, 'r') as f:
            lines = f.readlines()
            
        in_atoms = False
        for line in lines:
            # 1. Extract Bounding Box
            if "xlo xhi" in line:
                bounds_dict['xlo'], bounds_dict['xhi'] = map(float, line.split()[:2])
            elif "ylo yhi" in line:
                bounds_dict['ylo'], bounds_dict['yhi'] = map(float, line.split()[:2])
            elif "zlo zhi" in line:
                bounds_dict['zlo'], bounds_dict['zhi'] = map(float, line.split()[:2])
                
            # 2. Track the Atoms Section
            elif line.startswith("Atoms"):
                in_atoms = True
                continue
            elif line.startswith("Velocities") or line.startswith("Masses") or line.startswith("Bonds"):
                in_atoms = False
                
            # 3. Extract Explicit Atom Data
            elif in_atoms and line.strip():
                parts = line.split()
                # LAMMPS charge atom_style columns: ID, Type, Charge, X, Y, Z
                if len(parts) >= 6:
                    atom_id = int(parts[0])
                    atom_type = str(parts[1]) # Save LAMMPS type as string
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    data.append([atom_id, atom_type, x, y, z])
                    
        bounds = Bounds(**bounds_dict)
        
        # 4. Create the DataFrame
        df = pd.DataFrame(data, columns=['Atom_ID', StructureColumns.ELEMENT.value, 
                                         StructureColumns.X.value, StructureColumns.Y.value, 
                                         StructureColumns.Z.value])
        
        # THE CRITICAL FIX: Set the dataframe index explicitly to the Atom_ID
        df.set_index('Atom_ID', inplace=True)
        
        # Build structure object
        result = cls()
        result._dataframe = df
        result.bounds = bounds
        
        return result
    
    @classmethod
    def fromXYZ(cls, file: str):
        """
        Loads structure data from an XYZ file

        Arguments:
            file: Name of file to load data from

        Returns:
            A new structure object
        """
        # Read file header
        with open(file, mode='r') as rf:
            # Get atom count
            try:
                first_line = rf.readline()
                atom_count = int(first_line)
            except (ValueError, TypeError):
                raise InvalidStructureFormat("Could not interpret 'atom count' from the first line as an integer (was "
                                             f"{first_line})")

            # Extract bounds data from comment line, if specified
            comment_line = rf.readline()
            bounds_dict = {}
            for key in ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi']:
                try:
                    value = float(search(rf'{key}: ([\d.-]+)', comment_line).groups()[0])
                except (AttributeError, TypeError, ValueError):
                    bounds_dict[key] = None
                else:
                    bounds_dict[key] = value
            # Convert to 'bounds' object
            bounds = Bounds(**bounds_dict)

            # Attempt to load data from file
            data = pd.read_csv(file, sep=' ', index_col=False, skiprows=2, header=None)

            # Verify dataframe shape
            if data.shape != (atom_count, 4):
                raise InvalidStructureFormat(
                    f"Structure had invalid shape (was {data.shape}, expected {(atom_count, 4)})")

            # Name the columns
            data.columns = [StructureColumns.ELEMENT.value,
                            StructureColumns.X.value,
                            StructureColumns.Y.value,
                            StructureColumns.Z.value]

            # Set the index to start at 1 to match atom ID
            data.index += 1

            # Enforce column types
            data[StructureColumns.ELEMENT.value] = data[StructureColumns.ELEMENT.value].astype(str)
            data[StructureColumns.X.value] = data[StructureColumns.X.value].astype(float)
            data[StructureColumns.Y.value] = data[StructureColumns.Y.value].astype(float)
            data[StructureColumns.Z.value] = data[StructureColumns.Z.value].astype(float)

            # Build structure object
            result = cls()
            result._dataframe = data
            result.bounds = bounds

            return result

    @classmethod
    def fromFile(cls, file: str, fmt: str = None):
        """
        Loads a structure from the specified file

        Arguments:
            file: File location to load data from
            fmt: File format to load as.  If not provided, this will guess based on the file extension.

        Returns:
            A structure object
        """
        # Try to guess format from extension if not provided
        if fmt is None:
            fmt = path.splitext(file)[1].strip('.')

        # Force to lower case
        fmt = fmt.lower()

        # Load file based on format
        if fmt == 'xyz':
            return cls.fromXYZ(file)
        elif fmt == 'dat':
            return cls.fromLAMMPS(file)
        else:
            raise ValueError(f"Unsupported file format '{fmt}'")


class MigrationHistory:
    """
    Represents the migration history of a singular vacancy
    """
    def __init__(self, transition_types: np.ndarray, transition_args: np.ndarray, vacancy_ids: np.ndarray):
        """
        Arguments:
            transition_types: Array of integers corresponding to the type of transition
            transition_args: Array of integers corresponding to additional transition arguments
            vacancy_ids: Array of integers corresponding to atom-id sites occupied by the vacancy post-transition
        """
        self.transition_types = transition_types
        self.transition_args = transition_args
        self.vacancies = vacancy_ids

        # Make sure all arrays are the same length
        if len(transition_types) != len(transition_args) or len(transition_args) != len(vacancy_ids):
            raise ValueError("All arrays must be the same length")

    def __len__(self) -> int:
        return len(self.vacancies)

    def totalRejected(self) -> int:
        """Returns the number of REJECTED transitions"""
        return int(np.sum(self.transition_types == TransitionTypes.NO_CHANGE.value))

    def totalAccepted(self) -> int:
        """Returns the total number of ACCEPTED transitions"""
        return len(self) - self.totalRejected()

    def checkCrossings(self) -> bool:
        """Returns whether any boundary crossings occurred in this particular history"""
        # Only look at site-to-site transitions
        slicer = self.transition_types == TransitionTypes.VACANCY_TO_LATTICE_SITE.value
        return np.any(self.transition_args[slicer] != BoundaryFlags.NONE)


class KMCResults:
    """
    Class for working with and analyzing the results of a KMC simulation
    """

    TRANSITION_TYPE_FMT = "Transition Type %s"

    def __init__(self):
        """
        Returns a new KMCResults object.  It is recommended to use KMCResults.fromFile to load results from a KMC
        simulation.
        """
        # self._df: pd.DataFrame = pd.DataFrame()
        self._histories: List[MigrationHistory] = []
        self._kB: float = 0.0
        self._temperature: float = 0.0
        self._initial_vacancies: Tuple[int] = tuple()
        self._barriers_source_file: str = ''
        self._log_format: str = ''

    def __len__(self) -> int:
        result = 0
        for history in self.iterHistories():
            result = max(result, len(history))
        return result

    def set_kB(self, new_value: float):
        """Sets the value of the Boltzmann constant in the transition results"""
        self._kB = float(new_value)

    def set_temperature(self, new_value: float):
        """Sets the value of the running temperature in the transition results"""
        self._temperature = float(new_value)

    def checkCrossings(self) -> bool:
        """Returns whether any boundary-crossings occurred"""
        # Check each history for boundary crossings
        for history in self.iterHistories():
            if history.checkCrossings():
                return True

        # No boundary crossings found
        return False

    def tallyRejects(self) -> int:
        """Returns the total number of REJECTED transitions"""
        result = 0
        for history in self.iterHistories():
            result += history.totalRejected()
        return result

    def tallyAttempts(self) -> int:
        """Returns the total number of ATTEMPTED transitions"""
        result = 0
        for history in self.iterHistories():
            result += len(history)
        return result

    def tallyAccepts(self) -> int:
        """Returns the total number of ACCEPTED transitions"""
        result = 0
        for history in self.iterHistories():
            result += history.totalAccepted()
        return result

    @property
    def kB(self) -> float:
        """Boltzmann constant used in the simulation"""
        return self._kB

    @property
    def temperature(self) -> float:
        """Temperature that the simulation was run at"""
        return self._temperature

    @property
    def T(self) -> float:
        """Alias for temperature"""
        return self._temperature

    @property
    def initialVacancies(self) -> Tuple[int]:
        """Atom ids occupied by the vacancies at the start of the simulation"""
        return self._initial_vacancies

    @property
    def vacancyCount(self) -> int:
        """Returns the number of vacancies found in the transitions file"""
        return len(self._histories)

    def timeArray(self, dt: float) -> np.ndarray:
        """
        Returns an array containing the time history across the simulation
        """
        return np.arange(0, len(self)+1, 1)*dt

    def iterHistories(self):
        """
        Iterator that returns the history for each individual vacancy

        Yields:
            A migration history object for each vacancy
        """
        for history in self._histories:
            yield history

    @classmethod
    def fromFile(cls, file_name: str):
        """
        Loads simulation results data from a file.  If the file extension is '.gz', the file is treated as a compressed
        gzip file and will be decompressed accordingly.

        Arguments:
            file_name: Name of file to load simulations results from

        Returns:
            A new KMCResults object
        """
        # Extract parameters from header
        histories = []
        initial_vacancies: Union[Tuple[int], None] = None
        constants: Dict[str, float] = {}
        barriers_source_file: str = ''
        log_format: str = ''

        # Check if file is compressed and use the appropriate opening function
        if path.splitext(file_name)[-1] == ".gz":
            open_func = gzip.open
        else:
            open_func = open

        with open_func(file_name, mode='rb') as rf:
            # Scan file for relevant fields
            while True:
                # Read next line
                line = rf.readline().decode()

                # Still in header
                if line[0] == '#':
                    # Found initial vacancies field
                    if LogParamNames.INITIAL_VACANCY_IDS in line or 'Initial Vacancies' in line:
                        # Found the "initial vacancies" field
                        line = rf.readline().decode()
                        line = line.strip('#').strip()
                        try:
                            initial_vacancies: Tuple[int] = tuple(int(i) for i in line.split(','))
                        except (ValueError, TypeError):
                            raise InvalidTransitionFormat("'Initial Vacancy Ids' field could not be interpreted as a "
                                                          f"list of integers (received {line})")

                    elif 'Constants' in line:
                        # Found the "constants" field
                        line = rf.readline().decode()
                        line = line.strip('#').strip()
                        for arg in line.split(','):
                            try:
                                key, value = arg.strip().split('=')
                            except ValueError:
                                raise InvalidTransitionFormat("'Constants' field was not formatted as comma-separated "
                                                              "key=value pairs (received %s)" % line)
                            else:
                                try:
                                    constants[str(key)] = float(value)
                                except (ValueError, TypeError):
                                    raise InvalidTransitionFormat("'Constants' field had incorrect format: could not "
                                                                  f"convert '{key}' ({value}) field to a float")

                    elif 'Barriers Source File' in line:
                        # Found the file location of the original barriers file
                        line = rf.readline().decode()
                        line = line.strip('#').strip()
                        barriers_source_file = line

                    elif 'Log Format' in line:
                        # Found the format of the following history data
                        line = rf.readline().decode()
                        line = line.strip('#').strip()
                        log_format = line

                # We've run out of header, exit the loop
                else:
                    break

            # Make sure we at least have the required fields
            if initial_vacancies is None:
                raise InvalidTransitionFormat("Could not locate initial vacancies field in header")
            else:
                vacancy_count = len(initial_vacancies)

        # Load transition history from file
        transitions = pd.read_csv(file_name, sep=',', index_col=False, comment='#')

        # Remove whitespaces from columns
        transitions.columns = [str(i).strip() for i in transitions.columns]

        # Check for correct shape
        if len(transitions.shape) != 2 or transitions.shape[1] < 2:
            raise InvalidTransitionFormat(f"Transitions table had unexpected shape (was {transitions.shape}, "
                                          f"expected a 2D array)")

        # Make sure we have three columns per vacancy
        if len(transitions.columns) != 3*vacancy_count:
            raise InvalidTransitionFormat(
                "Each vacancy should have three associated columns: transition type, transition "
                f"args, and final vacancy ID.  There are {vacancy_count} count, but {len(transitions.columns)} "
                f"columns!")

        # Attempt to extract transition history data for each vacancy
        for i in range(1, vacancy_count+1):
            # Find column locations
            try:
                type_col = transitions[TransitionColumns.TRANSITION_TYPE % i].values
                args_col = transitions[TransitionColumns.TRANSITION_ARGS % i].values
                vac_col = transitions[TransitionColumns.VACANCY % i].values

            except (ValueError, KeyError):
                raise InvalidTransitionFormat("Transition file column names had bad format, expected: \n"
                                              f"{TransitionColumns.TRANSITION_TYPE % 1}, "
                                              f"{TransitionColumns.TRANSITION_ARGS % 1}, "
                                              f"{TransitionColumns.VACANCY % 1}, ..., "
                                              f"{TransitionColumns.TRANSITION_TYPE % 'N'}, "
                                              f"{TransitionColumns.TRANSITION_ARGS % 'N'}, "
                                              f"{TransitionColumns.VACANCY % 'N'}\n"
                                              f"Received: \n{transitions.columns}")
                # Columns were unlabeled/labeled incorrectly, guess from their order
                # warnings.warn('Transition file column names had bad format, so their order must be guessed: {err}',
                #               PoorTransitionFormat)

            else:
                histories.append(MigrationHistory(type_col, args_col, vac_col))

        # Build KMC results object
        result = KMCResults()
        result._histories = histories
        result._initial_vacancies = initial_vacancies
        result._barriers_source_file = barriers_source_file
        result._log_format = log_format
        if 'kB' in constants:
            result._kB = constants['kB']
        if 'T' in constants:
            result._temperature = constants['T']

        # Return result
        return result

    def getPositionHistories(self, structure: Union[Structure, str], include_crossings=True) -> List[np.ndarray]:
        """
        Converts the atom ids from the simulation history to actual atom positions in xyz space.

        Arguments:
            structure: Structure object to reference position data from.  Optionally may be a string, in which case
                a structure object will be loaded from the file path specified by the string.
            include_crossings: If "true," atoms that cross periodic boundaries will have an offset applied to their
                position, so they don't spontaneously jump from one end of the structure to the other.

        Returns:
            List of 3xN arrays corresponding to the position history of each vacancy.
        """
        # Verify that 'structure' is a structure object
        if isinstance(structure, str):
            structure = Structure.fromFile(structure)
        if not isinstance(structure, Structure):
            raise TypeError(f"structure must be a 'Structure' object, (was {type(structure)})")

        # Box dimensions are REQUIRED if there are any boundary crossings
        if self.checkCrossings() and include_crossings:
            for axis, valid in zip(['x', 'y', 'z'], structure.areBoundsValid()):
                if not valid:
                    raise ValueError(f"Box dimensions are required for cross-boundary transitions "
                                     f"({axis}-size was 'None')")

        # Translate visited atom sites to physical positions.
        histories = []
        for initial_vacancy, transitions in zip(self.initialVacancies, self.iterHistories()):
            # First, add in the vacancy's initial location
            vacancies = np.concatenate([[initial_vacancy], transitions.vacancies])

            # Now extract the positions of the vacancy before and after each jump
            x, y, z = structure.rawPositions(vacancies)

            # Some jumps cross the boundary, so we need to account for that
            if transitions.checkCrossings() and include_crossings:
                px_slicer = (transitions.transition_args & BoundaryFlags.PX).astype(bool)
                nx_slicer = (transitions.transition_args & BoundaryFlags.NX).astype(bool)
                py_slicer = (transitions.transition_args & BoundaryFlags.PY).astype(bool)
                ny_slicer = (transitions.transition_args & BoundaryFlags.NY).astype(bool)
                pz_slicer = (transitions.transition_args & BoundaryFlags.PZ).astype(bool)
                nz_slicer = (transitions.transition_args & BoundaryFlags.NZ).astype(bool)

                x[1:] = x[1:] - structure.lx*np.cumsum(px_slicer)
                x[1:] = x[1:] + structure.lx*np.cumsum(nx_slicer)
                y[1:] = y[1:] - structure.ly*np.cumsum(py_slicer)
                y[1:] = y[1:] + structure.ly*np.cumsum(ny_slicer)
                z[1:] = z[1:] - structure.lz*np.cumsum(pz_slicer)
                z[1:] = z[1:] + structure.lz*np.cumsum(nz_slicer)

            histories.append(np.array([x, y, z]).T)

        return histories

    def calcMSD(self, structure: Union[Structure, str], skip=1) -> np.ndarray:
        """
        Calculates the total mean-squared displacement across the entire simulation

        Arguments:
            structure: Structure object to reference position data from.  Optionally may be a string, in which case
                a structure object will be loaded from the file path specified by the string.
            skip: Applies a skip interval, where only evey N timesteps are kept.

        Returns:
            The mean-squared displacement across the simulation history
        """
        # Compute for each vacancy
        result = np.zeros(len(self) // skip, dtype=float)

        # Obtain vacancy position histories
        positions_histories = self.getPositionHistories(structure, include_crossings=True)

        for positions in positions_histories:
            # Extract X, Y, Z position from array
            x, y, z = positions.T

            # Apply skip
            x = x[::skip]
            y = y[::skip]
            z = z[::skip]

            # Compute jump distance
            dx = x[1:] - x[:-1]
            dy = y[1:] - y[:-1]
            dz = z[1:] - z[:-1]

            distance2 = dx ** 2 + dy ** 2 + dz ** 2
            cumulative_distance2 = np.cumsum(distance2)

            result = result + cumulative_distance2

        # Take the average and return the result
        return result / self.vacancyCount

    def calcDiffusion(self, structure: Union[Structure, str], dt: float, skip=1) -> float:
        """
        Computes the diffusion coefficient

        Arguments:
            structure: Structure object to reference position data from.  Optionally may be a string, in which case
                a structure object will be loaded from the file path specified by the string.
            dt: Assumed timestep size for the KMC simulation.
            skip: Applies a skip interval, where only evey N timesteps are kept.

        Returns:
            Diffusion coefficient at the simulation's end
        """
        # Compute the MSD
        MSD = self.calcMSD(structure, skip=skip)

        # Compute the diffusion coefficients and return the results
        return calcDiffusionFromMSD(MSD, dt*skip)

    def calcWindowedDiffusion(self, structure: Structure, window_size: int, dt: float) -> List[np.ndarray]:
        """
        Computes diffusion coefficients for each vacancy individually using a windowed method

        Arguments:
            structure: Structure object to reference position data from.
            window_size: Size of the moving window to calculate diffusion coefficients over
            dt: Timestep used in calculations

        Returns:
            List of diffusion coefficient histories, each entry corresponds to a different vacancy.
        """
        # Obtain vacancy position histories
        positions_histories = self.getPositionHistories(structure, include_crossings=True)
        time = self.timeArray(dt)
        windowed_final_time = time[window_size:] - time[:-window_size]

        # Compute square displacement
        results = []
        windowed_positions = []
        for positions in positions_histories:
            # Extract X, Y, Z position from array
            x, y, z = positions.T

            # Compute jump distance
            dx = x[1:] - x[:-1]
            dy = y[1:] - y[:-1]
            dz = z[1:] - z[:-1]

            delta2 = dx ** 2 + dy ** 2 + dz ** 2

            # Compute windowed MSD
            windowed_MSD = np.cumsum(delta2)
            windowed_MSD[window_size:] = windowed_MSD[window_size:] - windowed_MSD[:-window_size]
            windowed_MSD = windowed_MSD[window_size-1:]

            # Compute diffusion coefficient
            windowed_D = windowed_MSD / windowed_final_time / 6
            results.append(windowed_D)

            # Compute windowed positions
            windowed_positions.append(np.array([moving_average(x, window_size),
                                                moving_average(y, window_size),
                                                moving_average(z, window_size)]).T)

        return results

    def getWindowedPositionHistories(self, structure: Structure, window_size: int,
                                     include_crossings=True) -> List[np.ndarray]:
        """
        Takes a moving average of the vacancy positions and returns the result.

        Arguments:
            structure: Structure object to reference position data from.
            window_size: Size of the moving window to calculate diffusion coefficients over.
            include_crossings: If 'False' the resulting coordinates will be wrapped to fall within the structure's
                bounding box.
        """
        # Obtain vacancy position histories
        positions_histories = self.getPositionHistories(structure, include_crossings=True)

        # Compute square displacement
        windowed_positions = []
        for positions in positions_histories:
            # Extract X, Y, Z position from array
            x, y, z = positions.T

            # Undo crossings, if necessary
            if not include_crossings:
                x = ((x - structure.bounds.xlo) % structure.lx) + structure.bounds.xlo
                y = ((y - structure.bounds.ylo) % structure.ly) + structure.bounds.ylo
                z = ((z - structure.bounds.zlo) % structure.lz) + structure.bounds.zlo

            # Compute windowed positions and wrap back into array
            windowed_positions.append(np.array([moving_average(x, window_size),
                                                moving_average(y, window_size),
                                                moving_average(z, window_size)]).T)

        return windowed_positions

    def calcMeanActivationEnergy(self):
        """
        Computes the average activation energy based on the proportion of accepted transitions in a KMC simulation

        Returns:
            Average activation energy
        """
        # Get the total proportion of transitions that were accepted
        rejected, total = self.tallyRejects(), self.tallyAttempts()
        accepted_rate = (total - rejected) / total

        # Compute the average activation energy using the Arrhenius equation
        return -self.kB * self.T * np.log(accepted_rate)

    def calcIonConcentration(self, lx: float, ly: float, lz: float) -> float:
        """
        Computes the concentration of ionic carriers

        Arguments:
            lx: x-size of structure bounding box
            ly: y-size of structure bounding box
            lz: z-size of structure bounding box

        Returns:
            Concentration per unit volume
        """
        return self.vacancyCount / (lx*ly*lz)

    def calcIonicConductivity(self, structure: Union[str, Structure], dt: float, charge: float, skip=1) -> float:
        """
        Computes the ionic conductivity over the course of the simulation

        Arguments:
            structure: Structure object to reference position data from.  Optionally may be a string, in which case
                a structure object will be loaded from the file path specified by the string.
            dt: Assumed timestep of the simulation
            charge: Charge of the ionic carriers
            skip: Applies a skip interval, where only evey N timesteps are kept.

        Returns:
            Ionic conductivity over the course of the simulation
        """
        # Verify that 'structure' is a structure object
        if isinstance(structure, str):
            structure = Structure.fromFile(structure)
        if not isinstance(structure, Structure):
            raise TypeError(f"structure must be a 'Structure' object, (was {type(structure)})")

        # Compute the diffusion coefficient
        diff = self.calcDiffusion(structure, dt, skip=skip)

        # Compute carrier concentration
        concentration = self.calcIonConcentration(structure.lx, structure.ly, structure.lz)

        # Compute the ionic conductivity and return the result
        # [Length²/Time][Length⁻³][Charge]²/[Energy]
        sigma = diff * concentration * charge**2 / (self.kB * self.T)
        return sigma

    def calcWindowedIonicConductivity(self, structure: Union[str, Structure], window_size: int, dt: float,
                                      charge: float) -> List[np.ndarray]:
        """
        Computes the ionic conductivity over a moving window for each vacancy.

        Arguments:
            structure: Structure object to reference position data from.  Optionally may be a string, in which case
                a structure object will be loaded from the file path specified by the string
            window_size: Size of the moving window to calculate diffusion coefficients over
            dt: Assumed timestep of the simulation
            charge: Charge of the ionic carriers

        Returns:
            A list of ionic conductivity arrays corresponding to each vacancy.
        """
        # Verify that 'structure' is a structure object
        if isinstance(structure, str):
            structure = Structure.fromFile(structure)
        if not isinstance(structure, Structure):
            raise TypeError(f"structure must be a 'Structure' object, (was {type(structure)})")

        # Compute the diffusion coefficient
        diffusions = self.calcWindowedDiffusion(structure, window_size, dt)

        # Compute carrier concentration
        concentration = self.calcIonConcentration(structure.lx, structure.ly, structure.lz)

        # Compute the ionic conductivity and return the result
        # [Length²/Time][Length⁻³][Charge]²/[Energy]
        sigmas = [df * concentration * charge**2 / (self.kB * self.T) for df in diffusions]
        return sigmas

    def stats(self) -> dict:
        """
        Computes a bunch of stats from the simulation results and returns everything as a dictionary.  This includes
        the reject rate, accept rate, and the average activation energy.
        """
        # Build dictionary and return the results
        result = {"Reject Rate": self.tallyRejects() / self.tallyAttempts(),
                  "Accept Rate": self.tallyAccepts() / self.tallyAttempts(),
                  "Mean ΔE    ": self.calcMeanActivationEnergy()}
        return result


def moving_average(x, n) -> np.ndarray:
    """Computes a moving average of window size 'n' over the provided array 'x'"""
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def weighted_average(values, errors) -> Tuple[float, float]:
    """
    Computes weighted average using errors

    Arguments:
        values: Array of values to average
        errors: Array of errors

    Returns:
        Tuple containing the weighted average and error
    """
    # Remove any nans
    nans = np.logical_or(np.isnan(values), np.isnan(errors))
    values = values[~nans]
    errors = errors[~nans]

    # Skip if these are zero-length arrays
    if len(values) == 0:
        return float(np.nan), float(np.nan)

    # Fix zero-uncertainties
    errors[errors == 0.0] = 1e-5

    # Compute weighted average
    weights = 1 / errors ** 2
    avg = float(np.average(values, weights=weights))
    err = float(1 / np.sqrt(np.sum(weights)))

    # Return result
    return avg, err


def calcDiffusionFromMSD(MSD: np.ndarray, dt: float) -> float:
    """
    Computes the diffusion coefficient from the results of a simulation.

    Arguments:
        MSD: Mean-squared displacement array calculated elsewhere
        dt: Assumed timestep of the simulation

    Returns:
        Diffusion coefficient at simulation's end
    """
    # Convert these to diffusion coefficients
    t = np.linspace(0, dt * len(MSD), len(MSD))
    # Length² / Time
    with np.errstate(divide='ignore', invalid='ignore'):
        Df = MSD / (6*t)
    return Df[-1]
