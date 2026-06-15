"""
This submodule contains the main KMC simulator code
"""

# Set module-level dunder names
__author__ = 'Will Ebmeyer'
__version__ = 'v0.5.2'
__all__ = ['KMCSimulator']

# Import modules
from random import random

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple, List, Iterable
from .Enums import TransitionTypes, TransitionColumns, LogParamNames
from .Exceptions import SimValidationException


# Helper classes
@dataclass
class Constants:
    """Dataclass used to contain important constants in the simulation"""
    kB: float = 1.0
    T: float = 1.0


# Core Simulator
class KMCSimulator:
    """Main simulator class used for handling and running simulations"""
    MOVING_ATOM_ID = 'Moving Atom ID'
    VACANT_ATOM_ID = 'Vacant Atom ID'
    ENERGY_BARRIER = 'Energy Barrier'
    BOUNDARY_FLAGS = 'Boundary Flags'

    class ValidationStatus(Enum):
        INVALID_STARTING_VACANCIES = "Invalid vacancies"
        """Warns that the starting vacancies have not been declared or are invalid"""
        ZERO_TEMPERATURE = 'Temperature cannot be zero'
        """Warns that the simulation is being run at absolute zero"""
        ZERO_BOLTZMANN_CONSTANT = 'Boltzmann constant cannot not be zero'
        """Warns that the Boltzmann constant is zero"""
        NO_TRANSITIONS = 'Barriers table is blank'
        """Warns that no barriers data is present"""
        NO_ISSUES = 'No issues found.  You should not be reading this.'
        """No warning"""

    def __init__(self, log_file: str = None, barriers_file: str = None,  vacancies: Iterable[int] = tuple(),
                 pick_vacancies: int = None, kB: float = 1.0, temperature: float = 1.0):
        """
        Create and return a new simulator object

        Arguments:
            log_file: Name of log file to output results to
            barriers_file: Name of file to load barriers data from
            vacancies: List of atom ids representing initial vacancy positions
            pick_vacancies: If provided, randomly converts N atoms into vacancies.  Any atom id already marked as a
                vacancy will not be chosen twice
            kB: Value of the Boltzmann constant to use
            temperature: Temperature to run the simulation at

        Returns:
            A new KMCSimulator object
        """

        # Use provided log output file
        if log_file is not None:
            self.log_file = str(log_file)
            self._log_file_enabled = True
        else:
            self._log_file_enabled = False
        self._log_file_created = False

        # Declare constants
        self.consts = Constants(kB=float(kB),
                                T=float(temperature))

        # Declare attributes
        self._barriers_source: str = ""
        self._max_len: int = 1

        # Attempt to load barriers file, if provided
        if barriers_file is not None:
            self.importBarriers(barriers_file)
        else:
            self._barriers_table: pd.DataFrame = pd.DataFrame(columns=[self.MOVING_ATOM_ID,
                                                                       self.VACANT_ATOM_ID,
                                                                       self.ENERGY_BARRIER,
                                                                       self.BOUNDARY_FLAGS])

        # Parse the list of integers the user provided for explicit vacancies (if any)
        try:
            self._vacancies: List[int] = list([int(i) for i in vacancies])
        except TypeError:
            self._vacancies: List[int] = []

        # Randomly assign more vacancies if requested
        if pick_vacancies is not None:
            self.pickRandomVacancies(int(pick_vacancies))

    @property
    def vacancyCount(self) -> int:
        """The number of vacancies currently present in the simulation"""
        return len(self._vacancies)

    @property
    def barriersSource(self) -> str:
        """The file location(s) that the barriers data was loaded from"""
        return self._barriers_source

    def pickRandomVacancies(self, count: int) -> None:
        """
        Randomly assigns vacancies to the provided number of sites

        Arguments:
            count: Number of vacancies to create
        """
        # Get list of all known atom ids
        atom_ids = np.unique(np.concatenate([self._barriers_table[self.MOVING_ATOM_ID].values,
                                             self._barriers_table[self.VACANT_ATOM_ID].values]))

        # Remove already assigned vacancies from array
        atom_ids = atom_ids[~np.isin(atom_ids, self._vacancies)]

        # Make sure the number of vacancies makes sense
        if count < 0:
            raise ValueError("Cannot pick less than zero vacancies!")

        elif count > len(atom_ids):
            raise ValueError("To many vacancies to populate amongst the available sites")

        # Randomly assign vacancies
        additional_vacancies: List[int] = list(np.random.choice(atom_ids, count, replace=False))
        self._vacancies += additional_vacancies

    def setBoltzmannConstant(self, value: float) -> None:
        """
        Sets the value of the Boltzmann constant used in the Arrhenius equation.

        Arguments:
            value: New Boltzmann constant to use.  Should be a float.
        """
        self.consts.kB = float(value)

    def setTemperature(self, value: float) -> None:
        """
        Sets the value of the temperature used in the Arrhenius equation

        Arguments:
            value: New temperature constant to use.  Should be a float.
        """
        self.consts.T = float(value)

    def setLogFile(self, new_destination: Union[str, None]) -> None:
        """
        Sets or disabled the destination file for log output

        Arguments:
            new_destination: New destination of log, or 'None' to disable logging entirely
        """
        if new_destination is None:
            self.log_file = None
            self._log_file_enabled = False
        else:
            self.log_file = str(new_destination)
            self._log_file_enabled = True

    def importBarriers(self, barriers_file: str):
        """
        Loads energy barrier data from the provided file

        Arguments:
            barriers_file: The name of the .csv file to load from.  Expected format is: 'Moving Atom ID,
        Vacant Atom ID,Starting Vacancy ID,Energy Barrier,Boundary Flags'.  'Starting Vacancy ID' and 'Boundary Flags'
        are not required.
        """
        # Attempt to load from file
        data = pd.read_csv(barriers_file, index_col=False)

        # Remove whitespaces from column names
        data.columns = [str(i).strip() for i in data.columns]

        # Get important columns from file and convert to their expected types
        moving_atom_id = data[KMCSimulator.MOVING_ATOM_ID].values.astype(int)
        vacant_atom_id = data[KMCSimulator.VACANT_ATOM_ID].values.astype(int)
        energy_barrier = data[KMCSimulator.ENERGY_BARRIER].values.astype(float)
        boundary_flags = data[KMCSimulator.BOUNDARY_FLAGS].values.astype(int)

        # Compile into main lookup table and store to internal variable
        self._barriers_table = pd.DataFrame({KMCSimulator.MOVING_ATOM_ID: moving_atom_id,
                                             KMCSimulator.VACANT_ATOM_ID: vacant_atom_id,
                                             KMCSimulator.ENERGY_BARRIER: energy_barrier,
                                             KMCSimulator.BOUNDARY_FLAGS: boundary_flags})

        # Remember barriers source
        self._barriers_source = barriers_file

        # Get max atom id character length for formatting in the logs file
        self._max_len = len(str(int(max(self._barriers_table[KMCSimulator.MOVING_ATOM_ID].max(),
                                        self._barriers_table[KMCSimulator.VACANT_ATOM_ID].max()))))

    def getPossibleTransitions(self, vacant_atom: int) -> pd.DataFrame:
        """
        Returns a table of possible transitions from the given vacant atom

        Arguments:
            vacant_atom: Atom ID of vacancy that will be moving
        Returns:
            Dataframe of possible transitions
        """
        # Get all transitions that reference this atom as the vacancy
        allowed = self._barriers_table[self._barriers_table[KMCSimulator.VACANT_ATOM_ID] == vacant_atom]

        # Remove all transitions that would lead to an existing vacancy
        exclude = np.isin(allowed[KMCSimulator.MOVING_ATOM_ID], self._vacancies)
        allowed = allowed[~exclude]

        # Return the result
        return allowed

    def calcTransitionProb(self, transition: pd.DataFrame) -> float:
        """
        Computes to probability of a state change based on the provided barrier and current internal constants.  Used
        during simulation runtime.

        Arguments:
            transition: Row from the barriers dataframe corresponding to the transition were calculating the
        probability of.  This must have an 'Energy Barrier' column.

        Returns:
            Probability of state change occurring
        """
        # Get energy barrier
        energy_barrier = float(transition[self.ENERGY_BARRIER])
        energy_barrier += self.calcVacancyRepulsion()

        # Compute probability using Arrhenius equation and constants
        return np.exp(-energy_barrier / (self.consts.kB * self.consts.T))

    def calcVacancyRepulsion(self) -> float:
        """
        Computes the energy associated with vacancy repulsion.  Override in subclass.

        Returns:
            A float representing an offset to the energy barrier
        """
        return 0.0

    def runInitialChecks(self) -> Tuple[bool, str]:
        """
        Performs a series of checks to ensure the current simulation settings are valid

        Returns:
            A tuple containing a boolean (indicating whether the verification passed) and a string (indicating a
        reason for that status)
        """
        # Check constants
        if self.consts.kB == 0.0:
            return False, self.ValidationStatus.ZERO_BOLTZMANN_CONSTANT.value
        if self.consts.T == 0.0:
            return False, self.ValidationStatus.ZERO_TEMPERATURE.value

        # Check initial vacancies
        if len(self._vacancies) == 0:
            return False, self.ValidationStatus.INVALID_STARTING_VACANCIES.value

        # No barriers
        if len(self._barriers_table) == 0:
            return False, self.ValidationStatus.NO_TRANSITIONS.value

        # All tests passed
        return True, self.ValidationStatus.NO_ISSUES.value

    def run(self, timesteps: int):
        """
        Runs the simulation for the given number of timesteps.

        Arguments:
            timesteps: Number of timesteps to run the simulation for.  If -1, the simulation will run forever until
        terminated by an external program.
        """
        # Create log file if it doesn't already exist
        if self._log_file_enabled and not self._log_file_created:
            with open(self.log_file, mode='w', newline='\n') as wf:
                # Write header
                lines = [f"### KMC Simulator {__version__} ###\n",
                         f"# {LogParamNames.CONSTANTS}:\n",
                         f"#     kB={self.consts.kB}, T={self.consts.T}\n",
                         f"# {LogParamNames.INITIAL_VACANCY_IDS}:\n",
                         f"#     {', '.join([str(i) for i in self._vacancies])}\n",
                         f"# {LogParamNames.BARRIERS_SOURCE_FILE}:\n",
                         f"#     {self.barriersSource}\n",
                         f"# {LogParamNames.LOG_FORMAT}:\n",
                         f"#     History\n"]

                # Modify heading separator
                sep_length = max(map(len, lines))
                lines[0] = lines[0][:-1] + '#'*(max(0, sep_length-(len(lines[0])-1))) + '\n'

                wf.writelines(lines)
                wf.write('#'*sep_length + '\n')

            self._log_file_created = True

        # Run initial checks before starting simulation.  Raise an exception if it didn't pass
        status, reason = self.runInitialChecks()
        if not status:
            if self._log_file_enabled:
                with open(self.log_file, mode='a', newline='\n') as wf:
                    wf.write(f'[FATAL]: Simulation failed initial check: {reason}\n')
            raise SimValidationException(reason)

        # Open log file for using during simulation
        if self._log_file_enabled:
            log_file = open(self.log_file, mode='a', newline='\n')
            # log_file.write("Transition Type, Initial Vacancy ID, Final Vacancy ID\n")
            column_line = ""
            for i in range(self.vacancyCount):
                column_line += TransitionColumns.TRANSITION_TYPE % (i+1) + ", "
                column_line += TransitionColumns.TRANSITION_ARGS % (i+1) + ", "
                column_line += TransitionColumns.VACANCY % (i+1) + ", "
            log_file.write(column_line[:-2] + "\n")
        else:
            log_file = None

        # Main loop
        t = int(timesteps)
        while t != 0:
            # Pick a random vacancy
            new_line = ""
            for chosen_index, vacancy_id in enumerate(self._vacancies):
                # chosen_index = randrange(0, len(self._vacancies))
                # vacancy_id = self._vacancies[chosen_index]

                # Get a list of possible transitions from barriers table.  These should have already filtered out
                # vacancy → vacancy transitions
                possible_transitions = self.getPossibleTransitions(vacancy_id)

                # Pick a random transition for that vacancy
                if len(possible_transitions) > 0:
                    transition = possible_transitions.sample()
                    # Accept or reject based on some probability
                    passing_prob = self.calcTransitionProb(transition)
                else:
                    # No transitions available, declare transition as "rejected"
                    passing_prob = -1.0

                if random() < passing_prob:
                    # Accept transition
                    moving_atom_id = int(transition[self.MOVING_ATOM_ID])
                    self._vacancies[chosen_index] = moving_atom_id

                    # Print state change to log file
                    if self._log_file_enabled:
                        # Get boundary crossing flags if they exist
                        try:
                            boundary_flags = int(transition[self.BOUNDARY_FLAGS])
                        except (KeyError, IndexError, ValueError, TypeError):
                            boundary_flags = 0
                        new_line += f'{TransitionTypes.VACANCY_TO_LATTICE_SITE.value}, '
                        new_line += f'{boundary_flags}, '
                        new_line += f'{moving_atom_id: >{self._max_len}}, '

                else:
                    # Reject transition
                    # Print state change (or lack thereof) to log file
                    if self._log_file_enabled:
                        new_line += f'{TransitionTypes.NO_CHANGE.value}, '
                        new_line += f'0, '
                        new_line += f'{vacancy_id: >{self._max_len}}, '

            # Advance timestep
            t -= 1

            # Force the log file to update
            log_file.write(new_line[:-2] + '\n')
            log_file.flush()

        # Close the log file
        if self._log_file_enabled:
            log_file.close()
