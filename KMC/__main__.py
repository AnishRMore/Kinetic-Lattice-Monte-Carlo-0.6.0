"""
This program is the core kinetic Monte-Carlo simulator,
implemented in Python.  Written as part of Will Ebmeyer's
capstone project for Rochester Institute of Technology.

------------------------------------------------------------

Written and tested with:
• python 3.9.13
• numpy 1.23.3
• pandas 1.4.4

------------------------------------------------------------
"""
__author__ = 'Will Ebmeyer'
__version__ = 'v0.5.2'

# Import modules
import argparse
from random import seed
from sys import stderr, stdout
from textwrap import dedent
from .Exceptions import InvalidArguments


class ArgumentParser(argparse.ArgumentParser):
    """Overrides argparse's default behavior of exiting the program upon error"""

    def error(self, message):
        msg = '%s: error: %s\n' % (self.prog, message)
        stderr.write(msg)
        raise InvalidArguments(msg)


# Main command-line program
if __name__ == "__main__":
    # Create root parser
    try:
        parser = ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=dedent(__doc__))
        subparsers = parser.add_subparsers(dest='subcommand', required=True, help='Subcommand to run, whether it be a '
                                                                                  'KMC simulation or an analysis '
                                                                                  'thereof.')

        # Add subparser for the KMC simulator
        kmc_parser = subparsers.add_parser('KMC', description='Runs a KMC simulation')

        kmc_parser.add_argument('barriers', help='File name of energy barriers table', type=str)
        kmc_parser.add_argument('duration', help='Number of timesteps to run for.  Set to -1 to run forever', type=int)
        kmc_parser.add_argument('-v', '--vacancies', help='List of atom ids corresponding to the initial vacancies', type=int, nargs='*', default=None)
        kmc_parser.add_argument('-p', '--pick', help='Number of vacancies to randomly assign', type=int, default=None)
        kmc_parser.add_argument('-l', '--log', help='File name to output logs to', type=str, default=None)
        kmc_parser.add_argument('-k', '--kB', help='Value of the Boltzmann constant to use', type=float, default=1.0)
        kmc_parser.add_argument('-T', '--temperature', help='Temperature to run the simulation at', type=float, default=1.0)
        kmc_parser.add_argument('-s', '--seed', help='Random number generator seed', type=int, default=0)

        # Add subparser for Analysis computes
        compute_parser = subparsers.add_parser('analysis', description='Runs the provided analysis on the output data '
                                                                       'from a KMC simulation')

        compute_subparser = compute_parser.add_subparsers(dest='compute', help='Analysis to run')
        msd_subparser = compute_subparser.add_parser('MSD', help='Computes the mean-square-displacement for the KMC '
                                                                 'simulation')
        msd_subparser.add_argument('structure', help='File to load structure data from', type=str)
        msd_subparser.add_argument('transitions', help='File to load KMC transitions log from', type=str)
        msd_subparser.add_argument('-lx',
                                   help='X-width of structure.  Required for analysis on cross-boundary transitions if '
                                        'not already provided by structure file.',
                                   type=float, default=None)
        msd_subparser.add_argument('-ly',
                                   help='X-width of structure.  Required for analysis on cross-boundary transitions if '
                                        'not already provided by structure file.',
                                   type=float, default=None)
        msd_subparser.add_argument('-lz',
                                   help='X-width of structure.  Required for analysis on cross-boundary transitions if '
                                        'not already provided by structure file.',
                                   type=float, default=None)

        rejects_subparser = compute_subparser.add_parser('rejectRate', help='Computes the proportion of transitions '
                                                                            'that were rejected in the KMC simulation')
        rejects_subparser.add_argument('transitions', help='File to load KMC transitions log from', type=str)

        activation_E_subparser = compute_subparser.add_parser('meanE', help='Computes the average activation energy '
                                                                            'using the acceptation rate and the '
                                                                            'Arrhenius equation')
        activation_E_subparser.add_argument('transitions', help='File to load KMC transitions log from', type=str)

        diffusion_subparser = compute_subparser.add_parser('diff', help='Computes the diffusion coefficient from the '
                                                                        'KMC simulation results')
        diffusion_subparser.add_argument('transitions', help='File to load KMC transitions log from', type=str)
        diffusion_subparser.add_argument('structure', help='File to load structure data from', type=str)
        diffusion_subparser.add_argument('-dt', help='Assumed timestep for the simulation', type=float, default=1.0)
        diffusion_subparser.add_argument('-lx',
                                         help='X-width of structure.  Required for analysis on cross-boundary '
                                              'transitions if not already provided by structure file.',
                                         type=float, default=None)
        diffusion_subparser.add_argument('-ly',
                                         help='X-width of structure.  Required for analysis on cross-boundary '
                                              'transitions if not already provided by structure file.',
                                         type=float, default=None)
        diffusion_subparser.add_argument('-lz',
                                         help='X-width of structure.  Required for analysis on cross-boundary '
                                              'transitions if not already provided by structure file.',
                                         type=float, default=None)

        ionic_subparser = compute_subparser.add_parser('ionic', help='Computes the diffusion coefficient from the '
                                                                     'KMC simulation results')
        ionic_subparser.add_argument('transitions', help='File to load KMC transitions log from', type=str)
        ionic_subparser.add_argument('structure', help='File to load structure data from', type=str)
        ionic_subparser.add_argument('charge', help='Charge of the ionic carriers', type=float)
        ionic_subparser.add_argument('-dt', help='Assumed timestep for the simulation', type=float, default=1.0)
        ionic_subparser.add_argument('-lx',
                                     help='X-width of structure.  Required for analysis on cross-boundary '
                                          'transitions if not already provided by structure file.',
                                     type=float, default=None)
        ionic_subparser.add_argument('-ly',
                                     help='X-width of structure.  Required for analysis on cross-boundary '
                                          'transitions if not already provided by structure file.',
                                     type=float, default=None)
        ionic_subparser.add_argument('-lz',
                                     help='X-width of structure.  Required for analysis on cross-boundary '
                                          'transitions if not already provided by structure file.',
                                     type=float, default=None)

        stats_subparser = compute_subparser.add_parser('stats', help='Computes a handful of statistics that may be '
                                                                     'useful for quickly analyzing the simulation '
                                                                     'results')
        stats_subparser.add_argument('transitions', help='File to load KMC transitions log from', type=str)

        # Parse arguments
        args = parser.parse_args()

        # User chose to run a KMC simulation
        if args.subcommand == 'KMC':
            # Set RNG seed
            seed(args.seed)

            # Check if the user at least provided *something* for the vacancies
            if args.vacancies is None and args.pick is None:
                raise InvalidArguments("No initial vacancies provided.  Use -v (--vacancies) or -p (--pick)")

            from .Simulator import KMCSimulator

            # Run main program with provided arguments
            sim = KMCSimulator(log_file=args.log,
                               barriers_file=args.barriers,
                               vacancies=args.vacancies,
                               pick_vacancies=args.pick,
                               kB=args.kB,
                               temperature=args.temperature)
            sim.run(args.duration)

        # User chose to run a computation
        elif args.subcommand == 'analysis':
            # Import analysis module
            from .Analysis import KMCResults

            # Act on arguments
            if args.compute == 'MSD':
                # Run MSD analysis
                sim_results = KMCResults.fromFile(args.transitions)
                msd = sim_results.calcMSD(args.structure, args.lx, args.ly, args.lz)
                # Print to stdout
                stdout.write(str(tuple(msd)))

            elif args.compute == 'rejectRate':
                # Run computation
                sim_results = KMCResults.fromFile(args.transitions)
                rejects, total = sim_results.tallyRejects(), sim_results.tallyAttempts()
                # Print to stdout
                stdout.write(str(rejects / total))

            elif args.compute == 'meanE':
                # Run computation
                sim_results = KMCResults.fromFile(args.transitions)
                # Print to stdout
                stdout.write(str(sim_results.calcMeanActivationEnergy()))

            elif args.compute == 'diff':
                # Run computation
                sim_results = KMCResults.fromFile(args.transitions)
                # Print to stdout
                Dv = sim_results.calcDiffusion(args.structure, args.dt, args.lx, args.ly, args.lz)
                stdout.write(str(Dv))

            elif args.compute == 'ionic':
                # Run computation
                sim_results = KMCResults.fromFile(args.transitions)
                # Print to stdout
                Dv = sim_results.calcIonicConductivity(args.structure, args.dt, args.charge, args.lx, args.ly, args.lz)
                stdout.write(str(Dv))

            elif args.compute == 'stats':
                # Run computation
                sim_results = KMCResults.fromFile(args.transitions)
                result = ""
                for name, value in sim_results.stats().items():
                    result += f'{name}: {value}\n'
                # Print to stdout
                stdout.write(result)

    except InvalidArguments:
        pass
