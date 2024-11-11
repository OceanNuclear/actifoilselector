import click
from pathlib import Path
from foilselector.script.step1 import main as main_step1
from foilselector.script.step2 import main as main_step2
from foilselector.script.step3 import main as main_step3
import os


@click.group()
@click.version_option()
def cli():
    """Foil selector cli

    Tools and scripts used to read nuclear data and thus select foils used in irradiation
    """


@cli.command("step1", no_args_is_help=False)
@click.argument("filepath", type=click.Path(exists=True), default=Path.cwd())
def step1(filepath):
    """Interact with the user to convert the neutorn spectrum into the desired input format and group structure."""
    print("Acting on directory {}".format(filepath))
    main_step1(filepath)


@cli.command("step2", no_args_is_help=True)
@click.option(
    "-c",
    "--composition",
    type=click.Path(exists=True),
    required=True,
    help=".json file specifiying the composition.",
)
@click.option(
    "-L",
    "--library",
    type=click.Path(exists=True),
    required=True,
    help="directory(ies) where the cross-sections are stored",
    multiple=True,
)
@click.option(
    "-e",
    "--photopeak-efficiency",
    type=click.Path(exists=True),
    help="""file with data of the absolute photopeak efficiency of the gamma detector used in its current configuration.
The accepted file types are:
.csv (energy in MeV in column 0, efficiency in column 1.)
.dat (same as .csv, but space delimited instead)
.o (mcnp output)
.ecc (GENIE/ISOCS output) """,
    default=Path(
        os.path.dirname(__file__),
        "physicalparameters",
        "photopeak_efficiency",
        "Absolute_photopeak_efficiencyMeV.csv",
    ),
)
@click.option(
    "-g",
    "--gamma-energy-limits-keV",
    type=float,
    nargs=2,
    help="The minimum and maximum gamma energies (keV) that the detector can detect.\nThe defaults are 20 keV - 4600 keV.",
)
@click.option(
    "-G",
    "--group-structure",
    type=click.Path(exists=True),
    help="newline-separated file listing the pairs of group boundaries (comma-separated) in ascending energies.",
)
def step2(
    composition, library, photopeak_efficiency, gamma_energy_limits_keV, group_structure
):
    """Extract the relevant cross-sections and decay data from the nuclear data library.
    Then save them as functionst ath will never be used again.
    This is analogous to the 'collapse' and 'condense' step in FISPACT."""
    main_step2(
        composition,
        library,
        photopeak_efficiency,
        gamma_energy_limits_keV,
        group_structure,
    )


@cli.command("step3", no_args_is_help=True)
@click.option(
    "-f",
    "--a-priori-flux",
    type=click.Path(exists=True),
    required=True,
    help="""newline-separated file listing the total flux expected in each bin of ascending energy.
The number of bins (n) must match the number of bin boundaries (n+1) in the previous step.""",
)
@click.option(
    "-R",
    "--max-gamma-count-rate",
    type=float,
    default=10000,
    help="Maximum pulse rate that the gamma detector can handle without losing its resolution.",
)
@click.option(
    "-I",
    "--irradiation-duration",
    type=float,
    required=True,
    help="Number of seconds the foils spend getting activated in the neutron field (in the beamline/reactor).",
)
@click.option(
    "-T",
    "--transit-duration",
    type=float,
    required=True,
    help="Number of seconds the required to get the foil out of the neutron field onto the gamma detector.",
)
@click.option(
    "-D",
    "--measurement-duration",
    type=float,
    required=True,
    help="Number of seconds the detector spend acquiring a spectrum of the activated foil sample.",
)
def step3(
    a_priori_flux,
    max_gamma_count_rate,
    irradiation_duration,
    transit_duration,
    measurement_duration,
):
    """Calculate the number of decays from each reaction."""
    main_step3(
        a_priori_flux,
        max_gamma_count_rate,
        irradiation_duration,
        transit_duration,
        measurement_duration,
    )


@cli.command("step4", no_args_is_help=False)
@click.argument("filepath", type=click.Path(exists=True))
def step4(filepath):
    """TODO help docs"""
