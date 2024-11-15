import click
from pathlib import Path
from foilselector.script.step1 import main as main_step1
from foilselector.script.step2 import main as main_step2
from foilselector.script.step3 import main as main_step3
import os


@click.group()
@click.version_option()
def cli():
    """
    Foil selector CLI
    Tools and scripts used to read nuclear data and thus select foils used in activation
    foil unfolding experiments.
    """


@cli.command("step1", no_args_is_help=False)
@click.argument("filepath", type=click.Path(exists=True), default=Path.cwd())
def step1():
    """Interact with the user to convert the neutorn spectrum into the desired input format and group structure."""
    print("Acting on directory {}".format(Path.cwd()))
    main_step1()


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
    help="directory(ies) where the nuclear data (i.e. cross-sections and decay data) are stored",
    multiple=True,
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
def step2(
    composition, library, irradiation_duration, transit_duration, measurement_duration
):
    """Extract the relevant cross-sections and decay data from the nuclear data library.
    Then save them as functionst ath will never be used again.
    This is analogous to the 'collapse' and 'condense' step in FISPACT."""
    main_step2(
        composition,
        library,
        irradiation_duration,
        transit_duration,
        measurement_duration,
    )


@cli.command("step3", no_args_is_help=True)
@click.option(
    "-n",
    type=float,
    required=True,
    help="Number of foil required in the final foil set."
)
@click.option(
    "-r",
    "--max-gamma-count-rate",
    type=float,
    required=True,
    # default=10000,
    help="Maximum pulse rate that the gamma detector can handle without losing its resolution.",
)
def step3(max_gamma_count_rate):
    """Calculate the number of decays from each reaction."""
    main_step3(max_gamma_count_rate)


@cli.command("step4", no_args_is_help=False)
@click.argument("filepath", type=click.Path(exists=True))
def step4(filepath):
    """TODO help docs"""
