"""Define its behaviour on the command line. (e.g. `foilselector step? ...`)."""

from pathlib import Path

import click

from foilselector.script.input import main as main_step1
from foilselector.script.simulate import main as main_step2
from foilselector.script.step3 import main as main_step3


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
    """Interact with the user to convert the neutorn spectrum into the desired input
    format and group structure.
    """
    print(f"Acting on directory {Path.cwd()}")
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
    help=(
        "directory(ies) where the nuclear data (i.e. cross-sections and decay data) "
        "are stored"
    ),
    multiple=True,
)
@click.option(
    "-I",
    "--irradiation-duration",
    type=float,
    required=True,
    help=(
        "Number of seconds the foils spend getting activated in the neutron field "
        "(in the beamline/reactor)."
    ),
)
@click.option(
    "-T",
    "--transit-duration",
    type=float,
    required=True,
    help=(
        "Number of seconds the required to get the foil out of the neutron field onto "
        "the gamma detector."
    ),
)
@click.option(
    "-D",
    "--measurement-duration",
    type=float,
    required=True,
    help=(
        "Number of seconds the detector spend acquiring a spectrum of the activated "
        "foil sample."
    ),
)
@click.option(
    "-G",
    "--gamma-spectrum-parameters",
    type=float,
    required=False,
    help=(
        "If provided, a simulated gamma-ray spectrum will be generated, at gamma "
        "energies = np.arange(*gamma_spectrum_parameters) keV, where G specifies "
        "minimum gamma-ray energy, maximum gamma-ray energy, and the step size of "
        "the gamma-ray simulation."
    ),
    nargs=3,
)
def step2(
    composition,
    library,
    irradiation_duration,
    transit_duration,
    measurement_duration,
    gamma_spectrum_parameters,
):
    """Extract the relevant cross-sections and decay data from the nuclear data library.
    Then save them as functionst ath will never be used again.
    This is analogous to the 'collapse' and 'condense' step in FISPACT.
    """
    main_step2(
        composition,
        library,
        irradiation_duration,
        transit_duration,
        measurement_duration,
        gamma_spectrum_parameters,
    )


@cli.command("step3", no_args_is_help=True)
@click.option(
    "-N",
    "--number-of-foils",
    type=int,
    required=True,
    help=("Number of foils per foil-set. Foil set requires that 100%" "foil sample."),
)
def step3(number_of_foils):
    """Calculate the number of decays from each reaction."""
    main_step3(number_of_foils)


@cli.command("step4", no_args_is_help=False)
@click.argument("filepath", type=click.Path(exists=True))
def step4(filepath):
    """TODO help docs."""
