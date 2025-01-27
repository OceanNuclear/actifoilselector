"""Define its behaviour on the command line. (e.g. `foilselector step? ...`)."""

from collections.abc import Iterable
from pathlib import Path

import click

from foilselector.script.input import main as input_preparation_script
from foilselector.script.simulate import main as simulate_script


@click.group()
@click.version_option()
def cli() -> None:
    """
    Foil selector CLI
    Tools and scripts used to read nuclear data and thus select foils used in activation
    foil unfolding experiments.
    """


@cli.command("step1", no_args_is_help=False)
def step1() -> None:
    """Interact with the user to convert the neutorn spectrum into the desired input
    format and group structure.
    """
    print(f"Acting on directory {Path.cwd()}")
    input_preparation_script()


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
    composition: Path,
    library: Iterable[Path],
    irradiation_duration: float,
    transit_duration: float,
    measurement_duration: float,
    gamma_spectrum_parameters: tuple[float, float, float],
) -> None:
    """Extract the relevant cross-sections and decay data from the nuclear data library.
    Then save them as functionst ath will never be used again.
    This is analogous to the 'collapse' and 'condense' step in FISPACT.
    """
    simulate_script(
        composition,
        library,
        irradiation_duration,
        transit_duration,
        measurement_duration,
        gamma_spectrum_parameters,
    )
