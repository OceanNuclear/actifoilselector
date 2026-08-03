"""Test the extended_io module."""

import json

import numpy as np
from numpy import array as ary
from uncertainties.core import Variable

from foilselector.openmcextension.extended_io import (
    deserialize_radiation_dict,
    serialize_radiation_dict,
)
from foilselector.openmcextension.library_reader import (
    ContinuousRadiationDistribution,
    DiscreteRadiation,
)
from foilselector.openmcextension.table import Tab1DExtended

a = ary([1.0, 2.0])
b = ary([1])


def test_serialize_deserialize():
    """
    Test whether serializing and then deserailizing an variable with uncertainty
    leads to the same result.
    """
    rng = np.random.default_rng()
    sample = {
        "sample_foil": {
            DiscreteRadiation(
                Variable(1, 0),
                Variable(1, 1),
                "dummy-source-MT=...",
            ): rng.random(5),
            ContinuousRadiationDistribution(
                Tab1DExtended(a, a, b),
                "dummy-source2-MT=1...",
            ): rng.random(5),
        },
    }
    s = json.dumps(serialize_radiation_dict(sample))
    sample2 = deserialize_radiation_dict(json.loads(s))
    assert str(sample) == str(sample2), "Expected exactly the same numbers"
