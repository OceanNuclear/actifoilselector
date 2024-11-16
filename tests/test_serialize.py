import json
from uncertainties.core import Variable, AffineScalarFunc
import numpy as np
from numpy import array as ary
from foilselector.openmcextension.table import Tab1DExtended
from foilselector.openmcextension.library_reader import DiscreteRadiation, ContinuousRadiationDistribution
from foilselector.openmcextension.extended_io import serialize_radiation_dict, deserialize_radiation_dict

a = ary([1.0,2.0])
b = ary([1])

sample = {
    "sample_foil":
    {
        DiscreteRadiation(Variable(1,0), Variable(1,1), "dummy-source-MT=..."):
            np.random.rand(5),
        ContinuousRadiationDistribution(Tab1DExtended(a, a, b), "dummy-source2-MT=1..."):
            np.random.rand(5)
    },
}
s = json.dumps(serialize_radiation_dict(sample))
sample2 = deserialize_radiation_dict(json.loads(s))
assert str(sample)==str(sample2), "Expected exactly the same numbers"