"""Functions to determine the positions of nuclides (nodes) when plotting the graph."""
import numpy as np
from numpy import array as ary
from numpy import pi; tau = 2*pi

def vogels_model(num_points):
    """
    Tessellate points on a unit circle (radius 1) using a vogel's algorithm,
    to generate a fermat spiral of points for tessellating a circle with points.
    It's the mathematically beautiful (and probably optimal too) way
    to distribute points sparsely that at the same time
    minimizing the number of edges which are overlapping AND (almost-) parallel with each other.
    """
    # planning: we need num_points points between 
    from scipy.constants import golden
    radius = np.sqrt(np.linspace(0, 1, num_points))
    angle = np.arange(num_points) * tau * (1-golden) # I like using the negative version of the golden ratio more.
    return np.array([radius * np.sin(angle), radius * np.cos(angle)]).T

def assert_isclose_mean(*args, varname="parameter"):
    """
    When the same variable are calculated more than once via multiple ways, they must equal each other.
    This function checks that these multiple results are numerically equal,
        and returns the mean of them if there are some small numerical imprecision involved.

    *args : Each of which is a number, or a numpy array of identical shapes.
        positional only argument.
    varname : the name of the parameter to be displayed at the AssertionError if they don't match.
    """
    if len(args)>2:
        return np.mean([assert_isclose_mean(a, b, varname=varname) for a, b in zip(args, np.roll(args, 1, axis=0))], axis=0)
    elif len(args)==2:
        a, b = args
        comparison = np.isclose(a, b, atol=0)
        if isinstance(comparison, bool):
            assert comparison, varname+" calculated in two different ways must match!"
        else:
            # must be an array if it's not a bool
            assert comparison.all(), varname+" calculated in two different ways must match!"
        return np.mean([a, b], axis=0)

class Disk():
    """
    A circle object, defined by at most 3 points.
    A membership method allows us to check if a new point lies within the circle or not.
    d = Disk (...)
    [x, y] in d # returns boolean
    """
    def __init__(self, points):
        """
        Build a Disk from object from 1 to 3 points.
        3 is the maximum number of points one can use to define a circle uniquely.
        case of 2 points: center = midpoint, diameter = distance between two points
        case of 1 point: trivial. Radius = 0
        """
        points = ary(points).reshape([-1, 2])
        if len(points)==1:
            self.center = points[0]
            self.radius = 0.0
        elif len(points)==2:
            self.center = points[0]/2 + points[1]/2
            self.radius = np.linalg.norm(self.center - points[0])
        elif len(points)==3:
            self.center = points
            nextpoints = np.roll(points, 1, axis=0)
            midpoints = points/2 + nextpoints/2
            directions = points - nextpoints
            bisector_dir = (ary([[0, -1], [1, 0]]) @ (directions.T) ).T
            def get_l(anchor1, dir1, anchor2, dir2):
                try:
                    return np.linalg.solve(ary([dir1, -dir2]).T, -anchor1 + anchor2)
                except np.linalg.LinAlgError:
                    return 0.0
            l1, _l2 = get_l(midpoints[0], bisector_dir[0], midpoints[1], bisector_dir[1])
            l2, _l3 = get_l(midpoints[1], bisector_dir[1], midpoints[2], bisector_dir[2])
            l3, _l1 = get_l(midpoints[2], bisector_dir[2], midpoints[0], bisector_dir[0])
            # make sure the floating point error isn't too big.
            c1 = midpoints[0] + assert_isclose_mean(l1, _l1, varname="lambda 1") * bisector_dir[0]
            c2 = midpoints[1] + assert_isclose_mean(l2, _l2, varname="lambda 2") * bisector_dir[1]
            c3 = midpoints[2] + assert_isclose_mean(l3, _l3, varname="lambda 3") * bisector_dir[2]
            self.center = assert_isclose_mean(c1, c2, c3, varname="center point of disk")
            self.radius = assert_isclose_mean(*[np.linalg.norm(self.center - p) for p in points], varname="radius")
            # vectorize the above.
            # and then find the stopping point that these parameter (l)'s represents
        else:
            raise ValueError(f"{len(points)} points is an invalid number of points for uniquely defining a circle!")

    @classmethod
    def from_parameters(cls, center, radius):
        self = cls.__new__(cls)
        self.center = center
        self.radius = radius
        return self

    def __contains__(self, test_point):
        """Membership test: check if a point lies within this disk."""
        distance = np.linalg.norm(ary(test_point) - self.center)
        return distance <= self.radius

    def __str__(self):
        return f"Disk centered at {self.center} with radius {self.radius}"

    def plot(self, ax):
        line = np.linspace(0, tau, 200)
        ax.plot(self.center[0]+self.radius*np.cos(line), self.center[1]+self.radius*np.sin(line))

if __name__=="__main__":
    import matplotlib.pyplot as plt
    radii = []
    for num_points in range(1, 200):
        spiral = vogels_model(num_points)
        disk = Disk(spiral[-3:])
        radii.append(disk.radius)
    plt.plot(radii)
    plt.show()