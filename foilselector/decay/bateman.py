import numpy as np
from numpy import array as ary
import uncertainties.unumpy as unpy
import uncertainties as unc
import scipy.linalg as spln

def kahan_sum(a, axis=0):
    """
    Carefully add together sum to avoid floating point precision problem.
    Retrieved (and then modified) from
    https://github.com/numpy/numpy/issues/8786
    """
    s = np.zeros(a.shape[:axis] + a.shape[axis+1:])
    c = np.zeros(s.shape)
    for i in range(a.shape[axis]):
        # http://stackoverflow.com/42817610/353337
        y = a[(slice(None),) * axis + (i,)] - c
        t = s + y
        c = (t - s) - y
    return t

# population variation due to decay after flash irradiation, calculated using the Bateman equation
def Bateman_equation_generator(branching_ratios, decay_constants, decay_constant_threshold=1E-23):
    """
    Generate the expression that calculates the radioisotope population at the end of the chain specified using the Bateman equation.
    Assume flash irradiation creating 1.0 of the root(parent?) isotope.

    branching_ratios : array
        the list of branching ratios of all its parents, and then itself, in the order that the decay chain was created.
    decay_constants: the list of decay constants in the linearized decay chain.
    Reduce decay_constant_threshold when calculating on very short timescales.
    """
    # assert len(branching_ratios)==len(decay_constants), "Both are lists of parameters describing the entire pathway of the decay cahin up to that isotope."
    if any([i<=decay_constant_threshold for i in decay_constants[:-1]]):
        return lambda x: x-x # Zero. Because any near-zero decay constant in the preceeding chain will lead to near-zero production rate of this isotope.
        # It's expressed this way to ensure the data output format matches the input format.
    if len(decay_constants)==1 and decay_constant[0]<=decay_constant_threshold:
        return lambda x: x-x+1 # stable population, no decay.
    # expand out the Bateman equation into a matrix, we can then multiply the columns together and then summed the rows.
    premultiplying_factor = np.product(decay_constants[:-1])*np.product(branching_ratios[1:])
    upside_down_matrix = np.diff(np.meshgrid(decay_constants, decay_constants)[::-1], axis=0)
    upside_down_matrix += np.diag(np.ones(len(decay_constants))) # 1/(lambda_i - lambda_j) term doesn't exist if i=j.
    final_matrix = 1/upside_down_matrix
    multiplying_factors = np.product(final_matrix, axis=-1) # multiply across columns
    def calculate_population(t):
        vector = (unpy.exp(-ary(decay_constants)*max(0,t))
                # -unpy.exp(-ary(decay_constants)*(c-a))
                # -unpy.exp(-ary(decay_constants)*b)
                # +unpy.exp(-ary(decay_constants)*(b-a))
                )
        return premultiplying_factor * (multiplying_factors @ vector) # sum across rows
    return calculate_population

# population variation due to decay after a drawn out irradiation schedule, calculated using the Bateman equation
def Bateman_convolved_generator(branching_ratios, decay_constants, a, decay_constant_threshold=1E-23):
    """
    Generate the expression that calculates the radioisotope population at the end of the chain specified.
    Assume a drawn out irradiation schedule between 0 to t, creating a gross 1.0 unit of the root (parent?) isotope.
    branching_ratios : array
        the list of branching ratios of all its parents, and then itself, in the order that the decay chain was created.
    decay_constants: the list of decay constants in the linearized decay chain.

    a : scalar
        End of irradiation period.
        Irradiation schedule = from 0 to a, 
        with rrdiation power = 1/a, so that total amount of irradiation = 1.0.
    Reduce decay_constant_threshold when calculating on very short timescales.
    """
    if any([i<=decay_constant_threshold for i in decay_constants[:-1]]):
        return lambda x: x-x # catch the cases where there are zeros in the decay rates.
    if len(decay_constants)==1 and decay_constant[0]<=decay_constant_threshold:
        return lambda x: x-x+1 # stable population, no decay.
    premultiplying_factor = np.product(decay_constants[:-1])*np.product(branching_ratios[1:])/a
    upside_down_matrix = np.diff(np.meshgrid(decay_constants, decay_constants)[::-1], axis=0)
    upside_down_matrix += np.diag(decay_constants)
    final_matrix = 1/upside_down_matrix
    multiplying_factors = np.product(final_matrix, axis=-1)
    def calculate_convoled_population(t):
        """
        Calculates the population at any given time t when a non-flash irradiation schedule is used,
        generated using irradiation duration a={} seconds
        """.format(a)
        vector_uncollapsed = ary([+unpy.exp(-ary([l*np.clip(t-a,0, None) for l in decay_constants])), -unpy.exp(-ary([l*np.clip(t,0, None) for l in decay_constants]))], dtype=object)
        vector = np.sum(vector_uncollapsed, axis=0)
        return premultiplying_factor*(multiplying_factors@vector)
    return calculate_convoled_population

# integrate(population * dt) (dimension: population * time) quantity obtained due to decay after a drawn out irradiation schedulde.
# remember that decay_const*(integrate (population * dt)) = total number of decays experienced for that duration.
def Bateman_num_decays_factorized(branching_ratios, decay_constants, a, b, c, DEBUG=False, decay_constant_threshold=1E-23):
    """
    Calculates the amount of decay radiation measured using the Bateman equation.
    Created for comparison/benchmarking against the other two matrix equation to know that they're correct.
    branching_ratio : array
        the list of branching ratios of all its parents, and then itself, in the order that the decay chain was created.
    a : scalar
        End of irradiation period.
        Irrdiation power = 1/a, so that total amount of irradiation = 1.
    b : scalar
        Start of gamma measurement time.
    c : scalar
        End of gamma measurement time.
    The initial population is always assumed as 1.
    Reduce decay_constant_threshold when calculating on very short timescales.
    """
    # assert len(branching_ratios)==len(decay_constants), "Both are lists of parameters describing the entire pathway of the decay chain up to that isotope."
    if any([i<=decay_constant_threshold for i in decay_constants]):
        if DEBUG: print(f"{decay_constants=} \ntoo small, escaped.")
        return Variable(0.0, 0.0) # catch the cases where there are zeros in the decay_rates
            # in practice this should only happen for chains with stable parents, i.e.
            # this if-condition would only be used if the decay chain is of length==1.    
    premultiplying_factor = np.product(decay_constants[:])*np.product(branching_ratios[1:])/a
    upside_down_matrix = np.diff(np.meshgrid(decay_constants, decay_constants)[::-1], axis=0)[0]
    # upside_down_matrix += np.diag(decay_constants)**2
    upside_down_matrix += np.diag(np.ones(len(decay_constants)))
    final_matrix = 1/upside_down_matrix
    multiplying_factors = np.product(final_matrix, axis=-1)
    try:
        vector_uncollapsed = ary([1/ary(decay_constants)**2
                        *unpy.expm1(ary(decay_constants)*a)
                        *unpy.expm1(ary(decay_constants)*(c-b))
                        *unpy.exp(-ary(decay_constants)*c) ], dtype=object)
    except OverflowError:
        # remove the element that causes such an error from the chain, and recompute
        decay_constants_copy = decay_constants.copy()
        for ind, l in list(enumerate(decay_constants))[::-1]:
            try:
                ary([1/ary(l)**2
                        *unpy.expm1(ary(l)*a)
                        *unpy.expm1(ary(l)*(c-b))
                        *unpy.exp(-ary(l)*c) ])
            except OverflowError: # catch all cases and pop them out of the list
                decay_constants_copy.pop(ind)

        return Bateman_num_decays_factorized(branching_ratios, decay_constants_copy, a, b, c, DEBUG, decay_constant_threshold)
    vector = np.sum(vector_uncollapsed, axis=0)
    if DEBUG:
        print(#upside_down_matrix, '\n', final_matrix, '\n',
        'multiplying_factors=\n', multiplying_factors,
        #'\n--------\nvector=\n', vector,
        )
        try:
            print("Convolution term = \n", unpy.expm1(ary(decay_constants)*a)/ary(decay_constants))
            print("measurement (integration) term\n", unpy.expm1(ary(decay_constants)*(c-b))/ary(decay_constants))
            print("end of measurement term\n", unpy.exp(-ary(decay_constants)*c))
        except:
            print("Overflow error")
    return premultiplying_factor * (multiplying_factors @ vector)

# decay from drawn out irradiation, calculated using matrix exponentiation
def create_lambda_matrix(l_vec, decay_constant_threshold=1E-23):
    lambda_vec = []
    for lamb_i in l_vec:
        if isinstance(lamb_i, uncertainties.core.AffineScalarFunc):
            lambda_vec.append(lamb_i.n)
        else:
            lambda_vec.append(float(lamb_i))
    return - np.diag(lambda_vec, 0) + np.diag(lambda_vec[:-1], -1)

_expm = lambda M: spln.expm(M)

def decay_mat_exp_population_convolved(branching_ratios, decay_constants, a, t, decay_constant_threshold : float = 1E-23):
    """
    Separated cases out that that will cause singular matrix or 1/0's.
    """
    if any(ary(decay_constants[:-1])<=decay_constant_threshold): # any stable isotope in the chain:
        return 0
    elif len(decay_constants)==1 and decay_constants[0]<=decay_constant_threshold: # single stable isotope in chain:
        return 1.0
    elif decay_constants[-1]<=decay_constant_threshold: # last isotope is stable; rest of the chain is unstable:
        if t<a:
            raise NotImplementedError("The formula for population during irradiation hasn't been properly derived yet.")
        matrix, iden = create_lambda_matrix(decay_constants[:-1]), np.identity(len(decay_constants)-1)
        inv = np.linalg.inv(matrix)
        initial_population_vector = ary( [1.0,]+[0.0 for _ in decay_constants[1:-1]] )
        
        # during_irradiation_production_matrix = -1/a * inv @ ( a*iden - inv @ (_expm(-matrix*a) - iden) )
        during_irradiation_production_matrix = -inv + 1/a * (_expm(matrix*a) - iden) @ inv @ inv
        during_irradiation_production = (during_irradiation_production_matrix @ initial_population_vector)[-1] * decay_constants[-2] * np.product(branching_ratios)
        post_irradiation_production   = decay_mat_exp_num_decays(branching_ratios, decay_constants[:-1], a, a, t)
        return during_irradiation_production + post_irradiation_production
    else: # all unstable:
        matrix, iden = create_lambda_matrix(decay_constants), np.identity(len(decay_constants))
        inv = np.linalg.inv(matrix)
        initial_population_vector = ary( [1.0,]+[0.0 for _ in decay_constants[1:]] )

        transformation = 1/a * _expm(matrix*np.clip(t-a, 0, None)) @ (_expm(matrix*np.clip(t, 0, a)) - iden) @ inv
        final_fractions = transformation @ initial_population_vector
        return np.product(branching_ratios[1:]) * final_fractions[-1]

def decay_mat_exp_num_decays(branching_ratios, decay_constants, a, b, c, decay_constant_threshold=1E-23):
    """
    decay_constants : a list of decay_constants
    a : the end time of irradiation (irradiation time starts at t=0), a> 0
    b : the start time of measurement, b> a
    c : the end time of measurement, c> b
    decay_constant_threshold : the threshold below which nuclides are considered as stable.
    """
    if any(ary(decay_constants)<=decay_constant_threshold):
        return 0
    matrix = create_lambda_matrix(decay_constants)
    iden = np.identity(len(decay_constants))
    multiplier = 1/a * (_expm(matrix*(b-a))) @ (_expm(matrix*(c-b)) - iden) @ (_expm(matrix*(a)) - iden)
    inv = np.linalg.inv(matrix)
    initial_population_vector = ary( [1,]+[0 for _ in decay_constants[1:]] ) # initial population of all nuclides = 0 except for the very first isotope, which has 1.0.
    final_fractions = multiplier @ inv @ inv @ initial_population_vector # result of the (population * dt) integral
    # total number of decays = branching_ratios * the integral * decay constant of that isotope .
    return np.product(branching_ratios[1:]) * final_fractions[-1] * decay_constants[-1] # multiplied by its own decay rate will give the number of decays over time period b to c.