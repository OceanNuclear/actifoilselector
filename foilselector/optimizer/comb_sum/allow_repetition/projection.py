"""
Generates the projection matrix required to uniquely project the N dimensional hypercube down to 2D.
"""

import numpy as np
from numpy import sqrt

def direction_111(N):
    """returns the direction pointing in the 1,1,1,1,1,1... etc."""
    return np.ones(N)

def direction_123(N):
    """ returns the direction pointing in the 1,2,3,4,... etc."""
    return np.arange(1, N+1)

def direction_s1s2s3(N):
    """ returns the direction pointing in the sqrt(1),sqrt(2),sqrt(3),sqrt(4),... etc."""
    return sqrt(np.arange(1, N+1))

def linear_indices(row_num):
    return np.arange(1, row_num+1)

def sqrt_indices(row_num):
    return sqrt(np.arange(1, row_num+1))

def stick_column_vector_onto_left_of_matrix(left_vector, matrix):
    column_vector = left_vector.reshape([len(left_vector), 1])
    return np.concatenate([ column_vector, matrix], axis=1)

def determine_next_vector_symmetric(bottom_matrix):
    """
    Assuming an N=num_dimensions hypercube needed to be projected;
    Assuming that n bottom rows of the projection matrix has been uniquely identified
        (i.e. the last n projection vectors has been calculated);
    This function identifies the next (n-1^th) row
        (i.e. calculates the n-1^th vector).

    Mathematically:
    We only need to provide the bottom n x n right corner to this algorithm,
    because by construction, we forces the indices in the bottom left corner to be duplicates of each other.
    e.g. for N=5, by construction
        P[1,0] = P[1,1]
        P[2,0] = P[2,1] = P[2,2]
        P[3,0] = P[3,1] = P[3,2] = P[3,3]
        P[4,0] = P[4,1] = P[4,2] = P[4,3] = P[4,4]
    
    Parameters
    ----------
    bottom_matrix : The nxn bottom right corner of the vector matrix.
    num_dimensions : The number of dimensions of the initial space we're projecting from..
    """
    corner_len, num_dimensions = bottom_matrix.shape
    num_missing_rows = num_dimensions - corner_len # number of repetition needed to fill in the beginning of the vector.
    corner = bottom_matrix[:, num_missing_rows:num_dimensions] # alias
    next_basis_vector = np.zeros(num_dimensions)

    # create a m-row,m+1-columns matrix from the bottom right corner of matrix by
    # duplicating its leftmost column to the left, and multiply that new left column by sqrt(num_missing_rows)
    extra_left_column = sqrt(num_missing_rows) * corner[:, 0]
    wedge_product_vectors = stick_column_vector_onto_left_of_matrix(extra_left_column, corner)[::-1]

    product_vector = - wedge_product(wedge_product_vectors)
    next_basis_vector[:num_missing_rows] = product_vector[0]/sqrt(num_missing_rows)
    next_basis_vector[num_missing_rows:] = product_vector[1:]
    return next_basis_vector

def determine_next_vector_linear(bottom_matrix):
    """
    Same as determine_next_vector_symmetric,
    but now we assume
    P[1,0] = P[1,1]/2
    P[2,0] = P[2,1]/2 = P[2,2]/3
    P[3,0] = P[3,1]/2 = P[3,2]/3 = P[3,3]/4
    P[4,0] = P[4,1]/2 = P[4,2]/3 = P[4,3]/4 = P[4,4]/5

    So we'll have to take in the bottom n rows rather than the nxn bottom right corner, since we can't assume they're duplicates.
    """
    num_vectors_calculated, num_dimensions = bottom_matrix.shape
    num_missing_rows = num_dimensions - num_vectors_calculated
    corner = bottom_matrix[:, num_missing_rows:num_dimensions] # alias
    next_basis_vector = np.zeros(num_dimensions)

    k = np.linalg.norm(linear_indices(num_missing_rows), 2) # L2 norm
    extra_left_column = k * bottom_matrix[:, 0]
    wedge_product_vectors = stick_column_vector_onto_left_of_matrix(extra_left_column, corner)[::-1]
    # term directly below i in the wedg-product matrix = (1+2+3+...)/k_a * n_1
    #   after translating into the index for a'
    # if we use a symmetric starting n (direction_111)
    wedge_product_vectors[0, 0] = sum(linear_indices(num_missing_rows))/k * bottom_matrix[-1, 0]
    print(wedge_product_vectors)

    product_vector = - wedge_product(wedge_product_vectors)
    next_basis_vector[:num_missing_rows] = linear_indices(num_missing_rows) * product_vector[0] / k
    next_basis_vector[num_missing_rows:] = product_vector[1:]
    return next_basis_vector

def determine_next_vector_sqrt(bottom_matrix):
    """
    Same as determine_next_vector_symmetric,
    but now we assume
    P[1,0] = P[1,1]/sqrt(2)
    P[2,0] = P[2,1]/sqrt(2) = P[2,2]/sqrt(3)
    P[3,0] = P[3,1]/sqrt(2) = P[3,2]/sqrt(3) = P[3,3]/sqrt(4)
    P[4,0] = P[4,1]/sqrt(2) = P[4,2]/sqrt(3) = P[4,3]/sqrt(4) = P[4,4]/sqrt(5)

    So we'll have to take in the bottom n rows rather than the nxn bottom right corner, since we can't assume they're duplicates.
    """
    raise NotImplementedError("Haven't been written yet.")

def wedge_product(matrix_formed_by_row_vectors):
    """Calculates the outerproduct/(AKA wedge product) of n-1 vectors each with len =s= n"""
    shape = matrix_formed_by_row_vectors.shape
    assert shape[1] == shape[0] + 1, f"must be a matrix of shape (n, n-1), not shape {shape}"

    product = np.zeros(shape[1]) # resultant product vector
    for i in range(shape[1]):
        left_matrix, right_matrix = matrix_formed_by_row_vectors[:, :i], matrix_formed_by_row_vectors[:, i+1:]
        resultant_square_matrix = np.concatenate([left_matrix, right_matrix], axis=1)
        product[i] = ((-1)**i) * np.linalg.det(resultant_square_matrix)
    return product

def projection_matrix(N, *, method='asymmetric_sqrt'):
    """
    Projection matrix required to turn an N-D space coordinate into 2D,
    assuming that every undetermined basis vector will be moved to the direction with symmetric original-basis-vector components (i.e. equal indices).
    This yields a matrix such that for 
    """
    next_row_calculator = {
        "asymmetric_linear": determine_next_vector_linear,
        "asymmetric_sqrt": determine_next_vector_sqrt,
        "symmetric": determine_next_vector_symmetric,
    }

    initial_direction_chooser = {
        "asymmetric_linear": direction_111,
        "asymmetric_sqrt": direction_111,
        "symmetric": direction_111,
    }

    projection = np.zeros([N, N])
    projection[-1] = initial_direction_chooser[method](N)
    projection[-1] /= np.linalg.norm(projection[-1]) # normalize

    for i in range(N-1, 0, -1):
        # fill in the i-1^th vector
        projection[i-1] = next_row_calculator[method](projection[i:])

    # tidy up the stray floating point errors, of small values that actually should've been zero.
    iszero = np.isclose(projection, 0, rtol=0, atol=np.finfo(projection.dtype).resolution)
    projection[iszero] = 0

    return projection[:2]
    # return projection[-2:]
    # return projection.T[:2]
    # return projection.T[-2:]

def projection_matrix_symmetric_fast(N):
    """
    Same as projection_matrix_symmetric, but faster and more accurate.
    Mathematically simplified as I noticed the pattern and extracted it, so it saved some of the the heavy lifting of doing arithmatic and cancellations.
    """
    matrix = np.zeros([N, N])
    matrix[-1] = np.full(N, sqrt(1/N))
    for i in range(0, N-1):
        matrix[i, :i+1] = (-1)**(N-i)/sqrt((i+1)*(i+2))
        matrix[i, i+1] = (-1)**(N-i)*-sqrt((i+1)/(i+2))
    matrix[-2, :-1] = -1/sqrt((N-1) * N)
    matrix[-2, -1] = sqrt((N-1) / N)
    return matrix

if __name__=='__main__':
    from matplotlib import pyplot as plt
    import sys
    N = int(sys.argv[1])
    np.set_printoptions(linewidth=212)
    print(f"Projecting {N} dimensions onto 2D.")
    projection = np.zeros([N, N])

    projection[-1] = np.full(N, sqrt(1/N))
    # projection[-1] = linear_indices(N)
    projection[-1] /= np.linalg.norm(projection[-1])
    print(projection)

    for i in range(N-1, 0, -1):
        print(f"Step={N-i}:")
        # fill in the i-1^th vector
        projection[i-1] = determine_next_vector_linear(projection[i:])
        print(projection)
        print(f"has rank {np.linalg.matrix_rank(projection)}")

    iszero = np.isclose(projection, 0, rtol=0, atol=np.finfo(projection.dtype).resolution)
    projection[iszero] = 0
    print(projection)
    print("The length of row vectors:\n", [np.linalg.norm(row) for row in projection])
    print("The length of column vectors:\n", [np.linalg.norm(col) for col in projection.T])