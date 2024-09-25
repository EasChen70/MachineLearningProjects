import numpy as np

def q1(n):
    # n*10 for elements above diagonal from top left to bottom right
    # -n*10 for elements below diagonal
    output = np.full((n,n), n*10, dtype='int')
    #decrement the diagonal by 10
    output[np.diag_indices(n)] = np.linspace(n*10, 10, n) # (start, end, dim)
    #upper matrix triangle excluding diagonal
    output[np.triu_indices(n, 1)] = n*10
    #lower matrix triangle excluding diagonal
    output[np.tril_indices(n, -1)] = -n*10
    return output

def q2(n):
    #set first row to 0 - (n - 1)
    row = np.arange(n, dtype='int')
    #None adds a dimension, then broadcasts
    output = row + np.arange(n, dtype='int')[:, None]
    return output

def q3(n):
    #Create the base matrices
    matrices = np.zeros((n,n,n), dtype='int')
    #[0,1,2..(n-1)] and each is multiplied by 10
    constant = 10 * np.arange(n)
    #broadcast
    output = matrices + constant[:, None, None]
    return output

def q4(array):
    #Credit to Claude for explaining what normalization is
    squared = np.square(array)
    sum = np.sum(squared, axis=-1, keepdims= True)
    magnitude = np.sqrt(sum)
    #Normalize
    answer = array/magnitude
    return answer
    
def q5(array):
    array = np.array(array)
    #If a element modulo 7 or 11 is 0 it means it's a multiple
    mask_7 = (array % 7 == 0)
    mask_11 = (array % 11 == 0)
    #Use masks to filter
    filter = mask_7 | mask_11
    output = array[filter]
    output = np.sum(output)
    return output

def q6(n):
    #Create column of 1 to n
    col = np.arange(1, n+1).reshape(n, 1)
    #Creates a row of 1 to n
    row = np.arange(1, n+1)
    #Broadcasting
    #By subtracting 1 from row element-wise, and using it as an exponent (credit to claude for showing me to use exponents instead of multiplication)
    output = col ** (row - 1)
    return output

def q7(array, k):
    input_shape = np.shape(array)
    #Slices list up to last two dimensions, and last two get modified for padding
    new_shape = list(input_shape[:-2]) + [input_shape[-2] + 2*k, input_shape[-1] + 2*k]
    padded_arr = np.zeros(new_shape, dtype='float')
    # Find the center or innermost(credit to chatgpt)
    center = tuple([slice(None)] * (len(input_shape) - 2) + [slice(k, -k), slice(k, -k)])
    # Place the original array in the center of the padded array
    padded_arr[center] = array
    return padded_arr

