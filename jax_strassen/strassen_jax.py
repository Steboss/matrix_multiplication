# This implementation is calibrated for square matrices 
# whose size is a multiple of 4 
import jax.numpy as jnp
import numpy as np 
import time
from typing import List 

BlockMatrix = List[List[jnp.ndarray]]
SIZES = [8192, 10240, 12288, 14336, 16384, 18432, 20480]

def block_split(matrix: jnp.ndarray, n_rows: int, n_cols: int) -> BlockMatrix:
  """Splits `matrix` into a `n_rows x n_cols` block matrix."""
  rows = jnp.split(matrix, n_rows, axis=0)
  return [jnp.split(row, n_cols, axis=1) for row in rows]


def strassen_tensor():
    # List of 7 factors, each of shape [3, 4].
    factors = [[[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]],
                [[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]],
                [[0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]],
                [[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]],
                [[0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]],
                [[-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]],
                [[1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]]]

    # Transpose into our standard format [3, S, R] = [3, 4, 7],
    return np.transpose(np.array(factors, dtype=np.int32), [1, 2, 0])


def _product_factors(factors1: np.ndarray, factors2: np.ndarray) -> np.ndarray:
    """Computes the Kronecker product of `factors1` and `factors2`.

    Args:
        factors1: [3, n1**2, R1] factors of a tensor T1
        factors2: [3, n2**2, R2] factors of a tensor T2

    Returns:
        [3, n1**2 * n2 ** 2, R1 * R2] factorization of the Kronecker square tensor
        Reshape(kron(RT1, RT2)), where `RT1` and `RT2` are the reshapes of T1 and T2
        into 6-dimensional tensors, and `Reshape` reshapes the tensor back into a
        3-dimensional one.
    """
    _, side1, rank1 = np.shape(factors1)
    _, side2, rank2 = np.shape(factors2)

    n1 = int(np.round(np.sqrt(side1)))
    n2 = int(np.round(np.sqrt(side2)))

    if n1 * n1 != side1 or n2 * n2 != side2:
        raise ValueError(f'The sides {side1}, {side2} of factors passed to '
                        '`product_factors` must be both perfect squares.')
    product = np.einsum('...abi,...cdj->...acbdij',
                        factors1.reshape((3, n1, n1, rank1)),
                        factors2.reshape((3, n2, n2, rank2))
                        )  # [3, n1, n2, n1, n2, R1, R2]
    return np.reshape(product, (3, n1 * n2 * n1 * n2, rank1 * rank2))


def f(a: BlockMatrix, b: BlockMatrix) -> BlockMatrix:
    """Multiplies block matrices `a` and `b`."""
    n = len(a)
    result = [[None] * n for _ in range(n)]
    for alpha in range(rank):
      left = None
      for i in range(n):
        for j in range(n):
          if factors[0][i, j, alpha] != 0:
            curr = factors[0][i, j, alpha] * a[i][j]
            if left is None:
              left = curr
            else:
              left += curr
      right = None
      for j in range(n):
        for k in range(n):
          if factors[1][j, k, alpha] != 0:
            curr = factors[1][j, k, alpha] * b[j][k]
            if right is None:
              right = curr
            else:
              right += curr

      matrix_product = left @ right

      for i in range(n):
        for k in range(n):
          if factors[2][i, k, alpha] != 0:
            curr = factors[2][i, k, alpha] * matrix_product
            if result[i][k] is None:
              result[i][k] = curr
            else:
              result[i][k] += curr
    return result

# fix the random generator 
np.random.seed(42)
# definition of lists for plots 
results = {}
# and save on a file 
ofile = open("jax_results.csv", "w")
ofile.write("SIZE, BASE_RES_MEAN, BASE_RES_STD, OPT_RES_MEAN, OPT_RES_STD\n")
for SIZE in SIZES: 
    # generate INTEGERS [1-9] random matrices 
    A = np.random.randint(low=1, high=9, size=(SIZE, SIZE))
    B = np.random.randint(low=1, high=9, size=(SIZE,SIZE))
    # list to store results
    base_res = [] 
    strass_res = []
    # run at least 3 trials for each matrix
    for trial in range(0, 3):
        start = time.time()
        C = jnp.dot(A,B)
        end = time.time()
        total_time = end- start 
        base_res.append(total_time)
        print(f"Total time {total_time}")
        strassen = strassen_tensor() 
        # this is the Strassen's tensor to be applied on 4x4 matrices
        strassen_4_by_4 = _product_factors(strassen, strassen) # size (3, 16, 49) 
        # 3 tensors whose size is 16*49
        n = int(np.sqrt(strassen_4_by_4[0].shape[0]))
        # generate blocks 
        a = block_split(A, n, n)
        b = block_split(B, n, n)
        # here we can put the arrays on device 
        factors = [strassen_4_by_4[0].copy(), strassen_4_by_4[1].copy(), strassen_4_by_4[2].copy()]
        rank = factors[0].shape[-1]
        print(rank)
        factors[0] = strassen_4_by_4[0].reshape(n, n, rank)
        factors[1] = strassen_4_by_4[1].reshape(n, n, rank)
        factors[2] = strassen_4_by_4[2].reshape(n, n, rank)
        factors[2] = factors[2].transpose(1, 0, 2)
        start = time.time()
        c = b
        c = f(a, c)
        c[0][0].block_until_ready()
        end = time.time() 
        total_time = end - start
        strass_res.append(total_time)
        # convert back to original matrix 
        c_arr = np.hstack(np.concatenate(np.array(c), axis=1)).reshape(SIZE,SIZE)
        #print(c_arr)
        print(f"Totale time {total_time}")
        print(C[0])
        print(c_arr[0])
    # compute the average and the stddev 
    results[SIZE] = [np.mean(base_res), np.std(base_res),\
                     np.mean(strass_res), np.std(strass_res)]
    # for safety save in a file 
    ofile.write(f"{SIZE},{np.mean(base_res)}, {np.std(base_res)}, {np.mean(strass_res)}, {np.std(strass_res)}\n")
ofile.close()
# plot
# TODO Try to use device put