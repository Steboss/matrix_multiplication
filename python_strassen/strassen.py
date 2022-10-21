# This script is calibrated for square matrices of integers
import numpy as np
import time


SIZES = [2172, 2432, 2560, 2688, 3200, 4016  ]
THRESHOLD = 8


def strassen(A, B):
    r""" Strassen is applied recursively. 
    As a constraint we are stopping the iteration when 
    the submatrix has riched a size of THRESHOLD

    Parameters
    ----------
    A: np.array: block matrix A
    B: np.array: block matrix B

    Return 
    ------
    C: np.array: product matrix C
    """
    current_size = len(A) # we are taking just matrix A, as B has the same size
    if current_size%2!=0:
        C = np.matmul(A, B) 
        return C 
    else:
        current_new_size = current_size//2 
        # now extract all the blocks from the input matrix of size current_new_size 
        a11 = A[0:current_new_size,0:current_new_size] # top left
        a12 = A[0:current_new_size,current_new_size:current_size] # top right
        a21 = A[current_new_size:current_size, 0:current_new_size] # bottom left
        a22 = A[current_new_size:current_size, current_new_size:current_size] # bottom right
        # same fo rB 
        b11 = B[0:current_new_size,0:current_new_size] # top left
        b12 = B[0:current_new_size,current_new_size:current_size] # top right
        b21 = B[current_new_size:current_size, 0:current_new_size] # bottom left
        b22 = B[current_new_size:current_size, current_new_size:current_size] # bottom right
        # roll over Strassen
        a_ = np.add(a11, a22)
        b_ = np.add(b11, b22) 
        prod1 = strassen(a_, b_) # iterate over the first multiplication

        a_ = np.add(a21, a22) 
        prod2 = strassen(a_, b11) # second product 

        b_ = np.subtract(b12, b22) 
        prod3 = strassen(a11, b_) # third product 

        b_ = np.subtract(b21, b11)
        prod4 = strassen(a22, b_) # fourth product 

        a_ = np.add(a11, a12) 
        prod5 = strassen(a_, b22) # fifth product 

        a_ = np.subtract(a21, a11) 
        b_ = np.add(b11, b12) 
        prod6 = strassen(a_, b_) # sixth product 

        a_ = np.subtract(a12, a22) 
        b_ = np.add(b21, b22) 
        prod7 = strassen(a_, b_) # seventh product

        # compute the c element for the product matrix 
        c12 = np.add(prod3, prod5) 
        c21 = np.add(prod2, prod4) 
        
        a_ = np.add(prod1, prod4)
        b_ = np.add(a_, prod7)
        c11 = np.subtract(b_, prod5)

        a_ = np.add(prod1, prod3) 
        b_ = np.add(a_, prod6) 
        c22 = np.subtract(b_, prod2)

        # return the final matrix 
        C = np.zeros([current_size, current_size])
        C[0:current_new_size, 0:current_new_size] = c11 # top left
        C[0:current_new_size,current_new_size:current_size] = c12 # top right
        C[current_new_size:current_size, 0:current_new_size] = c21 # bottom left
        C[current_new_size:current_size, current_new_size:current_size] = c22 # bottom right

        return C

# fix the random generator 
np.random.seed(42)
# definition of lists for plots 
results = {}
# and save on a file 
ofile = open("python_results.csv", "w")
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
        # numpy multiplication 
        start = time.time()
        C = np.matmul(A, B)
        end = time.time()
        total_time = end - start
        print(f"total time {total_time}s")
        #print(C[0])
        base_res.append(total_time)
        # strassen 
        start = time.time() 
        C = strassen(A, B)
        end = time.time() 
        total_time = end - start
        print(f"total time {total_time}")
        strass_res.append(total_time)
        #print(C[0])
    # compute the average and the stddev 
    results[SIZE] = [np.mean(base_res), np.std(base_res),\
                     np.mean(strass_res), np.std(strass_res)]
    # for safety save in a file 
    ofile.write(f"{SIZE},{np.mean(base_res)}, {np.std(base_res)}, {np.mean(strass_res)}, {np.std(strass_res)}\n")
ofile.close()
# plot