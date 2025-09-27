import numpy as np
import pandas as pd

# numpy

# ufuncs are universal functions which operate on the whole of ndarrays from a single input (array, type broadcasting)
x = np.array([1,4,9])
np.log1p(x) # elementwise ops

# broadcasting with None
a = np.array([1,2,3]) # shape 3,
b = np.array([10,20]) # shape 2,

a[:, None] + b # places ndarray b into each entry of a shape 3,2

x = np.array([-2, -1, 0, 1, 2])
np.where(x>0, x, 0) # where x > 0, keep x otherwise insert 0 -> [0, 0, 0, 1, 2]

# boolean masking
x = np.arange(1, 10, 3) # 3 is step
y = np.linspace(1, 10, 3) # 3 is number of equally spaced intervals

# broadcasting and matrix operations
a = np.array([1,2,3]) # (3,) -> implictly (1,3) for matrix ops
# for matrix mult, numpy temporarily treats (3,) as (3,1)
# TxN (') 1xN for column wise
# TxN (') Tx1 for row wise

# Use wherever you're unsure
a[:, None] 

A = np.random.randn(10,100)
A_mean = A.mean(axis=0, keepdims=True) # w/o keepdims, we get 1-D ndarray (N,), keepdims returns matrix (1, N)
                                       # python still treats (N,) as (1,N)

a = np.array([1,2,3]) # shape (3,)
a_row = a[None,:] # places size 1 axis on None, other content on : (matrix shape)

# aligning higher rank matrices
X = np.random.rand(4,3)
w = np.array([1,2,3]) # (3,) -> (1,3) implicitly 
X * w # multiplies each column by 1, 2 and 3 resp.

# so use numpy's streching (broadcasting) to take advatange of space



# ---- pandas 




# Vectorization patterns vs loops

