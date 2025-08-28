## Problem Statement and Requirements
Consider a matrix A of size Rndxnd, composed of n^2 non-overlapping blocks of d x d matrices. Where each dxd block, Dij, is a diagonal matrix.
Design and implement a spatially and algorithmically efficient data structure to represent such matrices and provide efficient algorithms for common matrix operations, such as matrix multiplications, addition/subtraction, and transpose.

## Approach: 1D contiguous array storing blocks in row major order
Flatten all diagonal elements into a 1D array in row major ordering, traversing block rows first and then block columns, adding the diagonals from each block into a 1D array. This approach offers optimal memory locality and sequential access, since all diagonal elements are stored fully contiguously in a single array, making it ideal for matrix multiplications with SIMD and GPU computation. With that said, random access requires some computational overhead as the kth diagonal of the block at (i,j) is indexed with:
`A[x] = Dij(k)` Where x = ((i * n) + j) * d + k
