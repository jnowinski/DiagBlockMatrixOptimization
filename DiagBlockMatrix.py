import numpy as np
from numba import njit, prange


class DiagBlockMatrix:
    def __init__(self, n_blocks: int, block_size: int, data_1d: np.ascontiguousarray):
        self.n = n_blocks
        self.d = block_size
        self.data = data_1d
        if len(self.data) != n_blocks * n_blocks * block_size:
            raise ValueError("Data length does not match. Expected n^2*d elements.")

    @classmethod
    def from_2d_matrix(cls, matrix: np.ndarray, block_size: int):
        """Return a new flattened DiagBlockMatrix from a 2d numpy array"""
        n_total = matrix.shape[0]
        if n_total % block_size != 0 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Expected square matrix divisible by block size")
        n_blocks = n_total // block_size
        data_1d = []
        for i in range(n_blocks):
            for j in range(n_blocks):
                block = matrix[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                data_1d.extend(np.diagonal(block))
        return cls(n_blocks, block_size, np.ascontiguousarray(data_1d, dtype=np.float64))

    def to_2d_matrix(self):
        """Return a numpy array of shape (n*d, n*d) from the flattened diagonals"""
        n_total = self.n * self.d
        full_matrix = np.zeros((n_total, n_total), dtype=np.float64)

        for i in range(self.n):
            for j in range(self.n):
                diag = self.get_block_diagonal(i, j)
                for k in range(self.d):
                    full_matrix[i * self.d + k, j * self.d + k] = diag[k]
        return full_matrix

    def index(self, i, j, k):
        """Return the kth diagonal of block (i,j)"""
        if i >= self.n or j >= self.n or k >= self.d:
            raise ValueError("Failed to index. Index out of bounds.")
        return ((i * self.n) + j) * self.d + k

    def get_block_diagonal(self, i, j):
        """Return the diagonal of block (i,j)"""
        if i >= self.n or j >= self.n:
            raise ValueError("Failed to return block diagonal. Index out of bounds.")
        start = ((i * self.n) + j) * self.d
        return self.data[start:start + self.d]

    def __add__(self, other):
        """Add two diagonal block matrices"""
        if self.n != other.n or self.d != other.d:
            raise ValueError("Both matrices must have the same dimensions and block size to add.")
        return DiagBlockMatrix(self.n, self.d, self.data + other.data)

    def __sub__(self, other):
        """Subtract two diagonal block matrices"""
        if self.n != other.n or self.d != other.d:
            raise ValueError("Both matrices must have the same dimensions and block size to add.")
        return DiagBlockMatrix(self.n, self.d, self.data - other.data)

    def __mul__(self, other):
        """Handles scalar and matrix multiplication"""
        if isinstance(other, DiagBlockMatrix):
            if self.n != other.n or self.d != other.d:
                raise ValueError("Both matrices must have the same dimensions and block size.")
            n, d = self.n, self.d
            result = diag_block_multiply(self.data, other.data, n, d)
            return DiagBlockMatrix(n, d, result)
        if np.isscalar(other):
            return DiagBlockMatrix(self.n, self.d, self.data * other)
        return NotImplemented

    def __rmul__(self, other):
        """Delegates Scalar × Matrix to __mul__"""
        return self.__mul__(other)

    def transpose(self):
        """
        Return the transpose matrix of the diagonal block matrix.
        Changes stride order so each (i,j) block is in the (j,i)th index.
        """
        blocks = self.data.reshape(self.n, self.n, self.d)
        transposed = blocks.transpose(1, 0, 2)  # Switch i and j indices
        return DiagBlockMatrix(self.n, self.d, transposed.ravel())



@njit(parallel=True, fastmath=True)
def diag_block_multiply(a_data, b_data, n, d):
    product = np.zeros(n * n * d, dtype=np.float64)

    for idx in prange(n * n): # Parallelize block indexing
        i = idx // n
        j = idx % n
        block = np.zeros(d, dtype=np.float64)
        for k in range(n):
            start_a = ((i * n) + k) * d
            start_b = ((k * n) + j) * d
            block += a_data[start_a:start_a+d] * b_data[start_b:start_b+d]
        start_result = ((i * n) + j) * d
        product[start_result:start_result+d] = block
    return product

