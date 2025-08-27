import numpy as np
import time

class DiagBlockMatrix:
    def __init__(self, n_blocks: int, block_size: int, data_1d: np.ndarray):
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
        return cls(n_blocks, block_size, np.array(data_1d, dtype=float))

    def to_2d_matrix(self):
        """Return a numpy array of shape (n*d, n*d) from the flattened diagonals"""
        n_total = self.n * self.d
        full_matrix = np.zeros((n_total, n_total), dtype=float)

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
        """Multiply two diagonal block matrices using block-wise vectorized dot product"""
        if self.n != other.n or self.d != other.d:
            raise ValueError("Both matrices must have the same dimensions and block size to multiply.")
        result_data = np.zeros(self.n * self.n * self.d, dtype=float)
        # Precompute all diagonals for A and B
        start = time.time()
        diags_a = np.ascontiguousarray([[self.get_block_diagonal(i, k) for k in range(self.n)] for i in range(self.n)])
        diags_b = np.ascontiguousarray([[other.get_block_diagonal(k, j) for j in range(self.n)] for k in range(self.n)])
        end = time.time()
        print(f"Time to get diag blocks: {end - start} seconds")
        # einsum: 'ikd,kjd->ijd' -> sum over k, multiply element-wise along d
        start = time.time()
        result_3d = np.einsum('ikd,kjd->ijd', diags_a, diags_b)
        end = time.time()
        print(f"Time to get calculate sums over all blocks: {end - start} seconds")
        # Flatten to 1D array in row-major block order
        result_data = result_3d.reshape(-1)
        return DiagBlockMatrix(self.n, self.d, result_data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def random_block_diag_matrix(n_blocks, block_size):
        matrix = np.zeros((n_blocks * block_size, n_blocks * block_size))
        for i in range(n_blocks):
            for j in range(n_blocks):
                block = np.random.rand(block_size, block_size)
                # Keep only diagonal
                block = np.diag(np.diagonal(block))
                matrix[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block
        return matrix

    n_blocks = 500
    block_size = 5
    n_total = n_blocks * block_size

    # 3. Create two random block-diagonal matrices
    A_np = random_block_diag_matrix(n_blocks, block_size)
    B_np = random_block_diag_matrix(n_blocks, block_size)

    # 2. Multiply using full naive numpy arrays
    start = time.time()
    C_np = A_np @ B_np
    end = time.time()
    print(f"Full NumPy multiplication took {end - start:.6f} seconds")

    # 3. Convert to DiagBlockMatrix and multiply
    A_block = DiagBlockMatrix.from_2d_matrix(A_np, block_size)
    B_block = DiagBlockMatrix.from_2d_matrix(B_np, block_size)

    start = time.time()
    C_block = A_block * B_block
    end = time.time()
    print(f"DiagBlockMatrix multiplication took {end - start:.6f} seconds")

    # 4. Compare results
    C_block_np = C_block.to_2d_matrix()
    error = np.max(np.abs(C_np - C_block_np))
    print(f"Max difference between NumPy and BlockDiagMatrix result: {error:e}")
