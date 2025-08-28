import pytest
import numpy as np
from DiagBlockMatrix import DiagBlockMatrix

def random_block_diag_matrix(n_blocks: int, block_size: int, identity: bool = False):
    matrix = np.zeros((n_blocks * block_size, n_blocks * block_size))
    for i in range(n_blocks):
        for j in range(n_blocks):
            block = np.identity(block_size) if identity else np.random.rand(block_size, block_size)
            # Keep only diagonal
            block = np.diag(np.diagonal(block))
            matrix[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block
    return matrix

@pytest.fixture
def matrix_fixture(request):
    """Unified fixture for all block-diagonal matrices, returning (matrix: np.ndarray, block_size: int, name: string)."""
    param = request.param
    if param == "mini_identity":
        return random_block_diag_matrix(1, 3, True), 3, param
    elif param == "mini_matrix":
        return random_block_diag_matrix(1, 3, False), 3, param
    elif param == "small_identity":
        return random_block_diag_matrix(3, 3, True), 3, param
    elif param == "small_matrix":
        return random_block_diag_matrix(3, 3, False), 3, param
    elif param == "medium_identity":
        return random_block_diag_matrix(5, 5, True), 5, param
    elif param == "medium_matrix":
        return random_block_diag_matrix(5, 5, False), 5, param
    elif param == "large_identity":
        return random_block_diag_matrix(30, 7, True), 7, param
    elif param == "large_matrix":
        return random_block_diag_matrix(30, 7, False), 7, param
    elif param == "huge_matrix":
        return random_block_diag_matrix(100, 10, False), 10, param
    elif param == "hugest_matrix":
        return random_block_diag_matrix(500, 30, False), 30, param
    else:
        raise ValueError(f"Unknown fixture name: {param}")