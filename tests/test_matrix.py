import pytest
import numpy as np
import time
from DiagBlockMatrix import DiagBlockMatrix


@pytest.mark.parametrize("matrix_fixture",
                         ["mini_identity", "mini_matrix", "small_identity", "small_matrix",
                          "medium_identity", "medium_matrix", "large_identity", "large_matrix", "huge_matrix"],
                         indirect=True)
def test_matrix_multiply(matrix_fixture, request):
    a_np, a_blocksize, fixture_name = matrix_fixture
    b_np, b_blocksize, _ = matrix_fixture
    print(f"Testing {fixture_name}")
    # Multiply using full naive NumPy arrays
    start = time.time()
    c_np = a_np @ b_np
    end = time.time()
    print(f"Full NumPy multiplication took {end - start:.6f} seconds")

    # Convert to DiagBlockMatrix
    a_block = DiagBlockMatrix.from_2d_matrix(a_np, a_blocksize)
    b_block = DiagBlockMatrix.from_2d_matrix(b_np, b_blocksize)

    # Multiply using DiagBlockMatrix
    start = time.time()
    c_block = a_block * b_block
    end = time.time()
    print(f"DiagBlockMatrix multiplication took {end - start:.6f} seconds")

    # Convert result back to NumPy
    c_block_np = c_block.to_2d_matrix()

    # Compare results
    error = np.max(np.abs(c_np - c_block_np))
    print(f"Max difference between NumPy and BlockDiagMatrix result: {error:e}")

    # Assert the error is negligible
    assert error < 1e-12
