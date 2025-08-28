import pytest
import numpy as np
import time
from DiagBlockMatrix import DiagBlockMatrix


@pytest.mark.parametrize("matrix_fixture",
                         ["mini_identity", "mini_matrix", "small_identity", "small_matrix",
                          "medium_identity", "medium_matrix", "large_identity", "large_matrix",
                          "huge_matrix", "hugest_matrix"],
                         indirect=True)
def test_matrix_multiply(matrix_fixture, request):
    a_np, a_blocksize, fixture_name = matrix_fixture
    b_np, b_blocksize, _ = matrix_fixture
    print(f"Testing {fixture_name}: n = {a_np.shape[0] / a_blocksize}, d = {a_blocksize}")

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

    assert error < 1e-11


@pytest.mark.parametrize("matrix_fixture",
                         ["mini_identity", "mini_matrix", "small_identity", "small_matrix",
                          "medium_identity", "medium_matrix", "large_identity", "large_matrix", "huge_matrix", "hugest_matrix"],
                         indirect=True)
def test_matrix_transpose(matrix_fixture):
    a_np, block_size, fixture_name = matrix_fixture
    print(f"Testing {fixture_name}: n = {a_np.shape[0] / block_size}, d = {block_size}")

    # Transpose with naive numpy array
    start = time.time()
    a_trans_np = a_np.T
    end = time.time()
    print(f"Numpy transpose took {end - start:.6f} seconds")

    # Transpose using DiagBlockMatrix
    a_block = DiagBlockMatrix.from_2d_matrix(a_np, block_size)
    start = time.time()
    a_trans_block = a_block.transpose()
    end = time.time()
    print(f"DiagBlockMatrix transpose took {end - start:.6f} seconds")

    # Check error
    a_trans_np_block = a_trans_block.to_2d_matrix()
    error = np.max(np.abs(a_trans_np - a_trans_np_block))
    print(f"Max difference in transpose: {error:e}")
    assert error < 1e-11
