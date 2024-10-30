#!/usr/bin/env python3

import unittest
import numpy as np

from TenSQR import ACGT_count


def preexisting_ACGT_count(M_E):
    """
    This is the ACGT_count implementation that existed previously in the TenSQR repo. Including it
    here so that I can compare it's output to the new version that handles more cases.
    """
    out = np.zeros((len(M_E[0, :]), 4))
    for i in range(4):
        out[:, i] = (M_E == (i + 1)).sum(axis=0)
    return out


class TestACGTCount(unittest.TestCase):

    def test_returns_nsnv_by_4_array(self):
        """It should return a n. SNV x 4 array."""
        n_reads = 9
        n_snvs = 5
        arr = np.random.choice([1, 2, 3, 4], n_snvs * n_reads).reshape(n_reads, n_snvs)
        out = ACGT_count(arr)
        self.assertEqual(out.shape, (n_snvs, 4))

    def test_matches_previous_implementation(self):
        """Test the new version matches the previous implementation."""
        n_reads = 9
        n_snvs = 5
        arr = np.random.choice([1, 2, 3, 4], n_snvs * n_reads).reshape(n_reads, n_snvs)
        out = ACGT_count(arr)
        out_old = preexisting_ACGT_count(arr)
        self.assertTrue((out == out_old).all())

    def test_single_read_case(self):
        """Should return a [5, 4] array."""
        n_reads = 1
        n_snvs = 5
        arr = np.random.choice([1, 2, 3, 4], n_snvs * n_reads).reshape(n_reads, n_snvs)
        out = ACGT_count(arr)
        self.assertEqual(out.shape, (5, 4))

    def test_single_snv_case_as_2d(self):
        """Should return a [1, 4] array. Here ACGT is passed as a proper 2D array."""
        n_reads = 9
        n_snvs = 1
        arr = np.random.choice([1, 2, 3, 4], n_snvs * n_reads).reshape(n_reads, n_snvs)
        out = ACGT_count(arr)
        self.assertEqual(out.shape, (1, 4))

    def test_passing_1d_array(self):
        """Passing a 1D array should throw a ValueError."""
        n_reads = 9
        n_snvs = 1
        arr = np.random.choice([1, 2, 3, 4], n_snvs * n_reads)
        with self.assertRaises(ValueError):
            ACGT_count(arr)


if __name__ == "__main__":
    unittest.main()
