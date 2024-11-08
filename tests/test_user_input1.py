import numpy as np
import unittest
import pytest

class TestInputter(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)
        np.testing.assert_allclose([0,1],[0,1])
        return