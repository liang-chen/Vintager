
from unittest import TestCase
from feature import hog
import numpy as np


class TestHog(TestCase):
    def test_hog(self):
        self.assertTrue(hog, np.zeros((50, 50), np.uint8))
