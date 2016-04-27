
import unittest
from unittest import TestCase
from feature import hog
import cv2
import numpy as np

class TestHog(TestCase):
    def test_hog(self):
        self.assertTrue(hog, np.zeros((50, 50), np.uint8))

if __name__ == '__main__':
    unittest.main()
