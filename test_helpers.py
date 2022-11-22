import unittest
from tagger import *


class TestCountFrequency(unittest.TestCase):
    def test_basic(self):
        transitions = [
            ("ONE", "ONE"),
            ("ONE", "ONE"),
            ("ONE", "TWO"),
            ("ONE", "ONE"),
            ("TWO", "ONE"),
            ("TWO", "THREE"),
            ("TWO", "FOUR-FIVE"),
            ("FOUR-FIVE", "ONE"),
            ("FOUR-FIVE", "ONE"),
        ]

        trans_freq_matrix: FreqMatrix = {}
        for tag1, tag2 in transitions:
            count_frequency(trans_freq_matrix, tag1, tag2)

        self.assertEqual(3, trans_freq_matrix["ONE"].frequencies["ONE"])
        self.assertEqual(1, trans_freq_matrix["ONE"].frequencies["TWO"])

        self.assertEqual(1, trans_freq_matrix["TWO"].frequencies["ONE"])
        self.assertEqual(1, trans_freq_matrix["TWO"].frequencies["THREE"])
        self.assertEqual(1, trans_freq_matrix["TWO"].frequencies["FOUR-FIVE"])

        self.assertEqual(2, trans_freq_matrix["FOUR-FIVE"].frequencies["ONE"])


if __name__ == "__main__":
    unittest.main()
