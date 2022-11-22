import unittest
from tagger import *


def generate_windows(data: List[object]) -> List[Tuple[object, object]]:

    windows = []
    i1, i2 = 0, 1

    while i2 < len(data):
        windows.append((data[i1], data[i2]))
        i1 = i2
        i2 += 1

    return windows

def tagify(sentence: str) -> List[str]:

    words = sentence.split()
    tagged_words = []

    for word in words:
        tagged_words.append(f"{word.strip()} : TAG")

    return tagged_words


class TestCount(unittest.TestCase):
    def test_no_quotes_returns_correct_num_lines(self):
        # Arrange
        trainer = Trainer([])
        training_data = tagify("Yo Chief ! It is me . Sup ?")
        line_windows = generate_windows(training_data)
        is_start = True

        # Act
        for line1, line2 in line_windows:
            is_start = trainer._count(line1, line2, is_start)

        # Assert
        self.assertEqual(2, trainer._lines)

    def test_quote_at_end_returns_correct_num_lines(self):
        # Arrange
        trainer = Trainer([])
        training_data = tagify(
            ' Remember the golden rule of life . " Jaldi ka kaam shatan ka hota hai . "'
        )
        line_windows = generate_windows(training_data)
        is_start = True

        # Act
        for line1, line2 in line_windows:
            is_start = trainer._count(line1, line2, is_start)

        # Assert
        self.assertEqual(2, trainer._lines)


if __name__ == "__main__":
    unittest.main()
