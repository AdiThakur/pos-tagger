import sys
from typing import *


FreqMatrix = Dict[str, 'Frequency']


class Frequency:
    for_str: str
    count: int
    frequencies: Dict[str, int]

    def __init__(self, for_str: str) -> None:
        self.for_str = for_str
        self.count = 0
        self.frequencies = {}

    def record(self, other_str: str) -> None:
        if other_str not in self.frequencies:
            self.frequencies[other_str] = 0
        self.count += 1
        self.frequencies[other_str] += 1

    def __repr__(self) -> str:
        return f"{self.for_str}({self.count}) -> {self.frequencies.__repr__()}"


def parse_line(line: List[str]) -> Tuple[str, str]:
    line = line.split(":")
    word, tag = line[0].strip(), line[1].strip()
    return word, tag


def count_frequency(matrix: FreqMatrix, str1: str, str2: str) -> None:
    if str1 not in matrix:
        matrix[str1] = Frequency(str1)
    matrix[str1].record(str2)


def train_from_file(
    training_file: str,
    trans_freq_matrix: FreqMatrix,
    emiss_freq_matrix: FreqMatrix) -> None:

    with open(training_file) as file:

        is_start = True
        line1 = file.readline()
        line2 = file.readline()

        while True:
            if not (line1 and line2):
                break

            word1, tag1 = parse_line(line1)
            word2, tag2 = parse_line(line2)

            count_frequency(trans_freq_matrix, tag1, tag2)
            count_frequency(emiss_freq_matrix, tag1, word1)

            line1 = line2
            line2 = file.readline()


def train_from_files(training_files: List[str]) -> None:

    trans_freq_matrix: FreqMatrix = {}
    emiss_freq_matrix: FreqMatrix = {}

    for training_file in training_files:
        train_from_file(
            training_file,
            trans_freq_matrix,
            emiss_freq_matrix
        )

    for tag in trans_freq_matrix:
        print(trans_freq_matrix[tag])
    for word in emiss_freq_matrix:
        print(emiss_freq_matrix[word])


def tag(training_files: List[str], test_file: str, output_file: str) -> None:
    train_from_files(training_files)


if __name__ == '__main__':
    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d") + 1 : parameters.index("-t")]
    test_file = parameters[parameters.index("-t") + 1]
    output_file = parameters[parameters.index("-o") + 1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
