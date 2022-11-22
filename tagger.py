import sys
from typing import *


FreqDict = Dict[str, 'Frequency']


class Frequency:

    def __init__(self, for_str: str) -> None:
        self.for_str: str = for_str
        self.count: int = 0
        self.frequencies: Dict[str, int] = {}

    def record(self, other_str: str) -> None:
        if other_str not in self.frequencies:
            self.frequencies[other_str] = 0
        self.count += 1
        self.frequencies[other_str] += 1

    def __repr__(self) -> str:
        return f"{self.for_str}({self.count}) -> {self.frequencies.__repr__()}"


class Trainer:

    def __init__(self, training_files: List[str]) -> None:
        self._files: List[str] = training_files
        self._initial_freq: Dict[str, int] = {}
        self._transition_freq: FreqDict = {}
        self._emission_freq: FreqDict = {}
        self._lines: int = 0

    def train(self) -> Tuple[int, Dict[str, int], FreqDict, FreqDict]:

        for file in self._files:
            with open(file) as f:

                is_start = True
                line1 = f.readline()
                line2 = f.readline()

                while True:
                    if not line2:
                        break

                    is_start = self._count(line1, line2, is_start)

                    line1 = line2
                    line2 = f.readline()

                last_word, _ = self._parse_line(line1)
                if last_word == '"':
                    self._lines += 1

        return (
            self._lines, self._initial_freq,
            self. _transition_freq, self._emission_freq
        )

    def _count(self, line1: str, line2: str, is_start: bool) -> bool:

        word1, tag1 = self._parse_line(line1)
        word2, tag2 = self._parse_line(line2)

        self._count_transition(tag1, tag2)
        self._count_emission(tag1, word1)

        if is_start and word1 != '"':
            self._count_initial(tag1)
            is_start = False

        # TODO: Account for complex sentence ends, such as "blah"? or "blah?"
        if self._is_sentence_end(word1, word2):
            is_start = True
            self._lines += 1

        return is_start

    def _parse_line(self, line: List[str]) -> Tuple[str, str]:
        line = line.split(":")
        word, tag = line[0].strip(), line[1].strip()
        return word, tag

    def _count_transition(self, tag1: str, tag2: str) -> None:
        if tag1 not in self._transition_freq:
            self._transition_freq[tag1] = Frequency(tag1)
        self._transition_freq[tag1].record(tag2)

    def _count_emission(self, tag: str, word: str) -> None:
        if tag not in self._emission_freq:
            self._emission_freq[tag] = Frequency(tag)
        self._emission_freq[tag].record(word)

    def _count_initial(self, tag: str) -> None:
        # TODO: Need to count the number of lines somewhere
        if tag not in self._initial_freq:
            self._initial_freq[tag] = 0
        self._initial_freq[tag] += 1

    def _is_sentence_end(self, word1: str, word2: str) -> bool:
        return word1 in [".", "!", "?"]


def tag(training_files: List[str], test_file: str, output_file: str) -> None:
    trainer = Trainer(training_files)
    num_lines, initial_freq, transition_freq, emissions_freq = trainer.train()
    print(emissions_freq)
    print(num_lines)

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
