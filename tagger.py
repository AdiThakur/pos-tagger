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

    def get_prob_of(self, word: str) -> float:
        if word not in self.frequencies:
            return 0
        return (self.frequencies[word] / self.count)

    def __repr__(self) -> str:
        return f"{self.for_str}({self.count}) -> {self.frequencies.__repr__()}"


class Parser:
    def parse_line(line: List[str]) -> Tuple[str, str]:
        line = line.split(":")
        word, tag = line[0].strip(), line[1].strip()
        return word, tag

    def is_sentence_end(word: str) -> bool:
        return word in [".", "!", "?"]


class Counter:

    def __init__(self, training_files: List[str]) -> None:
        self._files: List[str] = training_files
        self._initial_freq: Dict[str, int] = {}
        self._transition_freq: FreqDict = {}
        self._emission_freq: FreqDict = {}
        self._sentences: int = 0

    def count(self) -> Tuple[int, Dict[str, int], FreqDict, FreqDict]:

        for file in self._files:
            with open(file) as f:

                is_start = True
                line1 = f.readline()
                line2 = f.readline()

                while True:
                    if not line1:
                        break

                    is_start = self._count(line1, line2, is_start)

                    line1 = line2
                    next_line = f.readline()

                    if next_line == '':
                        line2 = None
                    else:
                        line2 = next_line

        return (
            self._sentences, self._initial_freq, self. _transition_freq,
            self._emission_freq
        )

    def _count(
        self, line1: str, line2: Optional[str], is_start: bool) -> bool:

        word1, tag1 = Parser.parse_line(line1)
        word2, tag2 = None, None

        if "-" in tag1:
            split_tags = tag1.split("-")
            tag1 = "-".join(sorted(split_tags))

        if line2:
            word2, tag2 = Parser.parse_line(line2)
            self._count_transition(tag1, tag2)

        self._count_emission(tag1, word1)

        if is_start and word1 != '"':
            self._count_initial(tag1)
            is_start = False

        if Parser.is_sentence_end(word1):
            is_start = True
            self._sentences += 1

        return is_start

    def _count_transition(self, tag1: str, tag2: str) -> None:
        if tag1 not in self._transition_freq:
            self._transition_freq[tag1] = Frequency(tag1)
        self._transition_freq[tag1].record(tag2)

    def _count_emission(self, tag: str, word: str) -> None:
        if tag not in self._emission_freq:
            self._emission_freq[tag] = Frequency(tag)
        self._emission_freq[tag].record(word)

    def _count_initial(self, tag: str) -> None:
        if tag not in self._initial_freq:
            self._initial_freq[tag] = 0
        self._initial_freq[tag] += 1


def viterbi(
    sentence: List[str],
    tags: List[str],
    num_sentences: int,
    initial_prob_matrix: List[float],
    transition_prob_matrix: List[List[float]],
    emission_prob_matrix: List[Dict[str, float]]) -> Tuple[List[List[int]], List[List[int]]]:

    num_tags = len(tags)
    prob: List[List[int]] = []
    prev: List[List[int]] = []

    for i in range(len(sentence)):
        prob.append([0] * num_tags)
        prev.append([-1] * num_tags)

    initial_prob_sum = 0

    for i in range(num_tags):
        emission_prob = emission_prob_matrix[i].get(sentence[0], 0)
        curr_prob = initial_prob_matrix[i] * emission_prob
        initial_prob_sum += curr_prob
        prob[0][i] = curr_prob

    for i in range(num_tags):
        if initial_prob_sum > 0:
            prob[0][i] = (prob[0][i] / initial_prob_sum)

    for t in range(1, len(sentence)):

        max_prob_sum = 0

        for i1 in range(num_tags):

            max_prob = 0
            max_prev_tag_index = i1
            emission_prob = emission_prob_matrix[i1].get(sentence[t], 0)

            for i2 in range(num_tags):

                prev_prob = prob[t - 1][i2]
                curr_prob = prev_prob * transition_prob_matrix[i2][i1] * emission_prob

                if curr_prob >= max_prob:
                    max_prob = curr_prob
                    max_prev_tag_index = i2

            prob[t][i1] = max_prob
            prev[t][i1] = max_prev_tag_index
            max_prob_sum += max_prob

        for i in range(num_tags):
            if max_prob_sum > 0:
                prob[t][i] = (prob[t][i] / max_prob_sum)

    return prob, prev


def follow_path(
    tags: List[str], prob: List[List[int]], prev: List[List[int]]) -> List[str]:

    # Determine highest prob last state
    max_prob_tag_index = 0
    max_prob = prob[-1][0]

    for i in range(len(prev[-1])):
        curr_prob = prob[-1][i]
        if curr_prob >= max_prob:
            max_prob = curr_prob
            max_prob_tag_index = i

    # Follow most-likely states back to first
    tag_index = max_prob_tag_index
    most_likely_tags = []

    for t in range(len(prob) - 1, -1, -1):
        most_likely_tags.insert(0, tags[tag_index])
        tag_index = prev[t][tag_index]

    return most_likely_tags


def tagify(sentence: List[str], tags: List[str]) -> List[str]:
    if len(sentence) != len(tags):
        raise ValueError("Params <sentence> and <tags> must have same len")
    return [f"{sentence[i]} : {tags[i]}" for i in range(len(sentence))]


def compute_probabilities(
    tags: List[str],
    num_sentences: int,
    initial_freq: Dict[str, int],
    transition_freq: FreqDict,
    emission_freq: FreqDict) -> Tuple[List[float], List[List[float]], List[Dict[str, float]]]:

    # Initial Probability
    initial_prob_matrix: List[float] = [0] * len(tags)

    for i, tag in enumerate(tags):
        if tag not in initial_freq:
            curr_prob = 0
        else:
            curr_prob = initial_freq[tag] / num_sentences
        initial_prob_matrix[i] = curr_prob

    # Transition Probability
    transition_prob_matrix: List[List[float]] = [0] * len(tags)

    for i1 in range(len(tags)):
        transition_prob_matrix[i1] = []
        for i2 in range(len(tags)):
            transition_prob_matrix[i1].append(transition_freq[tags[i1]].get_prob_of(tags[i2]))

    # Emission Probability
    emission_prob: List[Dict[str, float]] = [0] * len(tags)

    for i1 in range(len(tags)):
        emission_prob[i1] = {}
        for word in emission_freq[tags[i1]].frequencies:
            emission_prob[i1].update({word: emission_freq[tags[i1]].get_prob_of(word)})

    return initial_prob_matrix, transition_prob_matrix, emission_prob


def generate_probability_matrices(
    training_files: List[str]) -> Tuple[int, List[str], List[float], List[List[float]], List[Dict[str, float]]]:

    counter = Counter(training_files)
    num_sentences, initial_freq, transition_freq, emission_freq = counter.count()
    tags = list(emission_freq.keys())

    initial_prob_matrix, transition_prob_matrix, emission_prob_matrix = compute_probabilities(
        tags,
        num_sentences,
        initial_freq,
        transition_freq,
        emission_freq
    )

    return num_sentences, tags, initial_prob_matrix, transition_prob_matrix, emission_prob_matrix


def tag(
    test_filename: str,
    output_filename: str,
    tags: List[str],
    num_sentences: int,
    initial_prob_matrix: List[float],
    transition_prob_matrix: List[List[float]],
    emission_prob_matrix: List[Dict[str, float]]) -> None:

    input_file = open(test_filename)
    output_file = open(output_filename, "w")

    word = input_file.readline().strip()
    curr_sentence: List[str] = []

    while word:

        curr_sentence.append(word)
        next_word = input_file.readline().strip()

        # TODO: Handle cases where quotes make start/end of sentence hard to ID
        if Parser.is_sentence_end(word):

            # Call viterbi
            prob, prev = viterbi(
                curr_sentence,
                tags,
                num_sentences,
                initial_prob_matrix,
                transition_prob_matrix,
                emission_prob_matrix
            )

            # Follow path to get most likely tags
            most_likely_tags = follow_path(tags, prob, prev)

            # Write output
            tagged_words = tagify(curr_sentence, most_likely_tags)
            for tagged_word in tagged_words:
                output_file.write(tagged_word + "\n")

            # Rest current sentence
            curr_sentence = []

        word = next_word

    input_file.close()
    output_file.close()


def main(training_files: List[str], test_filename: str, output_filename: str) -> None:

    num_sentences, tags, initial_prob, transition_prob, emission_prob = generate_probability_matrices(training_files)

    tag(
        test_filename,
        output_filename,
        tags,
        num_sentences,
        initial_prob,
        transition_prob,
        emission_prob
    )


if __name__ == '__main__':
    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d") + 1 : parameters.index("-t")]
    test_file = parameters[parameters.index("-t") + 1]
    output_file = parameters[parameters.index("-o") + 1]
    main(training_list, test_file, output_file)


# Possible speedups?
# 1. Pre-compute all transition and emission probabilities; instead of computing
#    them every time, we can just do a simple lookup
# 2. In follow path, either use two stacks, or create a fixed size list and index
#    into it rather than constantly insert to the front
