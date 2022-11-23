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
    num_sentences: int,
    initial_freq: Dict[str, int],
    transition_freq: FreqDict,
    emission_freq: FreqDict) -> List[Dict[str, str]]:

    tags = emission_freq.keys()
    prob: List[Dict[str, int]] = [{}] * len(sentence)
    prev: List[Dict[str, str]] = [{}] * len(sentence)

    for tag in tags:

        if tag not in initial_freq:
            curr_prob = 0
        else:
            init_prob = (initial_freq[tag] / num_sentences)
            emission_prob = emission_freq[tag].get_prob_of(sentence[0])
            curr_prob = init_prob * emission_prob

        prob[0].update({ tag: curr_prob })
        prev[0][tag] = None

    for t in range(1, len(sentence)):
        for tag in tags:

            max_prob = 0
            max_prev_tag = tag

            for prev_tag in tags:
                prev_prob = prob[t - 1][prev_tag]
                transition_prob = transition_freq[prev_tag].get_prob_of(tag)
                emission_prob = emission_freq[tag].get_prob_of(tag)
                curr_prob =  prev_prob * transition_prob * emission_prob

                if curr_prob >= max_prob:
                    max_prob = curr_prob
                    max_prev_tag = prev_tag

            prob[t][tag] = max_prob
            prev[t][tag] = max_prev_tag

    return prob, prev


def follow_path(
    prob: List[Dict[str, int]], prev: List[Dict[str, str]]) -> List[str]:

    # Determine highest prob last state
    max_prob_tag = list(prev[-1].keys())[0]
    max_prob = prob[-1][max_prob_tag]

    for tag in prev[-1]:
        curr_prob = prob[-1][tag]
        if curr_prob >= max_prob:
            max_prob = curr_prob
            max_prob_tag = tag

    # Follow most-likely states back to first
    tag = max_prob_tag
    most_likely_tags = [max_prob_tag]

    for t in range(len(prob) - 1, 0, -1):
        most_likely_tags.insert(0, prev[t][tag])
        tag = prev[t][tag]

    return most_likely_tags


def tagify(sentence: List[str], tags: List[str]) -> List[str]:
    if len(sentence) != len(tags):
        raise ValueError("Params <sentence> and <tags> must have same len")
    return [f"{sentence[i]} : {tags[i]}" for i in range(len(sentence))]


def tag(
    test_filename: str,
    output_filename: str,
    num_sentences: int,
    initial_freq: Dict[str, int],
    transition_freq: FreqDict,
    emission_freq: FreqDict) -> None:

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
                num_sentences,
                initial_freq,
                transition_freq,
                emission_freq
            )

            # Follow path to get most likely tags
            most_likely_tags = follow_path(prob, prev)

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
    # Count
    counter = Counter(training_files)
    num_sentences, initial_freq, transition_freq, emission_freq = counter.count()

    tag(
        test_filename,
        output_filename,
        num_sentences,
        initial_freq,
        transition_freq,
        emission_freq
    )

    # TODO: For debugging; remove this shit boi
    # print(num_sentences)
    # print("Initial")
    # print(initial_freq)
    # print("Transition")
    # for key in transition_freq:
    #     print("\t" + transition_freq[key].__str__())
    # print("Emission")
    # for key in emission_freq:
    #     print("\t" + emission_freq[key].__str__())


if __name__ == '__main__':
    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d") + 1 : parameters.index("-t")]
    test_file = parameters[parameters.index("-t") + 1]
    output_file = parameters[parameters.index("-o") + 1]
    main(training_list, test_file, output_file)
    # main(
    #     ['data/training0.txt'],
    #     'data/test1.txt',
    #     'out.txt'
    # )

# Possible speedups?
# 1. Pre-compute all transition and emission probabilities; instead of computing
#    them every time, we can just do a simple lookup
# 2. In follow path, either use two stacks, or create a fixed size list and index
#    into it rather than constantly insert to the front
