import os
import time
from typing import *


def determine_accuracy(run_id: int, training_file_id: int, test_file_id: int) -> None:

    training_filename = f"data/training{training_file_id}.txt"
    test_filename = f"data/test{test_file_id}.txt"
    solution_filename = f"data/training{test_file_id}.txt"
    tagger_output_filename = f"output_{run_id}.txt"
    results_filename = f"results_{run_id}.txt"

    print(f"Training File: {training_filename} Test File: {test_filename}\n")

    start_time = time.time()
    os.system(f"python tagger.py -d {training_filename} -t {test_filename} -o {tagger_output_filename}")
    delta = time.time() - start_time

    with open(tagger_output_filename, "r") as output_file, \
            open(solution_filename, "r") as solution_file, \
            open(results_filename, "w") as results_file:

        output = output_file.readlines()
        solution = solution_file.readlines()
        total_matches = 0

        for index in range(len(output)):
            if output[index] != solution[index]:
                results_file.write(f"Line {index + 1}: "
                                   f"expected <{solution[index].strip()}> "
                                   f"but got <{output[index].strip()}>\n")
            else:
                total_matches = total_matches + 1

        # Add stats at the end of the results file.
        results_file.write(f"Time: {delta} seconds.\n")
        results_file.write(f"Total words seen: {len(output)}.\n")
        results_file.write(f"Total matches: {total_matches}.\n")
        results_file.write(f"\tAccuracy: {(total_matches / len(output)) * 100}.\n")

        print(f"\tTime: {delta} seconds.\n")
        print(f"\tTotal words seen: {len(output)}.\n")
        print(f"\tTotal matches: {total_matches}.\n")
        print(f"\tAccuracy: {(total_matches / len(output)) * 100}.\n")


if __name__ == '__main__':
    run_id = 1
    for training_file_id in range(1, 6):
        for test_file_id in range(1, 6):
            determine_accuracy(run_id, training_file_id, test_file_id)
            run_id += 1
