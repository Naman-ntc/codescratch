import itertools
import json
import multiprocessing
import numpy as np
from typing import Dict
from datasets import load_dataset
from .testing_util import run_test
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

DATASET = "codeparrot/apps"
TIMEOUT = 10


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=_temp_run, args=(sample, generation, debug, result)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0]


def evaluate_generations_by_problem(args):
    problem_generations: list[str] = args[0]
    sample = args[1]
    level: str = args[2]
    debug: bool = args[3]
    verbose: bool = debug

    res = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)
            if debug:
                print(f"\nSuccessful compilation of task {o_idx}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    print(f"Results were not True for all test cases")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            # break
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)
    if verbose:
        for i, r in enumerate(problem_generations):
            print("Sample\n")
            print(r)
            print("\n")
            print("Result\n")
            print(res[i])
            print("*" * 30 + "\n\n")
    return res


def evaluate_generations(
    apps_eval,
    generations_list: list[list[str]],
    level: str = "all",
    debug: bool = False,
):
    """We take the list of code generations and try to compile them
     and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS dataset)
        level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case
    """

    # generations are code generations in the same order of the dataset
    # apps_eval = load_dataset(DATASET, split="test", difficulties=[level])

    inputs = [
        [(generations_list[index], apps_eval[index], level, debug), index]
        for index in range(len(generations_list))
    ]

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(max_workers=1 if debug else None) as executor:
            futures = {
                executor.submit(evaluate_generations_by_problem, arg): index
                for arg, index in inputs
            }

            results = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                pbar.update(1)

    assert len(results) == len(
        inputs
    ), f"results = {len(results)} inputs = {len(inputs)} {results=}"
    # results = {i: r for r, (_, i) in zip(results, inputs)}

    return results


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def get_results(
    results: Dict[int, list], count_errors: bool = False, k_list: list = [1, 10, 100]
):
    """
    Given the results evaluated against the testcases we output some statistics.
    For single generations:
    >>> example_results = {0: [[-2]], 1: [[False,False]], 2: [[True,True]], 3: [[False,True,False,True]], 4: [[-1,-1]]}
    >>> get_results(example_results, count_errors=True)
    Computing accuracy metrics...
    number of compile errors = 1 avg = 0.2
    number of runtime errors = 1 avg = 0.2
    number of problems evaluated = 5
    Average Accuracy : 0.3
    Strict Accuracy : 0.2
    {'avg_accuracy': 0.3, 'strict_accuracy': 0.2, 'pass_at_k': None}

    For multiple generations:
    >>> example_results = {0: [[-2], [True, True, True]], 1: [[-1,-1, -1], [True, False, True]]}
    >>> get_results(example_results, k_list=[1, 2])
    Computing pass@k metric for multiple generations...
    {'pass@1': 0.25, 'pass@2': 0.5}
    {'avg_accuracy': None, 'strict_accuracy': None, 'pass_at_k': {'pass@1': 0.25, 'pass@2': 0.5}}
    """

    metrics = {
        "avg_accuracy": None,
        "strict_accuracy": None,
        "pass_at_k": None,
        "compile_errors": None,
        "runtime_errors": None,
        "wrong_answer": None,
    }

    if len(results[0]) == 1:
        # for single generations we compute average accuracy and stric accuracy: original APPS metrics
        print("Computing accuracy metrics...")
        res = []
        per_prob_res = []
        all_correct = []
        for index in sorted(results):
            problem_results = np.asarray(results[index])
            res.extend(problem_results)
            per_prob_res.append(np.mean(problem_results > 0))
            all_correct.append(np.all(problem_results > 0))
        # we count campilation and runtime errors once per pronlem
        compile_errors = len([e for e in res if -2 in e])
        runtime_errors = len([e for e in res if -1 in e])
        total_testcases = len(res)
        if count_errors:
            print(
                f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases}"
            )
            print(
                f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}"
            )
            print(f"number of problems evaluated = {total_testcases}")

        print(all_correct)
        print(f"Average Accuracy : {np.mean(per_prob_res)}")
        print(f"Strict Accuracy : {np.mean(all_correct)}")
        metrics["avg_accuracy"] = np.mean(per_prob_res)
        metrics["strict_accuracy"] = np.mean(all_correct)

    else:
        # for multiple generations we use pass@k metric used in the HumanEval benchmark
        # we use strict accuracy, a generation is valid if it has to pass all the tests
        print("Computing pass@k metric for multiple generations...")
        # total is list with nb generations per task (task=index)
        # correct is number of generations that passed all tests per task
        total = []
        correct = []
        compile_error_counts = []
        runtime_error_counts = []
        wrong_answer_counts = []
        assert all([isinstance(x, int) for x in results.keys()])
        for index in sorted(results):
            all_correct = []
            compile_error_counts_per_task = []
            runtime_error_counts_per_task = []
            wrong_answer_counts_per_task = []
            for generation in results[index]:
                gen = np.array(generation)
                all_correct.append(np.all(gen > 0))
                compile_error_counts_per_task.append(False)
                runtime_error_counts_per_task.append(False)
                wrong_answer_counts_per_task.append(False)

                contains_compile_error = (gen == -2).any()
                contains_runtime_error = (gen == -1).any()
                contains_wrong_answer = (gen == False).any()
                if contains_compile_error:
                    compile_error_counts_per_task[-1] = True
                elif contains_runtime_error:
                    runtime_error_counts_per_task[-1] = True
                elif contains_wrong_answer:
                    wrong_answer_counts_per_task[-1] = True
                # compile_error_counts_per_task.append((gen == -2).any())
                # runtime_error_counts_per_task.append((gen == -1).any())
                # wrong_answer_counts_per_task.append((gen == False).any())
            compile_error_counts.append(compile_error_counts_per_task)
            runtime_error_counts.append(runtime_error_counts_per_task)
            wrong_answer_counts.append(wrong_answer_counts_per_task)
            total.append(len(all_correct))
            correct.append(sum(all_correct))
        total = np.array(total)
        correct = np.array(correct)
        ks = k_list
        print(total.tolist())
        print(correct.tolist())
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
            for k in ks
            if (total >= k).all()
        }
        print(pass_at_k)
        metrics["pass_at_k"] = pass_at_k

        compile_error_counts = np.array(compile_error_counts)
        runtime_error_counts = np.array(runtime_error_counts)
        wrong_answer_counts = np.array(wrong_answer_counts)

        metrics["compile_errors_mean"] = compile_error_counts.mean(axis=1).mean()
        metrics["runtime_errors_mean"] = runtime_error_counts.mean(axis=1).mean()
        metrics["wrong_answer_mean"] = wrong_answer_counts.mean(axis=1).mean()

        metrics["correct_counts"] = correct.tolist()
        metrics["compile_errors_counts"] = compile_error_counts.sum(axis=1).tolist()
        metrics["runtime_errors_counts"] = runtime_error_counts.sum(axis=1).tolist()
        metrics["wrong_answer_counts"] = wrong_answer_counts.sum(axis=1).tolist()

    return metrics


def compute_metrics(
    dataset,
    generations,
    level="all",
    k_list=[1, 10, 100],
    count_errors=True,
    debug=False,
):
    """Return metrics for the given generations.
    Args:
        generations: list of code generations for each problem (each generation is a list of generations)
        k_list: list of k values to compute pass@k when using multiple generations
        count_errors: whether to count compilation and runtime errors when using single generations
        level: difficulty level in APPS dataset that was used for the given generations (from: "all", "introductory", "interview", "competition")
    Returns:
        metrics: dict of metrics

    Examples:

    >>> import json
    >>> # lists of solutions to the two first APPS problems (note not all solutions pass all tests)
    >>> solution_sample1 = json.load(open("test_examples/solutions_problem_1.json", "r"))
    >>> solution_sample2 = json.load(open("test_examples/solutions_problem_2.json", "r"))
    >>> single_solutions = [solution_sample1[:1], solution_sample2[:1]]
    >>> compute_metrics(single_solutions, level="all")
    Computing accuracy metrics...
    number of compile errors = 0 avg = 0.0
    number of runtime errors = 0 avg = 0.0
    number of problems evaluated = 2
    Average Accuracy : 1.0
    Strict Accuracy : 1.0
    {'avg_accuracy': 1.0, 'strict_accuracy': 1.0, 'pass_at_k': None}
    >>> multiple_solutions = [solution_sample1[:3], solution_sample2[:3]]
    >>> compute_metrics(multiple_solutions, level="all", k_list=[1, 2, 3])
    Computing pass@k metric for multiple generations...
    {'pass@1': 1.0, 'pass@2': 1.0, 'pass@3': 1.0}
    {'avg_accuracy': None, 'strict_accuracy': None, 'pass_at_k': {'pass@1': 1.0, 'pass@2': 1.0, 'pass@3': 1.0}}
    """
    results = evaluate_generations(dataset, generations, level=level, debug=debug)
    metrics = get_results(results, count_errors=count_errors, k_list=k_list)
    return metrics, results


# import doctest
# doctest.testmod()
