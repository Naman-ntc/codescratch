import os
import sys
import glob
import tqdm
import json
import random
from math import ceil

import torch
import numpy as np
from datasets import load_dataset
from collections import defaultdict
from transformers import AutoTokenizer

from dataloaders.enumerated_file_utils import (
    DATA_KEYS,
    PLAN_PAD_DATA_KEYS,
    PLAN_MASK_DATA_KEYS,
    TRANSLATION_KEYS,
    get_question_path,
    get_passed_path,
    translate_solution_path,
)

try:
    sys.set_int_max_str_digits(0)
except:
    print("sys.set_int_max_str_digits(0) not supported")

# borrowed and modified from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/finetuning/APPS/apps_dataset.py


def read_solution_with_smallest_plan(solution_path):
    return RefactoredDataset.read_file(solution_path)
    if "attempt_" not in solution_path:
        return RefactoredDataset.read_file(solution_path)
    paths = [
        os.path.join(os.path.dirname(solution_path), "attempt_0.py"),
        os.path.join(os.path.dirname(solution_path), "attempt_1.py"),
        os.path.join(os.path.dirname(solution_path), "attempt_2.py"),
    ]

    plans = [RefactoredDataset.read_file(path) for path in paths]
    smallest_plan = min(plans, key=lambda x: len(x))
    return smallest_plan


def load_all_question_solutions(
    refactored_base_path, filter_not_passed, refactored_style, translation_style
):
    globbed_solution_path = f"{refactored_base_path}/*/{DATA_KEYS[refactored_style]}"
    globbed_solution_path = sorted(glob.glob(globbed_solution_path))
    assert len(globbed_solution_path) > 0, f"no files found at {globbed_solution_path}"
    question_solutions = defaultdict(list)

    do_translation = (
        refactored_style in TRANSLATION_KEYS and translation_style in TRANSLATION_KEYS
    )
    if not do_translation:
        print(f"Skipping translation from {refactored_style} to {translation_style}")

    for solution_path in tqdm.tqdm(globbed_solution_path):
        if filter_not_passed:  # only necessary for base solutions
            passed_path = get_passed_path(solution_path)
            with open(passed_path, "r") as f:
                passed = json.load(f)
            if not passed["global_check"]:
                continue

        if do_translation:
            translated_solution_path = translate_solution_path(
                solution_path, refactored_style, translation_style
            )
            if not os.path.exists(translated_solution_path):
                continue

        solution = RefactoredDataset.read_file(solution_path)

        question_path = get_question_path(solution_path)
        question_solutions[question_path].append(solution)
    assert len(question_solutions) > 0, "no solutions found"
    return question_solutions


def split_dict(question_solutions: dict[str, list[str]], split_fraction: float):
    total_solutions = sum(len(solutions) for solutions in question_solutions.values())
    eval_breakpoint = int(total_solutions * split_fraction)

    eval_dict = {}
    train_dict = {}

    solution_counter = 0

    items = list(question_solutions.items())
    random.shuffle(items)

    for question, solutions in items:
        solution_counter += len(solutions)
        if solution_counter <= eval_breakpoint:
            eval_dict[question] = solutions
        else:
            train_dict[question] = solutions

    return train_dict, eval_dict


def build_refactored_datasets(tokenizer, data_args):
    random.seed(data_args.seed)
    question_solutions: dict[str, list[str]] = load_all_question_solutions(
        data_args.refactored_base_path,
        data_args.filter_on_passed,
        data_args.refactored_style,
        data_args.final_style,
    )
    # sample data_args.max_total_samples
    if data_args.max_total_samples is not None:
        question_solutions = dict(
            random.sample(
                question_solutions.items(), data_args.max_total_samples
            )  # noqa
        )
    split_fraction: float = data_args.eval_split_percentage * 0.01

    train_question_solutions, eval_question_solutions = split_dict(
        question_solutions, split_fraction
    )

    train_dataset = RefactoredDataset(train_question_solutions, tokenizer, data_args)
    eval_dataset = RefactoredDataset(eval_question_solutions, tokenizer, data_args)
    return train_dataset, eval_dataset


class RefactoredDataset(torch.utils.data.Dataset):
    def __init__(self, question_solutions, tokenizer, data_args) -> None:
        super().__init__()

        self.question_solutions = question_solutions
        self.data_args = data_args
        self.tokenizer = tokenizer

        self.refactored_base_path = data_args.refactored_base_path
        self.refactored_style = data_args.refactored_style
        self.filter_on_passed = data_args.filter_on_passed

        self.max_tokens = data_args.block_size

        self._initialize()
        self._initialize_plan_pad()
        self._initialize_mask_plan()

    @staticmethod
    def read_file(path):
        with open(path, "r") as f:
            return f.read()

    @staticmethod
    def read_small_plan_code(path):
        return RefactoredDataset.read_file(path)
        paths = [
            os.path.join(os.path.dirname(path), "attempt_0.py"),
            os.path.join(os.path.dirname(path), "attempt_1.py"),
            os.path.join(os.path.dirname(path), "attempt_2.py"),
        ]

        plans = [RefactoredDataset.read_file(path) for path in paths]
        smallest_plan = min(plans, key=lambda x: len(x))
        return smallest_plan

    def merge(self, samples: list[tuple[str]], merge_count: int):
        sample_lengths = [sum([len(x[0]) for x in sample]) for sample in samples]
        i = 0
        j = i + merge_count
        new_samples = []
        while i < len(samples):
            new_sample_length = sum(sample_lengths[i:j])
            if new_sample_length > self.max_tokens:
                j = j - 1
                if j == i:
                    i = i + 1
                    j = i + merge_count
                continue
            new_sample = tuple(sum(samples[i:j], []))
            new_samples.append(new_sample)
            i = j
            j = i + merge_count
        return new_samples

    def _initialize(self):
        skip_count = 0
        cutoff_count = 0
        all_samples: list[tuple[list[str], bool]] = []
        all_token_lengths = {
            "question": [],
            "plan": [],
            "solution": [],
            "total": [],
        }
        answer_type = "\nUse Standard Input format\n"
        for question_path, solutions in tqdm.tqdm(self.question_solutions.items()):
            question_str = RefactoredDataset.read_file(question_path)

            for solution in solutions:
                # remove samples with long plan
                plan_lines = ""
                for solution_line in solution.split("\n"):
                    if solution_line.startswith("# "):
                        plan_lines += solution_line + "\n"
                    else:
                        break

                plan_str_tokens = self.tokenizer(plan_lines)["input_ids"]
                plan_tokens_count = len(plan_str_tokens)

                if plan_tokens_count > 450:
                    skip_count += 1
                    continue

                # remove samples with long questions
                q_str = f"QUESTION:\n{question_str}\n{answer_type}\nANSWER:\n\n"
                q_str_tokens = self.tokenizer(q_str)["input_ids"]

                q_tokens_count = len(q_str_tokens)

                if q_tokens_count > self.max_tokens:
                    skip_count += 1
                    continue

                solution_str_tokens = self.tokenizer(solution)["input_ids"] + [
                    self.tokenizer.eos_token_id
                ]
                # if solution_str_tokens[0] == self.tokenizer.bos_token_id:
                #     solution_str_tokens = solution_str_tokens[1:]

                solution_tokens_count = len(solution_str_tokens)

                total_tokens_count = q_tokens_count + solution_tokens_count

                if total_tokens_count > self.max_tokens:
                    cutoff_count += 1

                sample = [
                    (q_str_tokens, 1),
                    (solution_str_tokens, 0),
                ]
                all_token_lengths["question"].append(q_tokens_count)
                all_token_lengths["plan"].append(plan_tokens_count)
                all_token_lengths["solution"].append(solution_tokens_count)
                all_token_lengths["total"].append(total_tokens_count)
                all_samples.append(sample)

        print(f"Loaded {len(all_samples)} samples")
        print(f"Skipped {skip_count} samples")
        print(f"Solution cutoff in {cutoff_count} samples")
        avg_token_lengths = {k: np.mean(v) for k, v in all_token_lengths.items()}
        print(f"Average token lengths: {avg_token_lengths}")
        stdev_token_lengths = {k: np.std(v) for k, v in all_token_lengths.items()}
        print(f"Stdev token lengths: {stdev_token_lengths}")
        max_token_lengths = {k: np.max(v) for k, v in all_token_lengths.items()}
        print(f"Max token lengths: {max_token_lengths}")

        self.all_samples = all_samples

    def _initialize_plan_pad(self):
        skip_count = 0
        cutoff_count = 0
        all_plan_samples: list[tuple[list[str], bool]] = []
        all_token_lengths = {
            "question": [],
            "plan": [],
            "total": [],
        }
        if self.refactored_style not in PLAN_PAD_DATA_KEYS:
            return
        all_plan_txt_files = glob.glob(
            self.refactored_base_path
            + "/*/"
            + PLAN_PAD_DATA_KEYS[self.refactored_style]
        )
        for plan_txt_file in tqdm.tqdm(all_plan_txt_files):
            question_path = get_question_path(plan_txt_file)
            if question_path not in self.question_solutions:
                continue

            question_str = RefactoredDataset.read_file(question_path)
            plan_txt_str = RefactoredDataset.read_file(plan_txt_file)

            q_str = f"PLAN-QUESTION:\n{question_str}\nPLAN-ANSWER:\n\n"

            q_str_tokens = self.tokenizer(q_str)["input_ids"]

            if q_str_tokens[-1] == self.tokenizer.eos_token_id:
                q_str_tokens = q_str_tokens[:-1]

            q_tokens_count = len(q_str_tokens)

            if q_tokens_count > self.max_tokens:
                skip_count += 1
                continue

            plan_str_tokens = self.tokenizer(plan_txt_str)["input_ids"] + [
                self.tokenizer.eos_token_id
            ]
            if plan_str_tokens[0] == self.tokenizer.bos_token_id:
                plan_str_tokens = plan_str_tokens[1:]

            plan_tokens_count = len(plan_str_tokens)

            total_tokens_count = q_tokens_count + plan_tokens_count

            if total_tokens_count > self.max_tokens:
                cutoff_count += 1

            sample = [
                (q_str_tokens, 1),
                (plan_str_tokens, 0),
            ]
            all_token_lengths["question"].append(q_tokens_count)
            all_token_lengths["plan"].append(plan_tokens_count)
            all_token_lengths["total"].append(total_tokens_count)
            all_plan_samples.append(sample)

        print(f"Loaded {len(all_plan_samples)} samples")
        print(f"Skipped {skip_count} samples")
        print(f"Solution cutoff in {cutoff_count} samples")
        avg_token_lengths = {k: np.mean(v) for k, v in all_token_lengths.items()}
        print(f"Average token lengths: {avg_token_lengths}")
        stdev_token_lengths = {k: np.std(v) for k, v in all_token_lengths.items()}
        print(f"Stdev token lengths: {stdev_token_lengths}")
        max_token_lengths = {k: np.max(v) for k, v in all_token_lengths.items()}
        print(f"Max token lengths: {max_token_lengths}")

        all_plan_samples = self.merge(
            all_plan_samples, self.data_args.plan_pad_merge_count
        )
        print(f"After merging, {len(all_plan_samples)} samples")
        self.all_samples.extend(all_plan_samples)

    def _initialize_mask_plan(self):
        skip_count = 0
        cutoff_count = 0
        all_maskplan_samples: list[tuple[list[str], bool]] = []
        all_token_lengths = {
            "question": [],
            "plan": [],
            "solution": [],
            "total": [],
        }
        if self.refactored_style not in PLAN_MASK_DATA_KEYS:
            return
        all_plan_attempt_files = glob.glob(
            self.refactored_base_path
            + "/*/"
            + PLAN_MASK_DATA_KEYS[self.refactored_style]
        )
        answer_type = "\nUse Standard Input format\n"
        for plan_attempt_file in tqdm.tqdm(all_plan_attempt_files):
            question_path = get_question_path(plan_attempt_file)
            if question_path not in self.question_solutions:
                continue

            question_str = RefactoredDataset.read_file(question_path)
            solution_str = RefactoredDataset.read_file(plan_attempt_file)

            # remove samples with long plan
            solution_lines = solution_str.split("\n")
            code_line_index = ["# CODE" in line for line in solution_lines]
            assert (
                sum(code_line_index) == 1
            ), f"Multiple code lines in {plan_attempt_file}"
            code_line_index = code_line_index.index(True)

            plan_str = "\n".join(solution_lines[:code_line_index])
            plan_str_tokens = self.tokenizer(plan_str)["input_ids"]

            plan_tokens_count = len(plan_str_tokens)

            if plan_tokens_count > 450:
                skip_count += 1
                continue

            # remove samples with long questions
            q_str = f"QUESTION:\n{question_str}\n{answer_type}\nANSWER:\n\n"
            q_str_tokens = self.tokenizer(q_str)["input_ids"]

            q_tokens_count = len(q_str_tokens)

            if q_tokens_count > self.max_tokens:
                skip_count += 1
                continue

            code_str = "\n".join(solution_lines[code_line_index:])
            code_str_tokens = self.tokenizer(code_str)["input_ids"] + [
                self.tokenizer.eos_token_id
            ]

            code_tokens_count = len(code_str_tokens)

            total_tokens_count = q_tokens_count + plan_tokens_count + code_tokens_count

            if total_tokens_count > self.max_tokens:
                cutoff_count += 1

            sample = [
                (q_str_tokens, 1),
                (plan_str_tokens, 0),
                (code_str_tokens, 1),
            ]
            all_token_lengths["question"].append(q_tokens_count)
            all_token_lengths["plan"].append(plan_tokens_count)
            all_token_lengths["solution"].append(code_tokens_count)
            all_token_lengths["total"].append(total_tokens_count)
            all_maskplan_samples.append(sample)

        print(f"Loaded {len(all_maskplan_samples)} samples")
        print(f"Skipped {skip_count} samples")
        print(f"Solution cutoff in {cutoff_count} samples")
        avg_token_lengths = {k: np.mean(v) for k, v in all_token_lengths.items()}
        print(f"Average token lengths: {avg_token_lengths}")
        stdev_token_lengths = {k: np.std(v) for k, v in all_token_lengths.items()}
        print(f"Stdev token lengths: {stdev_token_lengths}")
        max_token_lengths = {k: np.max(v) for k, v in all_token_lengths.items()}
        print(f"Max token lengths: {max_token_lengths}")

        self.all_samples.extend(all_maskplan_samples)

    def pack_samples(self, idx):
        input_ids = []
        label_ids = []
        curr_num_tokens = 0

        sample = self.all_samples[idx]

        while curr_num_tokens < self.max_tokens:
            for tokens, mask in sample:
                input_ids.extend(tokens)
                len_tokens = len(tokens)
                if mask:
                    label_ids.extend([-100] * len_tokens)
                else:
                    label_ids.extend(tokens)

                curr_num_tokens += len_tokens

        input_ids = input_ids[: self.max_tokens]
        label_ids = label_ids[: self.max_tokens]
        return input_ids, label_ids

    def __getitem__(self, idx):
        input_ids, label_ids = self.pack_samples(idx)
        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(label_ids),
        }

    def __len__(self):
        return len(self.all_samples)


if __name__ == "__main__":
    from dataloaders.data_arguments import DataArguments

    setattr(DataArguments, "seed", 0)
    setattr(DataArguments, "cache_dir", None)
    setattr(
        DataArguments,
        "refactored_base_path",
        "/home/naman/Repos/CodeQuality/apps_enumerated_old",
        # "/home/naman/Repos/CodeQuality/code_contests_enumerated_train",
    )
    setattr(DataArguments, "refactored_style", "remodularize_merged")
    setattr(DataArguments, "final_style", "remodularize_merged")
    setattr(DataArguments, "filter_on_passed", False)
    setattr(DataArguments, "max_total_samples", 800)
    # setattr(DataArguments, "final_style", "modularize_original")

    tokenizer = AutoTokenizer.from_pretrained(
        "codellama/CodeLlama-7b-hf",
        use_auth_token=True,
        trust_remote_code=True,
    )

    dataset, _ = build_refactored_datasets(tokenizer, DataArguments)

    print(len(dataset))
    # print(dataset[0])

    # print(tokenizer.convert_ids_to_tokens(dataset[0]["input_ids"]))
    # print(
    #     tokenizer.convert_ids_to_tokens(
    #         [tokenizer.eos_token_id if x == -100 else x for x in dataset[0]["labels"]]
    #     )
    # )
