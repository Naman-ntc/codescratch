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


def load_all_oai_question_solutions(refactored_base_path):
    globbed_passed_path = f"{refactored_base_path}/*/openai_solutions/*_passed.json"
    globbed_passed_path = sorted(glob.glob(globbed_passed_path))

    question_solutions = defaultdict(list)

    for passed_path in tqdm.tqdm(globbed_passed_path):
        with open(passed_path, "r") as f:
            passed_data = json.load(f)
        if not passed_data["global_check"]:
            continue
        question_path = (
            passed_path.split("openai_solutions")[0] + "question/question.txt"
        )
        solution_path = passed_path.replace("_passed.json", ".py")

        solution = OAIDataset.read_file(solution_path)

        question_solutions[question_path].append(solution)
    question_solutions = {q: list(set(s)) for q, s in question_solutions.items()}
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


def convert_to_remod(question_solutions: dict[str, list[str]]):
    remod_question_solutions = defaultdict(list)
    for question, oai_solutions in question_solutions.items():
        num_oai_solutions = len(oai_solutions)
        folder_path = question.split("question")[0]
        remod_soln_path = f"{folder_path}/{DATA_KEYS['remodularize_merged']}"
        globbed_soln_path = sorted(glob.glob(remod_soln_path))
        globbed_soln_path = globbed_soln_path
        globbed_soln_path = globbed_soln_path[:num_oai_solutions]
        for soln_path in globbed_soln_path:
            solution = OAIDataset.read_file(soln_path)
            remod_question_solutions[question].append(solution)

    return remod_question_solutions


# def filter_by_remod(question_solutions: dict[str, list[str]]):
#     for question, oai_solutions in question_solutions.items():
#         num_oai_solutions = len(oai_solutions)
#         folder_path = question.split("question")[0]
#         remod_soln_path = f"{folder_path}/{DATA_KEYS['remodularize_merged']}"
#         globbed_soln_path = sorted(glob.glob(remod_soln_path))
#         if len(globbed_soln_path) < num_oai_solutions:
#             print(
#                 f"Found {num_oai_solutions} keeping {len(globbed_soln_path)} @ {question}"
#             )
#             question_solutions[question] = oai_solutions[: len(globbed_soln_path)]
#     return question_solutions


def build_oai_datasets(tokenizer, data_args):
    assert data_args.refactored_style in ["base_original", "remodularize_merged"]
    question_solutions: dict[str, list[str]] = load_all_oai_question_solutions(
        data_args.refactored_base_path,
    )

    if data_args.refactored_style == "remodularize_merged":
        question_solutions = convert_to_remod(question_solutions)
    # else:
    #     question_solutions = filter_by_remod(question_solutions)

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

    train_dataset = OAIDataset(train_question_solutions, tokenizer, data_args)
    eval_dataset = OAIDataset(eval_question_solutions, tokenizer, data_args)
    return train_dataset, eval_dataset


class OAIDataset:
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

    @staticmethod
    def read_file(path):
        with open(path, "r") as f:
            return f.read()

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
        for question_path, solutions in tqdm.tqdm(self.question_solutions.items()):
            question_str = OAIDataset.read_file(question_path)
            answer_type = "\nUse Standard Input format\n"

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

                if len(plan_str_tokens) > 450:
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
    setattr(DataArguments, "refactored_style", "base_original")
    setattr(DataArguments, "refactored_style", "remodularize_merged")
    # setattr(DataArguments, "max_total_samples", 100)

    tokenizer = AutoTokenizer.from_pretrained(
        "codellama/CodeLlama-7b-hf",
        use_auth_token=True,
        trust_remote_code=True,
    )

    dataset, _ = build_oai_datasets(tokenizer, DataArguments)

    print(len(dataset))
    # print(dataset[0])

    # print(tokenizer.convert_ids_to_tokens(dataset[0]["input_ids"]))
    # print(
    #     tokenizer.convert_ids_to_tokens(
    #         [tokenizer.eos_token_id if x == -100 else x for x in dataset[0]["labels"]]
    #     )
    # )
