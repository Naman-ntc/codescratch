"""Measuring Coding Challenge Competence With APPS
https://arxiv.org/abs/2105.09938

APPS is a benchmark for code generation with 10000 problems. With three difficulty levels: introductory, interview and competition.
It can be used to evaluate the ability of language models to generate code from natural language specifications.

Homepage: https://github.com/hendrycks/apps
"""

import os
import json
import glob
import random

import numpy as np
import pandas as pd
from evaluate import load
from datasets import concatenate_datasets

from lm_eval.base import Task

_CITATION = """
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""


LEVELS = ["easy", "medium", "hard"]
DIFFICULTIES = {
    "easy": [7, 8, 9],
    "medium": [10, 11],
    "hard": [12, 13, 14, 15, 16, 17, 18, 19, 20],
}


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {codecontests: Task, codecontests-easy: Task}
    """
    tasks1 = {f"codecontests-testplan": create_task() for level in LEVELS}
    return {**tasks1}


def create_task():
    class CodeContestsPlan(GeneralCodeContestsTestPlan):
        def __init__(self):
            super().__init__()

    return CodeContestsPlan


class GeneralCodeContestsTestPlan(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    if os.path.exists("../hf_data/datasets--deepmind--code_contests/"):
        DATASET_PATH = "../hf_data/datasets--deepmind--code_contests/"
    else:
        DATASET_PATH = "deepmind/code_contests"
    SPLITS = ["test"]

    def __init__(self):
        super().__init__(
            stop_words=["\nQUESTION", "\n---", "\nANSWER"],
            requires_execution=True,
        )
        self.filter_by_platform()
        all_plans = "../gptturbo_plans.json"
        with open(all_plans, "r") as f:
            self.all_plans = json.load(f)

    def filter_by_platform(self):
        self.dataset = self.dataset["test"]
        self.dataset_pandas = self.dataset.to_pandas()

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def get_prompt(self, doc):
        """Generate prompts for APPS
        Finetuning setup: prompt=question  with some starter code and function name if they exist.
        We also specify the type of the prompt, i.e. whether it is call-based or standard input-based.
        """
        index = int(
            self.dataset_pandas[(self.dataset_pandas.name == doc["name"])].index[0]
        )
        plans_per_q = self.all_plans[index]
        select_plan = random.choice(plans_per_q)
        select_plan = "\n".join(
            ["# " + pline for pline in select_plan.split("\n") if pline]
        )

        question_str = doc["description"]
        answer_type = "\nUse Standard Input format\n"
        q_str = f"QUESTION:\n{question_str}\n{answer_type}\nANSWER:\n\n"
        if select_plan:
            q_str = q_str + "# PLAN\n" + select_plan + "\n\n# CODE\n" + "\n"
        prompt = q_str
        return prompt

    def get_prompt_old(self, doc):
        """Generate prompts for APPS
        Finetuning setup: prompt=question  with some starter code and function name if they exist.
        We also specify the type of the prompt, i.e. whether it is call-based or standard input-based.
        """
        index = int(
            self.dataset_pandas[(self.dataset_pandas.name == doc["name"])].index[0]
        )
        glob_plan_path = f"/home/naman/Repos/CodeQuality/code_contests_enumerated_test/row_{index}/all_solutions/*/planning/EPC_LPFN/plans/*.txt"
        all_plans = glob.glob(glob_plan_path)
        select_plan = None
        if len(all_plans) > 0:
            for i in range(len(all_plans)):
                with open(all_plans[i], "r") as f:
                    plan = f.read()
                all_plans[i] = plan
            all_plans = [
                "\n".join(["# " + pline for pline in plan.split("\n") if pline])
                for plan in all_plans
            ]
            filtered_all_plans = [
                plan for plan in all_plans if len(plan.split("\n")) < 6
            ]
            if filtered_all_plans:
                all_plans = filtered_all_plans
            select_plan = max(all_plans, key=lambda x: len(x.split("\n")))

        question_str = doc["description"]
        answer_type = "\nUse Standard Input format\n"
        q_str = f"QUESTION:\n{question_str}\n{answer_type}\nANSWER:\n\n"
        if select_plan:
            q_str = q_str + "# PLAN\n" + select_plan + "\n\n# CODE\n" + "\n"
        # prompt = "\nQUESTION:\n" + question_str + "\n" + answer_type + "\nANSWER:\n"
        prompt = q_str
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return None

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for APPS)
        """
        try:
            generation = generation.split("\nANSWER:", 1)[1]
            generation = self._stop_at_stop_token(generation, self.stop_words)
            generation = generation.strip()
        except IndexError:
            # happens when prompts were very long and got truncated
            pass
        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences (not needed for APPS Task)
        """
        # code_metric = load("codeparrot/apps_metric")
        # results = code_metric.compute(
        #     predictions=generations, k_list=[1, 10, 100], level=self.DATASET_NAME
        # )
        from lm_eval.tasks.custom_metrics.code_contests_custom_metrics.utils import (
            compute_metrics,
        )

        results = compute_metrics(
            self.dataset, generations, k_list=[1, 5, 10, 25, 50, 75, 100, 125, 150]
        )
        return results


if __name__ == "__main__":
    task = GeneralCodeContestsTestPlan()
    print(task)
    dataset = task.get_dataset()
    sample = dataset[10]
    p1 = task.get_prompt(sample)
    print(p1)

    # sample = dataset[1]
    # p1 = task.get_prompt(sample)
    # print(p1)

    # sample = dataset[28]
    # p1 = task.get_prompt(sample)
    # print(p1)

    public = sample["public_tests"]
    private = sample["private_tests"]
    generated = sample["generated_tests"]
    all_inputs = public["input"] + private["input"] + generated["input"]
    all_outputs = public["output"] + private["output"] + generated["output"]
    assert len(all_inputs) == len(all_outputs)
    in_outs = {"inputs": all_inputs, "outputs": all_outputs}

    print(len(all_inputs))
