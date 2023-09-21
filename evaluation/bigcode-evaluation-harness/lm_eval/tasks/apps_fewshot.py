"""Measuring Coding Challenge Competence With APPS
https://arxiv.org/abs/2105.09938

APPS is a benchmark for code generation with 10000 problems. With three difficulty levels: introductory, interview and competition.
It can be used to evaluate the ability of language models to generate code from natural language specifications.

Homepage: https://github.com/hendrycks/apps
"""

import os
import json

import numpy as np
import pandas as pd
from evaluate import load

from lm_eval.base import Task

_CITATION = """
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""


LEVELS = ["introductory", "interview", "competition"]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {apps-interview: Task, apps-competitoon: Task}
    """
    # tasks1 = {f"apps-{level}": create_task(level) for level in LEVELS}
    tasks2 = {
        f"appsfewshot-{level}-{style}": create_task(
            level, style, ["codeforces", "codechef", "atcoder"]
        )
        for level in LEVELS
        for style in ["base", "rename", "modularize", "plan"]
    }
    # tasks3 = {
    #     f"apps-{level}-cf-kattis": create_task(
    #         level, ["codeforces", "codechef", "atcoder", "kattis"]
    #     )
    #     for level in LEVELS
    # }
    return {**tasks2}


def create_task(level, style, platforms=None):
    class APPS(FewShotAPPS):
        def __init__(self):
            super().__init__(level, style, platforms)

    return APPS


class FewShotAPPS(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    if os.path.exists("../hf_data/datasets--codeparrot--apps/"):
        print("using local apps")
        DATASET_PATH = "../hf_data/datasets--codeparrot--apps/"
    else:
        DATASET_PATH = "codeparrot/apps"
    DATASET_NAME = None
    SPLITS = ["test"]

    def __init__(self, level, style, platforms=None):
        self.DATASET_NAME = level
        self.style = style
        self.platforms = platforms
        super().__init__(
            stop_words=["\nQUESTION", "\n---", "\nANSWER"],
            requires_execution=True,
        )
        self.filter_by_platform()
        self.build_few_shot_examples()

    def filter_by_platform(self):
        self.dataset = self.dataset["test"]
        indices = np.where(
            (self.dataset.to_pandas()["difficulty"] == self.DATASET_NAME).values
        )[0]
        self.dataset = self.dataset.select(indices)

        platforms = pd.Series(self.dataset["url"]).str.split(".")
        platforms0 = platforms.str[0].str.split("/").str[-1]
        platforms0[platforms0.isin(["open", "www"])] = platforms[
            platforms0.isin(["open", "www"])
        ].str[1]
        if self.platforms is not None:
            indices = np.where(platforms0.isin(self.platforms).values)[0]
            self.dataset = self.dataset.select(indices)
            print(
                f"Filtering by platforms: {self.platforms} - {len(self.dataset)} problems left"
            )

    def build_few_shot_examples(self):
        import os

        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(
            f"{dir_path}/few_shot_examples/apps_few_shot_examples.json", "r"
        ) as fp:
            print(self.style)
            few_shot_examples = json.load(fp)[self.style]

        self.base_prompt = ""
        for example in few_shot_examples:
            question_str = example["question"]
            answer_type = "\nUse Standard Input format\n"
            q_str = f"QUESTION:\n{question_str}\n{answer_type}\nANSWER:\n\n"
            self.base_prompt += q_str
            answer = example["answer"]
            a_str = answer + "\n\n---\n"
            self.base_prompt += a_str

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def get_prompt_old(self, doc):
        """Generate prompts for APPS
        Finetuning setup: prompt=question  with some starter code and function name if they exist.
        We also specify the type of the prompt, i.e. whether it is call-based or standard input-based.
        """
        starter_code = None if len(doc["starter_code"]) == 0 else doc["starter_code"]
        try:
            input_outpout = json.loads(doc["input_output"])
            fn_name = (
                None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
            )
        except ValueError:
            fn_name = None
        prompt = "\nQUESTION:\n"
        prompt += doc["question"]
        if starter_code:
            prompt += starter_code
        if not fn_name:
            call_format = "\nUse Standard Input format"
            prompt += call_format
        else:
            call_format = "\nUse Call-Based format"
            prompt += call_format
        prompt += "\nANSWER:\n"
        return prompt

    def get_prompt(self, doc):
        # starter_code = "" if len(doc["starter_code"]) == 0 else doc["starter_code"]

        # try:
        #     input_outpout = json.loads(doc["input_output"])
        #     fn_name = (
        #         None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
        #     )
        # except ValueError:
        #     fn_name = None

        # answer_type = (
        #     "\nUse Standard Input format\n"
        #     if not fn_name
        #     else "\nUse Call-Based format\n"
        # )
        question_str = doc["question"]
        # prompt = (
        #     "\nQUESTION:\n"
        #     + question_str
        #     + "\n"
        #     + starter_code
        #     + "\n"
        #     + answer_type
        #     + "\nANSWER:\n"

        # )
        answer_type = "\nUse Standard Input format\n"
        q_str = f"QUESTION:\n{question_str}\n{answer_type}\nANSWER:\n\n"
        prompt = q_str
        # q_str_tokens = self.tokenizer(q_str)["input_ids"]

        # if q_str_tokens[-1] == self.tokenizer.eos_token_id:
        #     q_str_tokens = q_str_tokens[:-1]
        return self.base_prompt + prompt

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
            generation = generation.split("\nANSWER:")[-1]
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
        from lm_eval.tasks.custom_metrics.apps_custom_metrics.utils import (
            compute_metrics,
        )

        results = compute_metrics(
            self.dataset, generations, k_list=[1, 5, 10, 25, 50, 75, 100, 125, 150]
        )
        return results


if __name__ == "__main__":
    task = FewShotAPPS("introductory", "plan", ["codeforces", "codechef", "atcoder"])
    print(task)
    dataset = task.get_dataset()
    p1 = task.get_prompt(dataset[0])
    # p2 = task.get_prompt2(dataset[0])
    # print(p1 == p2)
    print(p1)
    # print(p2)
