from pprint import pprint

from . import (
    apps,
    code_contests,
    apps_fewshot,
    code_contests_fewshot,
    codecontests_test,
    codecontests_testplan,
    codecontests_fewshotplan,
)

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **code_contests.create_all_tasks(),
    **apps_fewshot.create_all_tasks(),
    **code_contests_fewshot.create_all_tasks(),
    **codecontests_test.create_all_tasks(),
    **codecontests_testplan.create_all_tasks(),
    **codecontests_fewshotplan.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
