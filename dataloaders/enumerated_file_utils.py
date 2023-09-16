DATA_KEYS = {
    "base_original": "all_solutions/*/base/base/original/solution.py",
    "rename_original": "all_solutions/*/rename/EPC_RV2/original/solution.py",
    "modularize_original": "all_solutions/*/modularize/EPC_M2/original/solution.py",
    "remodularize_merged": "all_solutions/*/remodularize/EPC_RMFN/remod_merged/solution.py",
    "plan_merged1": "all_solutions/*/planning/EPC_LPFN/plans_merged/attempt_[0].py",
    "plan_merged2": "all_solutions/*/planning/EPC_LPFN/plans_merged/attempt_[01].py",
    "plan_merged4": "all_solutions/*/planning/EPC_LPFN/plans_merged/attempt_[0123].py",
    "plan_merged6": "all_solutions/*/planning/EPC_LPFN/plans_merged/attempt_[012345].py",
    "plan_merged1padall": "all_solutions/*/planning/EPC_LPFN/plans_merged/attempt_[0].py",
    "plan_merged1pad1": "all_solutions/*/planning/EPC_LPFN/plans_merged/attempt_[0].py",
    "plan_merged1pad2": "all_solutions/*/planning/EPC_LPFN/plans_merged/attempt_[0].py",
    "plan_merged2padall": "all_solutions/*/planning/EPC_LPFN/plans_merged/attempt_[01].py",
    "plan_merged2pad1": "all_solutions/*/planning/EPC_LPFN/plans_merged/attempt_[01].py",
    "plan_merged2pad2": "all_solutions/*/planning/EPC_LPFN/plans_merged/attempt_[01].py",
    "plan_merged4padall": "all_solutions/*/planning/EPC_LPFN/plans_merged/attempt_[0123].py",
}

PLAN_PAD_DATA_KEYS = {
    "plan_merged1padall": "all_solutions/*/planning/EPC_LPFN/plans/plan_[12345].txt",
    "plan_merged2padall": "all_solutions/*/planning/EPC_LPFN/plans/plan_[2345].txt",
    "plan_merged4padall": "all_solutions/*/planning/EPC_LPFN/plans/plan_[45].txt",
    "plan_merged1pad2": "all_solutions/*/planning/EPC_LPFN/plans/plan_[12].txt",
    "plan_merged2pad2": "all_solutions/*/planning/EPC_LPFN/plans/plan_[23].txt",
    "plan_merged1pad1": "all_solutions/*/planning/EPC_LPFN/plans/plan_[1].txt",
    "plan_merged2pad1": "all_solutions/*/planning/EPC_LPFN/plans/plan_[2].txt",
}

TRANSLATION_KEYS = {
    "base_original": "base/base/original",
    "rename_original": "rename/EPC_RV2/original",
    "modularize_original": "modularize/EPC_M2/original",
    "remodularize_merged": "remodularize/EPC_RMFN/remod_merged",
}


def get_passed_path(solution_path: str):
    return solution_path.replace(".py", "_passed.json")


def get_question_path(solution_path: str):
    return solution_path.split("all_solutions")[0] + "question/question.txt"


def translate_solution_path(solution_path: str, current_key: str, final_key: str):
    assert current_key in TRANSLATION_KEYS
    assert final_key in TRANSLATION_KEYS

    return solution_path.replace(
        TRANSLATION_KEYS[current_key], TRANSLATION_KEYS[final_key]
    )
