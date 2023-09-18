import json

base1 = "/home/naman/Repos/CodeQuality/apps_enumerated_old/row_2/all_solutions/1/base/base/original/solution.py"
rename1 = "/home/naman/Repos/CodeQuality/apps_enumerated_old/row_2/all_solutions/1/rename/EPC_RV2/original/solution.py"
remod1 = "/home/naman/Repos/CodeQuality/apps_enumerated_old/row_2/all_solutions/1/remodularize/EPC_RMFN/remod_merged/solution.py"
plan1 = "/home/naman/Repos/CodeQuality/apps_enumerated_old/row_2/all_solutions/1/planning/EPC_LPFN/plans_merged/attempt_0.py"
question1 = (
    "/home/naman/Repos/CodeQuality/apps_enumerated_old/row_2/question/question.txt"
)

base2 = "/home/naman/Repos/CodeQuality/apps_enumerated_old/row_285/all_solutions/11/base/base/original/solution.py"
rename2 = "/home/naman/Repos/CodeQuality/apps_enumerated_old/row_285/all_solutions/11/rename/EPC_RV2/original/solution.py"
remod2 = "/home/naman/Repos/CodeQuality/apps_enumerated_old/row_285/all_solutions/11/remodularize/EPC_RMFN/remod_merged/solution.py"
plan2 = "/home/naman/Repos/CodeQuality/apps_enumerated_old/row_285/all_solutions/11/planning/EPC_LPFN/plans_merged/attempt_0.py"
question2 = (
    "/home/naman/Repos/CodeQuality/apps_enumerated_old/row_285/question/question.txt"
)
all_vars = [
    "base1",
    "rename1",
    "remod1",
    "plan1",
    "question1",
    "base2",
    "rename2",
    "remod2",
    "plan2",
    "question2",
]
for var in all_vars:
    with open(locals()[var], "r") as fp:
        locals()[var] = fp.read()

few_shot_examples = {
    "base": [
        {
            "question": question1,
            "answer": base1,
        },
        {
            "question": question2,
            "answer": base2,
        },
    ],
    "rename": [
        {
            "question": question1,
            "answer": rename1,
        },
        {
            "question": question2,
            "answer": rename2,
        },
    ],
    "modularize": [
        {
            "question": question1,
            "answer": remod1,
        },
        {
            "question": question2,
            "answer": remod2,
        },
    ],
    "plan": [
        {
            "question": question1,
            "answer": plan1,
        },
        {
            "question": question2,
            "answer": plan2,
        },
    ],
}

with open("apps_few_shot_examples.json", "w") as fp:
    json.dump(few_shot_examples, fp, indent=4)
