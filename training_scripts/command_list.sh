# ----- CODE CONTENTS

# 7b
# original vs remodularization
bash training_scripts/run_contests_llama7b_a100_80.sh base remod
bash training_scripts/run_contests_llama7b_a100_80.sh remod remod

# planning
bash training_scripts/run_contests_llama7b_a100_80.sh planm1 remod
bash training_scripts/run_contests_llama7b_a100_80.sh planm1pad1 remod
bash training_scripts/run_contests_llama7b_a100_80.sh planm1padall remod

# ablation - type of cleaning
bash training_scripts/run_contests_llama7b_a100_80.sh rename remod
bash training_scripts/run_contests_llama7b_a100_80.sh mod remod

# ----- APPS

# 7b
# original vs remodularization
bash training_scripts/run_apps_llama7b_a100_80.sh base remod 2
bash training_scripts/run_apps_llama7b_a100_80.sh remod remod 2

# planning
bash training_scripts/run_apps_llama7b_a100_80.sh planm1 remod 2
bash training_scripts/run_apps_llama7b_a100_80.sh planm2 remod 1
bash training_scripts/run_apps_llama7b_a100_80.sh planm2padall remod 1
bash training_scripts/run_apps_llama7b_a100_80.sh planm1padall remod 2
bash training_scripts/run_apps_llama7b_a100_80.sh planm2pad2 remod 1

# ablation - type of cleaning
bash training_scripts/run_apps_llama7b_a100_80.sh rename remod 2
bash training_scripts/run_apps_llama7b_a100_80.sh mod remod 2

# ablation - data size
bash training_scripts/run_apps_datasize_a100_80.sh remod remod 200
bash training_scripts/run_apps_datasize_a100_80.sh remod remod 400
bash training_scripts/run_apps_datasize_a100_80.sh remod remod 800
bash training_scripts/run_apps_datasize_a100_80.sh remod remod 1600
bash training_scripts/run_apps_datasize_a100_80.sh base remod 200
bash training_scripts/run_apps_datasize_a100_80.sh base remod 400
bash training_scripts/run_apps_datasize_a100_80.sh base remod 800
bash training_scripts/run_apps_datasize_a100_80.sh base remod 1600

# ablation - openai