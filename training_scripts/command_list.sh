# ----- CODE CONTENTS
# 13b
# original vs remodularization
bash training_scripts/run_contests_llama13b_a100_80.sh base remod
bash training_scripts/run_contests_llama13b_a100_80.sh remod remod

# planning
bash training_scripts/run_contests_llama13b_a100_80.sh planm1 remod
bash training_scripts/run_contests_llama13b_a100_80.sh planm1maskall remod
bash training_scripts/run_contests_llama13b_a100_80.sh planm1padall remod

# ablation - type of cleaning
bash training_scripts/run_contests_llama13b_a100_80.sh rename remod
bash training_scripts/run_contests_llama13b_a100_80.sh mod remod

# 7b
# original vs remodularization
bash training_scripts/run_contests_llama7b_a100_80.sh base remod
bash training_scripts/run_contests_llama7b_a100_80.sh remod remod

# planning
bash training_scripts/run_contests_llama7b_a100_80.sh planm1 remod
bash training_scripts/run_contests_llama7b_a100_80.sh planm1maskall remod
bash training_scripts/run_contests_llama7b_a100_80.sh planm1padall remod

# ablation - type of cleaning
bash training_scripts/run_contests_llama7b_a100_80.sh rename remod
bash training_scripts/run_contests_llama7b_a100_80.sh mod remod

# ----- APPS

# 13b
# original vs remodularization
bash training_scripts/run_apps_llama13b_a100_80.sh base remod 2
bash training_scripts/run_apps_llama13b_a100_80.sh remod remod 2

# planning
bash training_scripts/run_apps_llama13b_a100_80.sh planm1 remod 2
bash training_scripts/run_apps_llama13b_a100_80.sh planm2 remod 1
bash training_scripts/run_apps_llama13b_a100_80.sh planm2padall remod 1
bash training_scripts/run_apps_llama13b_a100_80.sh planm1padall remod 2
bash training_scripts/run_apps_llama13b_a100_80.sh planm2maskall remod 1
bash training_scripts/run_apps_llama13b_a100_80.sh planm1maskall remod 2

# ablation - type of cleaning
bash training_scripts/run_apps_llama13b_a100_80.sh rename remod 2
bash training_scripts/run_apps_llama13b_a100_80.sh mod remod 2


# 7b
# original vs remodularization
bash training_scripts/run_apps_llama7b_a100_80.sh base remod 2
bash training_scripts/run_apps_llama7b_a100_80.sh remod remod 2

# planning
bash training_scripts/run_apps_llama7b_a100_80.sh planm1 remod 2
bash training_scripts/run_apps_llama7b_a100_80.sh planm2 remod 1
bash training_scripts/run_apps_llama7b_a100_80.sh planm2padall remod 1
bash training_scripts/run_apps_llama7b_a100_80.sh planm1padall remod 2
bash training_scripts/run_apps_llama7b_a100_80.sh planm2maskall remod 1
bash training_scripts/run_apps_llama7b_a100_80.sh planm1maskall remod 2

# ablation - type of cleaning
bash training_scripts/run_apps_llama7b_a100_80.sh rename remod 2
bash training_scripts/run_apps_llama7b_a100_80.sh mod remod 2

# ablation - data size
bash training_scripts/run_apps_datasize_a100_80.sh remod remod 100 5
bash training_scripts/run_apps_datasize_a100_80.sh remod remod 200 4
bash training_scripts/run_apps_datasize_a100_80.sh remod remod 400 3 
bash training_scripts/run_apps_datasize_a100_80.sh remod remod 800 2
bash training_scripts/run_apps_datasize_a100_80.sh base remod 100 5
bash training_scripts/run_apps_datasize_a100_80.sh base remod 200 4
bash training_scripts/run_apps_datasize_a100_80.sh base remod 400 3
bash training_scripts/run_apps_datasize_a100_80.sh base remod 800 2
bash training_scripts/run_apps_datasize_a100_80.sh remod remod 1600 2
bash training_scripts/run_apps_datasize_a100_80.sh base remod 1600 2
