# Performing Bayesian Optimization with N_global=10

# This is a conceptual backbone of the pipeline sript to be adapted to the HPC environment.

# STEP 1: Bayesian optimization
# The codes in this section can be run in parallel

python BO_run.py --config-py NiPS3_config.py --init-seed 0 --run-ind 1
python BO_run.py --config-py NiPS3_config.py --init-seed 10 --run-ind 2
python BO_run.py --config-py NiPS3_config.py --init-seed 20 --run-ind 3
python BO_run.py --config-py NiPS3_config.py --init-seed 30 --run-ind 4
python BO_run.py --config-py NiPS3_config.py --init-seed 40 --run-ind 5
python BO_run.py --config-py NiPS3_config.py --init-seed 50 --run-ind 6
python BO_run.py --config-py NiPS3_config.py --init-seed 60 --run-ind 7
python BO_run.py --config-py NiPS3_config.py --init-seed 70 --run-ind 8
python BO_run.py --config-py NiPS3_config.py --init-seed 80 --run-ind 9
python BO_run.py --config-py NiPS3_config.py --init-seed 90 --run-ind 10

# STEP 2: Greedy refinement
# After BO is finished:

# Query number of best points satisfying selection criteria in the config file

python BO_refine.py --config-py NiPS3_config.py --num-points

## returns e.g. 23

# Greedy refinement for the best points, here e.g. in batches of six
# The codes of the section below can be run in parallel

python BO_refine.py --config-py NiPS3_config.py --index1 0 --index2 6
python BO_refine.py --config-py NiPS3_config.py 6 12
python BO_refine.py --config-py NiPS3_config.py 12 18
python BO_refine.py --config-py NiPS3_config.py 18 23

# combine outputs of greedy fits into single file

./combine_greedy_results.sh

# STEP 3: Clustering refined points to get physically nonequivalent candidates

python BO_cluster.py --config-py NiPS3_config.py --input NiPS3_L1sum_Final_s0.025.txt --output NiPS3_clusters

# STEP 4: Select candidate point, and write coordinates to json, e.g. testparam_NiPS3.json

# STEP 5: Calculate confidence bounds and annotate

python BO_annotate.py --config-py NiPS3_config.py --params testparam_NiPS3.json
python BO_intervals.py --config-py NiPS3_config.py --params testparam_NiPS3.json

# STEP 6: Plot results using jupyter notebook

# Cross-check: test distance value at the point specified by testparam_NiPS3.json

python BO_test.py --config-py NiPS3_config.py --params testparam_NiPS3.json
