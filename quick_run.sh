# Run once with config
# python run_pmds.py -d digits012 -m MAP3 --use_pre_config --normalize_dists
# python run_pmds.py -d cities_us_toy -m MAP2 --use_pre_config --normalize_dists --no_logging

# commands to run experiment + plots for paper
# use the same command as usual with `--experiment_mode` and `--exp_xxx`
# by default it will remake the plots using stored embedding files.
# add `--exp_re_run` to re-run the exp.

# Alignment of US cities
# python plot_with_us_map.py

# run experiment with missing pairs 20 times for 20 values of p
# without --exp_re_run: only replot figures:
# plots/MAP2/digits5/{Z_with_missing_pairs.png, score_with_missing_pairs.png}
# if re_run, override score file{scores.csv} and embedding files {*.z} in embeddings/digits5
# python run_pmds.py -m MAP2 --use_pre_config --normalize_dists --no_logging -d digits5 --experiment_mode --exp_missing_pairs


# reproduce figure for other datasets with fixed points
# note: add --exp_re_run to re run iPMDS with fixed points
# result: plots/MAP2/fmnist/fmnist.png
# python run_pmds.py -m MAP2 --no_logging --normalize_dists --use_pre_config -d fmnist --experiment_mode --exp_with_fixed_points
# python run_pmds.py -m MAP2 --no_logging --normalize_dists --use_pre_config -d automobile --experiment_mode --exp_with_fixed_points

###
### Note to run plot for digits5 with missing data:
# 1. make sure the complete dataset is created
# in the config, make sure no `fixed_points` and no `missing_pairs`
# python run_pmds.py -m MAP2 --no_logging --normalize_dists --use_pre_config -d digits5
# 2. run iPMDS with fixed points, without `missing_pairs`
#  python run_pmds.py -m MAP2 --no_logging --normalize_dists --use_pre_config -d digits5 --experiment_mode --exp_with_fixed_point --exp_re_run
# 3. now enable `missing_pairs` and repeat step 2 run iPMDS with fixed points with `missing_pairs`
# this will re-run PMDS without fixed points and with fixed points, both in missing data setting
#  python run_pmds.py -m MAP2 --no_logging --normalize_dists --use_pre_config -d digits5 --experiment_mode --exp_with_fixed_point --exp_re_run
# result: result: plots/MAP2/digits5/{digits5.png -> no missing; digits5_30.png -> missing 30% pairs}