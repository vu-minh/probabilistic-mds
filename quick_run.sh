# python run_pmds.py -d digits012 -m MAP3 --use_pre_config --normalize_dists
# python run_pmds.py -d cities_us_toy -m MAP2 --use_pre_config --normalize_dists --no_logging

# commands to run experiment + plots for paper
# use the same command as usual with `--experiment_mode` and `--exp_xxx`
# by default it will remake the plots using stored embedding files.
# add `--exp_re_run` to re-run the exp.


# run experiment with missing pairs 20 times for 20 values of p
# python run_pmds.py -m MAP2 --use_pre_config --normalize_dists --no_logging -d digits5 --experiment_mode --exp_missing_pairs


# reproduce figure for automobile: 
# python run_pmds.py -m MAP2 --no_logging --normalize_dists --use_pre_config -d automobile --experiment_mode --exp_automobile
