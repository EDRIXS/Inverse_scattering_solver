import argparse, importlib.util

from optEDRIXS import run_bayesian_optimization, import_config

p = argparse.ArgumentParser(
    description="Run RIXS Bayesian optimization for a given config and run index."
)
p.add_argument("--config-py", required=True,
                   help="Path to the Python config file")
p.add_argument(
        "run_ind",
        type=int,
        nargs="?",           # allow it to be omitted
        default=1,           # use 1 if not provided
        help="Run index (default: %(default)s)"
    )
args = p.parse_args()

cfg = import_config(args.config_py)

# for all ions with occupation n_occu in the database:

record = cfg.record
record['run_ind'] = args.run_ind
run_bayesian_optimization(record, cfg.fun, output_dir = cfg.output_dir)