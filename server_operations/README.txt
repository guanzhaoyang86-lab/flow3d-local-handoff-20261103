Server source and task evidence; refreshed 2026-11-04.
JHTDB and FS-SPH retain server_own_models_20261103.
LBM uses lbm_split_fix_20261104; its original adapter/model packages are historical.
The completed release has 2499 training, 313 validation and 313 test cases.
No raw split, archive, model architecture or seed was changed.
Do not rewrite frozen paths or run server Slurm entry points on the laptop.
Local baselines use local_support/ and results/work_assignment.json.
Large datasets and checkpoints remain outside Git. GPU success requires actual execution.
