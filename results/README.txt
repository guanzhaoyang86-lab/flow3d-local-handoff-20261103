Flow3D completed-result metadata export — 2026-11-03

Start with summary.json. results_inventory.json gives source paths, dataset scope,
completion records, exported-file SHA256 values and the checkpoint inventory.
current_training_objectives.csv contains training/validation objectives only.
These values are NOT final held-out test reconstruction metrics. Different
models can optimize different objectives; their losses are not a ranking table.

Current new-data baselines:
- FS-SPH: one physical material case, chronological window holdout; 126 train,
  18 validation, 18 test windows, with temporal gaps. This is not cross-case
  generalization. Training uses time-conditioned frames and valid support.
- JHTDB: 2500 train, 312 validation, 313 test spatial crops from one frozen DNS
  snapshot. Generated trajectories do not establish actual DNS time prediction.
- Current training policy is seed31. Previously finished seed32/33 CNN/DeepONet
  artifacts are retained as historical extras, not new authorized repetitions.
- Mean/POD epoch0 means a completed statistical fit, not zero completed work.
- POD/RBF calibration uses validation. Missing CALIBRATION_COMPLETE.json means
  calibration is not yet complete. Pilot smoke outputs are separate and are not
  formal 100-epoch test results.

Legacy results are under legacy_conditional/ and legacy_comparisons/ and retain
their own data/protocol provenance. In particular the old repaired-model N2
comparison uses 44 validation recipients and matched donors on the old LBM
dataset. It must not be presented as new JHTDB/FS-SPH/LBM500k test performance.
Legacy batch completion comes from actual results.json plus its verified audit,
not the stale submission-time progress snapshot. Reused N2 training is checked
against its actual completed 100-epoch training_summary.json.

This repository handoff contains exported metrics and training histories.
Weights and data are indexed only; their binary payloads are not uploaded here.
checkpoint_files.csv lists every inventoried
checkpoint, its server path and byte count, including duplicate versions such
as best/latest/epoch files. Recorded SHA256 values come from existing completion
records; large weight bytes were not rehashed. Some individual weights exceed
100 MiB. Select a separate appropriate weight-transfer mechanism after reviewing
the size inventory. Dataset/provenance JSON does not include NPZ data arrays.

server_queue.json is a point-in-time queue snapshot maintained by the main
handoff process; jobs continue on the server. The exporter never queries,
submits, cancels, modifies production artifacts or starts training.

To rebuild these metadata snapshots on the server (Python 3.10+):
  /sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B export_results.py

The script reads fixed source paths and writes only beside itself. It copies
JSON/JSONL/CSV/TXT files of at most 16 MiB each. Rebuilding the export refreshes
the snapshot; it does not restart any experiment. Current source code packages
are supplied separately by the repository handoff.

Optional asset pulls from the receiving machine:
asset_manifest.json recommends one latest.pt per completed run and lists the
three adapted dataset directories. Weight destinations match their results/
metadata directories and retain the original latest.pt filename. The destination
root is handoff_20261103 itself, not the parent Git repository.

List asset IDs without connecting:
  python results/transfer_assets.py --list

Print a planned pull without connecting or creating directories:
  python results/transfer_assets.py --asset checkpoint-fs_sph-cnn_s31 --host deltaai --destination-root /path/to/handoff_20261103

Add --execute to run that pull. Repeat --asset for multiple IDs. For datasets,
use data-fs_sph, data-jhtdb or data-lbm. On Windows, run this transfer helper in
an environment such as WSL that provides rsync and the configured SSH alias.
No deletion option is supported. LBM is still being generated: an early pull
is a partial snapshot and does not replace the final COMPLETE.json check.
