LBM DiT3D split correction, 2026-11-04

The completed raw release has train2499 / validation313 / test313, totaling
3125 independent flow groups and500000 trajectories. The earlier2500/312/313
assumption was incorrect. This revision accepts the finalized release and
continues rejecting incorrect or mutually inconsistent counts. No source case
is moved between splits and no original field/trajectory array is changed.

Original frozen model:
 /u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/dit3d
Manifest SHA:
 400e72abd126bbd48555e3fbad55967191795a036c3ac9bf19c615dabe101084
The original model and previous dit3d_lbm package remain unchanged. Historical
records copied into history/ describe the old operation, not this submission.

N12, XZ/view0, fixed32 probes, observation seed47, train seed31, batch1 and total
100epochs are unchanged. boundary_projection=none; hidden lid velocity remains
excluded. Native lattice units and real exterior solid geometry are preserved.
The model is a full-volume DiT, trained on frozen cavity snapshots; no claim of
validated turbulence or SI-unit conversion is made.

The shared output remains:
 /work/hdd/biup/zguan2/datasets/experiment_adapters_20261103/lbm_frozen32
adapter_binding.json pins the revised adapter source and operation manifest.
The wrapper verifies those files and requires COMPLETE.adapter_sha256 to match.
Actual training requires successful completion of the revised adapter job,
matching published hashes, final split counts and all3125 adapted field files.
Source preflight may report waiting_for_complete_lbm_adapter before publication;
that state is not a successful full-data or GPU check.

New output:
 /work/hdd/biup/zguan2/results/lbm_split_fix_20261104/dit3d/lbm_s31
Slurm paths point to this new revision directory. Submit only after the parent
reviews the frozen manifest and assigns the revised adapter dependency.
This package does not itself submit or cancel jobs. The inherited controller
checks one complete formal epoch then resumes to total100 without repeating
completed epochs. CUDA capability/gradient checks occur in the actual allocation.

tests/test_final_release_counts.py checks the real COMPLETE, manifest and
checksum list without opening field NPZs. It also verifies that the old wrong
counts remain rejected even when synthetic markers agree on those wrong counts.
CPU mock/fixture tests cover restart, failure propagation and release gates.
See revision.json, adapter_binding.json and validation.json for exact evidence.
