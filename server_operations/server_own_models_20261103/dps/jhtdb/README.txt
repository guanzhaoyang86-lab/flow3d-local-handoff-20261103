JHTDB unconditional-prior DPS, prepared 2026-11-03

This independent package does not alter the old frozen experiments or submit jobs.
Seed31, 100 epochs, batch1; no particle tracks, test fields or validation fields enter
prior training or normalization. The prior sees train flow fields and fluid masks.
N12 singleXZ/view0, observation seed47, 32 disjoint probe particles; posterior uses
only projected tracks and hidden-depth inference. Probe and reference fields are
opened after posterior sampling for evaluation only.

Sampling protocol: 50 DDIM-DPS steps, eta1, 2 samples, seed47, RK4 with4substeps;
this is explicitly a shortened DDIM-DPS schedule, not the original1000step DDPM.
Sigma is selected by mean field-relativeL2 on validation cases0..7 from [.02,.05,.1].
The selected config and checkpoint hashes are frozen before all313 heldout cases.
The test stage requires the matching completed validation selection. No test tuning.

Submit train.sbatch, then calibrate.sbatch with afterok:TRAIN, then test.sbatch with
afterok:CALIBRATE. All submissions must export FLOW3D_DPS_FROZEN_SHA256 equal to
SHA256(frozen_manifest.json), fixed at submission, and use clean SBATCH environment.
Job names are own3_jhtdb_dps, own3_jhtdb_dps_calibration, own3_jhtdb_dps_evaluate.
Runtime checks frozen code/config, completed dataset manifest and a real allocated
CUDA tensor gradient. Preflight is CPU-only and does not claim CUDA was tested.

Resource requests are caps: train8h, validation8h, test16h, one GH200 GPU each.
These are not runtime measurements. Sampling progress resumes only after verifying
NPZ+JSON content hashes and exact config; partial training resumes the committed
checkpoint, preserving a backup before trimming uncommitted history rows. A rerun
never overwrites an orphan prediction. Weights/results go only to the new root:
/work/hdd/biup/zguan2/results/server_own_models_20261103/dps/jhtdb

The static reader preserves real solid masks and uses fluid-only statistics/loss.
Any future LBM config must explicitly label LBM/lattice units and depend on complete
dataset validation; this JHTDB package never silently substitutes a partial dataset.
