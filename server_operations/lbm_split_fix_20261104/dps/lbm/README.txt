LBM DPS split-fix revision, 2026-11-04

Preserve the published independent-flow split and row order exactly:
train2499, validation313, test313. No cases are reassigned. Raw split seed20260918,
observationseed47 and trainingseed31 have distinct roles. The release establishes
the dataset contract, not physical convergence or final reconstruction performance.
The config preserves the raw release scientific scope.

Unconditional DiT3D prior source is copied byte-for-byte from the previous package.
Seed31,100epochs,batch1,train-only fluid normalization/loss remain unchanged.
N12 singleXZ/view0,32disjoint probes,observationseed47,DDIM50,2samples,seed47,RK4/4
substeps are fixed. Boundary_projection=none: no per-case lid velocity, Reynolds
value, reference field or withheld probes condition posterior sampling. SingleXZ
probe errors are explicitly null if initial depth is unidentifiable; no GT depth
is substituted. Field metrics remain available.

Every train/calibrate/test entry verifies frozen code and the raw COMPLETE/manifest/
SHA256SUMS hash chain before CUDA:3125 independent groups,500000physical/singleview
trajectories,exact2499/313/313 counts,original case/group membership and row order,
native lattice units,N12/XZ/seed47/seed31,repaired adapter SHA,case checksums,and
unchanged arrays/data/split certificates. The repaired prepare.py resides under
this operation root at tensor_dit/lbm_adapter/prepare.py; its SHA is pinned in config.
Adapted output remains:
/work/hdd/biup/zguan2/datasets/experiment_adapters_20261103/lbm_frozen32

The adapted manifest is adopted only after its complete reviewed certificate exists.
Before then, preflight verifies the actual completed raw release and explicitly
reports waiting_for_completed_adapter; it does not claim adapted-data or GPU success.
Checkpoints and predictions bind the adopted manifest and reject changed-data resume.
Validation cases0..7 select sigma from[.02,.05,.1]; all313test cases run afterward.

Freeze only after CPU tests and documentation are final. Parent submits train.sbatch
after repaired adapter success,calibrate.sbatch afterok:TRAIN,test.sbatch afterok:CAL.
Export FLOW3D_DPS_FROZEN_SHA256=SHA256(frozen_manifest.json) at submission.
OneGH200/8CPU/64G requests have8h+8h+16h caps,not measured runtimes. Production checks
real CUDA tensor gradients only within Slurm. Parent records all actual job IDs.

New result root:
/work/hdd/biup/zguan2/results/lbm_split_fix_20261104/dps/lbm
Old frozen source,submission records and result roots remain untouched.
No old reports were copied. reports/source_copy_verification.json and
reports/raw_preflight.json contain new checks actually performed for this revision.
