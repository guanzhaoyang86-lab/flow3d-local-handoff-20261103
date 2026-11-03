LBM DPS, seed31, 100epochs, batch1, N12 XZ/view0, observationseed47.

Requires the complete 3125-case/500000-trajectory dataset and verified adapter:
/work/hdd/biup/zguan2/datasets/experiment_adapters_20261103/lbm_frozen32
Never trains on an incomplete or partial collection. The training submission must
depend on successful adapter completion, which itself depends on generation3515292.

Manifest is adopted at runtime only after COMPLETE verifies the actual manifest
checksum, reviewed adapter-code checksum, all3125 fields,500000physical trajectories,
and arrays_modified=false. Code/config have a submission-time frozen fingerprint.
The adopted manifest checksum is then fixed in prior checkpoints and all sampling
provenance; validation/test refuse any different manifest/checkpoint.

Units remain lattice cells/LBM steps. Boundary_projection=none matches the other
two models and the original DPS decode; only public solid geometry enters the
existing training/evaluation masks. No per-case lid velocity or other true physical
case parameters are supplied as inference conditions. No GT velocity-field values
or withheld probe targets guide posterior sampling.

Train→validation calibration→test are separate stages with internal afterok
dependencies. All313 test cases are evaluated only after first8validation cases
select sigma from [.02,.05,.1]. DDIM50,2samples,seed47, RK4/4substeps are fixed.
Resource caps8h+8h+16h are requests, not measured durations. Resumption verifies
existing predictions/checkpoints and preserves committed training history.
