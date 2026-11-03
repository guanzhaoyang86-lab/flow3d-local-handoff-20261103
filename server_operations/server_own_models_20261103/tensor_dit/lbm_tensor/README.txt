Tensor-DiT / LBM 500k / seed 31 adaptation

This independent package extends the frozen N12 Tensor implementation for the
complete LBM cavity release. The parent JHTDB/FS package and submitted jobs must
remain unchanged. Package preparation is not evidence of completed GPU training.

Required input:
  /work/hdd/biup/zguan2/datasets/experiment_adapters_20261103/lbm_frozen32
  COMPLETE.status=completed, verified_fields=3125, 500000 trajectories, matching
  manifest_sha256 and raw-finalization/checksum provenance.
  The shared adapter is job 3523604, afterok generation/validation job 3515292.

Protocol:
  Explicit dataset=lbm; N12, XZ only, observation seed47, training seed31.
  Preserve real exterior solid_mask and lattice cells per LBM step units.
  Never condition on the per-case true lid velocity, hidden Y or reference field.
  Boundary projection is none. Geometry may be public but moving-lid parameters
  are not privileged conditioning information for this method.
  Tucker fits train only with supported-fluid normalization. Excluded solid
  voxels use a train-only mean proxy; fluid-supported physical auxiliary has
  weight0.1. Validation selects rank4/8/12/16 with mean relativeL2<=0.05 and
  worst<=0.20. Failure stops before pilot or formal training.
  One GPU allocation runs preparation, full-path validation pilot, then 100epochs.
  The pilot exercises 8train/2validation and one validation DDIM prediction.
  Formal checkpoints cannot reuse a restricted pilot checkpoint.
  This workflow trains/validates only; pilot scores are not final test results.

LBM probe scoring must use the single-XZ observation_mask: raw archives have no
observation_mask_3d. Probe XYZ is read only after predictions have been fixed.
These are frozen lid-driven cavity snapshots, not validated turbulence.
