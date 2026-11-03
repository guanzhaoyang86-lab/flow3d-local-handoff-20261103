FS-SPH DPS uses an unconditional shared frame prior and the full 20-frame native-time
trajectory likelihood. It is a factorized frame prior, with frames coupled during
posterior reconstruction by physical particle advection; it does not learn a joint
temporal prior. The prior receives no particle tracks, masks or velocity support.

The 126 training windows supply flow-only normalization. Each training epoch draws
one native-time-weighted frame per training window, with supported-cell score loss
and independently normalized Gaussian nuisance score outside velocity support.
Validation uses all 20 frames, with native trapezoidal time weights. Train seed31,
100 epochs and batch1 are fixed. Ground-truth support is used only by training loss
and posthoc evaluation. Inference receives projected tracks, their observation mask,
native observation times, camera, bounds and published units; no reference fields,
3D particle tracks, withheld probes or ground-truth velocity support guide sampling.

Posterior state is [20,3,32,32,32]. Native uneven velocity times enter interpolation
and RK4 particle integration, while initial hidden Y depths are inferred jointly.
The trajectory likelihood and global gradient RMS clipping span all frames. The
shared denoiser acts independently on each frame. DDIM50, eta1, 2 samples, seed47,
N12 XZ/view0, observationseed47, and RK4 with4substeps are fixed. This is explicitly
a shortened DDIM-DPS schedule rather than a full1000step DDPM posterior sampler.

Stages are full100epoch train, validation-only sigma calibration on first8 validation
windows from [.02,.05,.1], then evaluation of all18 heldout test windows. Validation
freezes the selected sigma, manifest hash, prior checkpoint hash and sampling config
before test. No test cases tune sigma. Posthoc native-time physical field and probe
metrics are computed only after posterior samples are fixed. Failed sample validity
and finite-metric coverage must remain visible in the results.

Submit train.slurm, calibrate.slurm afterok:TRAIN, then test.slurm afterok:CALIBRATE.
Export FLOW3D_DPS_FS_FROZEN_SHA256 equal to the SHA256 of frozen_manifest.json at
submission; runtime rejects changed frozen source/config/data. Resource caps are
8h training,8h validation and16h testing on one GH200; these are not measured times.
CPU synthetic checks and real-manifest preflight precede submission. Real CUDA
arithmetic and CUDA gradient checks occur only within the allocated Slurm job.

This directory and ../fs_source are independent of already submitted static DPS;
do not rerun source-generation helpers or edit frozen files while jobs are queued.
