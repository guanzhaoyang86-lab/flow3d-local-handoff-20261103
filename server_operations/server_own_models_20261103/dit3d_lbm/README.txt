LBM full-volume DiT3D, seed31, 2026-11-03

This wrapper explicitly binds dataset=lbm and dataset_id=lbm32_xz_500k_20261103.
The scientific scope is frozen cavity snapshots, not validated turbulence.
Native lattice cells per LBM step and real exterior solid_mask are preserved.
N12, XZ/view0, observation seed47, fixed32 probes, batch1, total100epochs.
boundary_projection=none; hidden per-case lid velocity is not supplied to
the encoder or hard boundary projection. Only observed tracks/masks/projection/
times/bounds condition the denoiser. No conversion to SI units is claimed.

The immutable model source remains ../dit3d, manifest SHA256:
400e72abd126bbd48555e3fbad55967191795a036c3ac9bf19c615dabe101084
The wrapper reuses its GPU checks, allocation lock, checkpoint validation and
one formal pilot epoch followed by resume to total100epochs. Runtime path and
static-entry rebinding occurs only in an isolated module instance; frozen files
are never rewritten. Failure stops continuation. Completed epochs are not
repeated, including interrupted runs with a valid latest.pt.

The shared adapter job3523604 depends afterok3515292 and verifies all3125 raw and
copied NPZ hashes and500000 trajectories before publishing its COMPLETE marker.
This wrapper checks completion/manifest/source-version bindings, scientific
2500/312/313 disjoint flow-group splits, native units/protocol and file presence.
It does not repeat the full archive-byte checksum audit at every model start.
Before adapter completion, --preflight-only reports waiting, not data success.

Run preflight with the server PyTorch Python:
 /sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B run_lbm.py --preflight-only
Submission must export DIT_LBM_FROZEN_SHA256 equal to frozen_manifest.json SHA.
The submitted DiT job3523879 requires afterok3523604. The final LBM model order
is DiT3D, DPS, then Tensor-DiT; see ../execution_policy.json and submission.json.
No submission is performed by these files. lbm.slurm requests1GH200/8CPU/64G/24h.
GPU checks and real training are deferred to the allocation; CPU tests are mocked
or synthetic. Output is:
 /work/hdd/biup/zguan2/results/server_own_models_20261103/dit3d/lbm_s31
See validation.json and cpu_binding_checks.xml for checked scope and limitations.
