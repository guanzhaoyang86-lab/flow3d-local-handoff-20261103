LBM release split correction, 2026-11-04

The completed original generator release contains 3125 flow cases and 500000
trajectories, split into 2499 training, 313 validation and 313 test cases.
Its original split algorithm reserves one example per split before distributing
the remaining examples by largest remainder. The previous LBM consumers had
incorrectly reused JHTDB's 2500/312/313 expectation.

This revision preserves every raw archive and the exact raw manifest. It updates
the adapter, DiT3D, Tensor-DiT and DPS consumers to the completed release and binds
them to its hashes. Incorrect counts, source versions and adapter versions still
fail validation. Training remains seed31; the observation protocol and model
architectures are unchanged. No regeneration or repartition is performed.

The original frozen packages remain in server_own_models_20261103 as historical
evidence. This directory contains replacement frozen packages, regression checks,
and actual submission records. JHTDB and FS-SPH jobs retain their existing code
and queue positions.

LBM order: verified adapter -> DiT3D -> DPS training/calibration/test -> Tensor-DiT.
All models require successful adaptation. Independent model stages use afterany;
DPS calibration and test require their upstream stage to succeed.
The completed generation job has left Slurm's active controller, so the replacement
adapter uses the verified COMPLETE/manifest/checksum release gates instead of an
expired afterok job reference. Generation success is also checked through sacct
before replacement submission.

See repair.json, replacement_jobs.json, submissions.json and final_review.json
for actual state and fingerprints. CPU checks and submission preflight do not
prove real GPU training success. DiT3D and Tensor-DiT held-out prediction evaluation
remains a subsequent stage, as in the existing execution policy.
