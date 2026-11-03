Server operation source snapshot, 2026-11-03.
The user assigns DiT3D, Tensor-DiT and Flow3D-DPS to the server, on both JHTDB and FS-SPH;
LBM is appended only after full generation and adapter verification.
This directory contains source/configuration/provenance and submission evidence.
Server absolute paths and Slurm guards are intentionally preserved. Do not launch these
server jobs on the laptop or globally replace paths in frozen files. Local baseline work
uses local_support/ and results/work_assignment.json. Large data/checkpoints stay outside Git.
Pending source adaptations are not completed or submitted jobs; inspect per-model reports
and the timestamped server_queue.json. No CPU check proves real GPU training succeeded.
LBM uses native lattice units, public solid geometry and boundary_projection=none;
true per-case lid velocity is not supplied as an inference condition.
