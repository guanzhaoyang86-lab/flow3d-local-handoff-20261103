LBM complete-release adapter revision, 2026-11-04.

The completed raw release has train=2499, validation=313, test=313.
The original builder first reserves one group per split, then allocates the
remaining 3122 groups by largest remainders at 0.8/0.1/0.1. The previous
2500/312/313 gate assumed a different rounding rule and rejected valid data.

This revision accepts the original complete release only, pinning its checksum
list and raw manifest SHA256. It preserves every case's original split, all
NPZ bytes, exterior solid masks, and native lattice units. No data generation,
repartitioning, or per-case lid velocity inference condition is introduced.

The original frozen package remains untouched. This directory contains new
source and new verification reports only; prior execution evidence was not
copied. Full archive checks and byte-preserving adaptation run in Slurm, never
as a login-node test. One GPU is reserved because this partition rejects CPU-only
allocations; the checksum/copy program itself computes on CPU.
