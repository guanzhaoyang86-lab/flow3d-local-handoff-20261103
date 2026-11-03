JHTDB N12 single-seed pilot recovery, 2026-10-30 America/Chicago

This isolated revision repairs smoke config/result filename collisions for POD and RBF.
Config: pilot/smoke_configs/{model}.json; prediction: pilot/smoke_predictions/{model}.npz.
Existing pilot checkpoints and result_root are reused after provenance checks.
Only seed31 is submitted. The recovery plan contains 10 jobs and excludes completed preparation.
Original FS-SPH queued jobs keep their original frozen runtime and inputs.
plan.json is the prepared recovery plan; build_plan in the helper describes a fresh two-dataset batch.
