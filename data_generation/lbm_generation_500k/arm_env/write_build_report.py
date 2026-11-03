"""Record an installed ARM Taichi build without initializing any GPU."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

from check_build import file_sha256, runtime_metadata, source_metadata


def main():
    root = Path(__file__).resolve().parent
    wheel = Path(sys.argv[1]).resolve()
    result = dict(schema_version=1, status="compiled_and_installed",
                  compiled_at_utc=datetime.now(timezone.utc).isoformat(),
                  **source_metadata(root), **runtime_metadata(),
                  wheel=str(wheel), wheel_sha256=file_sha256(wheel),
                  build_slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  cuda_kernel_validated=False, lbm32_pilot_validated=False)
    path=root/"build_report.json"
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result,indent=2)+"\n")
    temporary.replace(path)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
