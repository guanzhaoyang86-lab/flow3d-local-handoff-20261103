"""Read the versioned, physical-unit FS-SPH compact particle snapshots."""
from pathlib import Path
import struct

import numpy as np

HEADER = struct.Struct("<8sQQd")
MAGIC = b"FSSPH001"
BYTES_PER_PARTICLE = 43


def read_compact(path):
    """Return points, particle fields and timing, validating the exact schema."""
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError(f"Truncated snapshot header: {path}")
        magic, step, count, time_s = HEADER.unpack(header)
        if magic != MAGIC or count < 1 or not np.isfinite(time_s) or time_s < 0:
            raise ValueError(f"Invalid snapshot header: {path}")
        if path.stat().st_size != HEADER.size + BYTES_PER_PARTICLE * count:
            raise ValueError(f"Snapshot size does not match its particle count: {path}")

        def column(dtype, components=1):
            values = np.fromfile(stream, dtype=dtype, count=count * components)
            if components > 1:
                values = values.reshape(count, components)
            return values

        ids = column("<i4")
        points = column("<f4", 3)
        momentum = column("<f4", 3)
        transport = column("<f4", 3)
        fixed = column("u1")
        jointed = column("u1")
        valid = column("u1")
    if np.unique(ids).size != count:
        raise ValueError(f"Duplicate particle IDs: {path}")
    if not all(np.isfinite(a).all() for a in (points, momentum, transport)):
        raise ValueError(f"Nonfinite compact particle data: {path}")
    if not np.isin(valid, (0, 1)).all():
        raise ValueError(f"Invalid transport validity mask: {path}")
    return {
        "points": points,
        "fields": {"unique_idx": ids, "velocity": momentum,
                   "transport_velocity": transport, "Fixed": fixed,
                   "Jointed": jointed, "transport_velocity_valid": valid},
        "step": int(step), "physical_time_s": float(time_s),
        "coordinate_units": "metres", "velocity_units": "metres/second",
        "velocity_scale_factor": 1.0,
    }
