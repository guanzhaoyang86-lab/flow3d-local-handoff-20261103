#!/usr/bin/env python3
"""Read FS particle snapshots and reconstruct physical velocity on XYZ grids.

Positions are metres. VTK velocities are solver velocities: divide by the run's
``velocity_scale_factor`` (10 for trial 3429148) to obtain metres/physical-second.
The grid array convention is [component XYZ, depth Z, height Y, width X].
Material and interpolation-extension masks are deliberately distinct: empty
space does not become a stationary wall merely because it has no particles.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


_HEADERS = re.compile(
    r"(?m)^[ \t]*(POINTS|POINT_DATA|CELL_DATA|SCALARS|VECTORS|NORMALS|TENSORS|"
    r"COLOR_SCALARS|FIELD|LOOKUP_TABLE|CELLS|CELL_TYPES|VERTICES|LINES|POLYGONS|"
    r"TRIANGLE_STRIPS)\b[^\r\n]*(?:\r?\n|$)"
)
_INTEGER_TYPES = {"bit", "char", "unsigned_char", "short", "unsigned_short", "int",
                  "unsigned_int", "long", "unsigned_long", "long_long", "unsigned_long_long"}


def read_vtk(path: str | Path, wanted_fields: Iterable[str] | None = None) -> dict:
    """Read legacy ASCII VTK points and scalar/vector point fields.

    Numeric tuples may span arbitrary lines. Integer fields, notably unique_idx,
    remain int64. Extra scalar/vector arrays are returned unless ``wanted_fields``
    restricts them. Cell data are never mistaken for particle data. Missing
    requested fields, malformed tuple counts and binary VTK fail explicitly.
    """
    path = Path(path)
    text = path.read_text(encoding="ascii")
    lines = text.splitlines()[:4]
    if len(lines) < 3 or lines[2].strip() != "ASCII":
        raise ValueError(f"Expected legacy ASCII VTK: {path}")
    headers = list(_HEADERS.finditer(text))
    wanted = None if wanted_fields is None else set(wanted_fields)
    points, fields, point_count, in_point_data = None, {}, None, False

    def numbers(start: int, stop: int, count: int, dtype, name: str):
        # Restrict to the numeric section so fromstring cannot silently stop at
        # the following header and hide an incorrect tuple count.
        values = np.fromstring(text[start:stop], dtype=dtype, sep=" ")
        if values.size != count:
            raise ValueError(f"{path}: {name} has {values.size} values; expected {count}")
        return values

    for index, header in enumerate(headers):
        parts = header.group().split()
        kind = parts[0]
        stop = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        if kind == "POINTS":
            if points is not None:
                raise ValueError(f"Multiple POINTS sections: {path}")
            point_count = int(parts[1])
            if point_count < 1:
                raise ValueError(f"No particle points: {path}")
            points = numbers(header.end(), stop, point_count * 3, np.float64, "POINTS").reshape(-1, 3)
        elif kind == "POINT_DATA":
            if point_count is None or int(parts[1]) != point_count:
                raise ValueError(f"POINT_DATA does not match POINTS: {path}")
            in_point_data = True
        elif kind == "CELL_DATA":
            in_point_data = False
        elif in_point_data and kind in {"SCALARS", "VECTORS", "NORMALS", "TENSORS"}:
            name = parts[1]
            if wanted is not None and name not in wanted:
                continue
            if name in fields:
                raise ValueError(f"Duplicate particle field {name}: {path}")
            components = int(parts[3]) if kind == "SCALARS" and len(parts) > 3 else 1
            if kind in {"VECTORS", "NORMALS"}:
                components = 3
            elif kind == "TENSORS":
                components = 9
            start = header.end()
            if kind == "SCALARS":
                if index + 1 >= len(headers) or headers[index + 1].group().split()[0] != "LOOKUP_TABLE":
                    raise ValueError(f"Missing LOOKUP_TABLE for {name}: {path}")
                start = headers[index + 1].end()
                stop = headers[index + 2].start() if index + 2 < len(headers) else len(text)
            dtype = np.int64 if parts[2] in _INTEGER_TYPES else np.float64
            values = numbers(start, stop, point_count * components, dtype, name)
            fields[name] = values if components == 1 else values.reshape(point_count, components)
        elif in_point_data and kind in {"FIELD", "COLOR_SCALARS"}:
            raise ValueError(f"Unsupported particle array section {kind}: {path}")
    if points is None or not np.isfinite(points).all():
        raise ValueError(f"Missing or nonfinite particle coordinates: {path}")
    if wanted is not None and not wanted.issubset(fields):
        raise ValueError(f"Missing point fields {sorted(wanted - fields.keys())}: {path}")
    if "unique_idx" in fields and np.unique(fields["unique_idx"]).size != len(points):
        raise ValueError(f"Duplicate particle identifiers: {path}")
    return {"points": points, "fields": fields}


def interpolate_particle_velocity(
    points, velocities, query_points, spacing: float, *,
    velocity_scale_factor: float = 10.0, k: int = 32, chunk_size: int = 2048,
    material_radius_factor: float = 0.9, support_radius_factor: float = 3.0,
    particle_ids=None, excluded_ids=None, fixed=None, jointed=None, groups=None,
    workers: int = 1, gradient_ridge: float = 1e-10,
) -> dict:
    """Chunked weighted local-affine velocity reconstruction at XYZ queries.

    The intercept gives velocity at the query. Coordinates are centered and
    scaled by local neighbour radius; only gradients receive a tiny ridge.
    This reproduces constants and well-conditioned affine fields to numerical
    precision, while regularizing planar or coincident donor configurations.

    Excluded IDs are removed *before* constructing the neighbour tree; held-out
    particles cannot leak into either fits or nearest-particle masks. The
    support mask means distance to a donor <= support_radius_factor*spacing;
    it is a reconstruction criterion, not a declaration of occupied material.
    Unsupported output velocities are NaN. Nearest labels are raw solver labels
    and must be interpreted by the caller. Donor order is preserved by indices.
    """
    points = np.asarray(points, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or velocities.shape != points.shape:
        raise ValueError("points and velocities must both have shape [N,3]")
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape [M,3]")
    if not (np.isfinite(points).all() and np.isfinite(velocities).all()
            and np.isfinite(query_points).all()):
        raise ValueError("points, velocities and queries must be finite")
    if not (np.isfinite(spacing) and spacing > 0 and
            np.isfinite(velocity_scale_factor) and velocity_scale_factor > 0):
        raise ValueError("spacing and velocity_scale_factor must be positive")
    if (k < 4 or chunk_size < 1 or gradient_ridge <= 0 or
            material_radius_factor <= 0 or support_radius_factor < material_radius_factor):
        raise ValueError("Invalid neighbor, chunk, ridge or radius setting")
    labels = {name: np.asarray(value) for name, value in
              (("id", particle_ids), ("fixed", fixed), ("jointed", jointed), ("group", groups))
              if value is not None}
    if any(value.ndim != 1 or len(value) != len(points) for value in labels.values()):
        raise ValueError("Particle identifiers and labels must have shape [N]")
    donors = np.arange(len(points))
    if excluded_ids is not None:
        if particle_ids is None:
            raise ValueError("excluded_ids requires particle_ids")
        donors = donors[~np.isin(labels["id"], np.asarray(list(excluded_ids)))]
    if len(donors) < 4:
        raise ValueError("At least four donor particles are required after exclusions")
    source_points = points[donors]
    source_velocity = velocities[donors] / velocity_scale_factor
    tree = cKDTree(source_points)
    count, neighbor_count = len(query_points), min(int(k), len(donors))
    result_velocity = np.full((count, 3), np.nan, dtype=np.float64)
    nearest_distance = np.empty(count, dtype=np.float64)
    nearest_index = np.empty(count, dtype=np.int64)
    for start in range(0, count, int(chunk_size)):
        end = min(start + int(chunk_size), count)
        queries = query_points[start:end]
        distance, indices = tree.query(queries, k=neighbor_count, workers=workers)
        nearest_distance[start:end] = distance[:, 0]
        nearest_index[start:end] = donors[indices[:, 0]]
        supported = distance[:, 0] <= support_radius_factor * spacing
        if not supported.any():
            continue
        local_distance = distance[supported]
        local_indices = indices[supported]
        radius = np.maximum(local_distance[:, -1], spacing)
        offset = (source_points[local_indices] - queries[supported, None, :]) / radius[:, None, None]
        design = np.concatenate((np.ones((*offset.shape[:-1], 1)), offset), axis=-1)
        weight = np.exp(-2.0 * (local_distance / radius[:, None]) ** 2)
        gram = np.einsum("mki,mk,mkj->mij", design, weight, design, optimize=True)
        rhs = np.einsum("mki,mk,mkc->mic", design, weight, source_velocity[local_indices], optimize=True)
        ridge = gradient_ridge * np.maximum(np.trace(gram[:, 1:, 1:], axis1=1, axis2=2) / 3.0, 1.0)
        for diagonal in range(1, 4):
            gram[:, diagonal, diagonal] += ridge
        coefficient = np.linalg.solve(gram, rhs)
        out_indices = np.arange(start, end)[supported]
        result_velocity[out_indices] = coefficient[:, 0]
    material_mask = nearest_distance <= material_radius_factor * spacing
    support_mask = nearest_distance <= support_radius_factor * spacing
    result = {
        "velocity": result_velocity,
        "material_mask": material_mask,
        "support_mask": support_mask,
        "extension_mask": support_mask & ~material_mask,
        "nearest_distance": nearest_distance,
        "nearest_index": nearest_index,
        "donor_count": int(len(donors)),
        "velocity_scale_factor": float(velocity_scale_factor),
    }
    for name, value in labels.items():
        result[f"nearest_{name}"] = value[nearest_index]
    return result


def grid_particle_velocity(
    points, velocities, bounds, grid_size=32, spacing: float = 0.00075, **kwargs,
) -> dict:
    """Reconstruct a node-centered grid, inclusive of physical XYZ bounds.

    ``bounds`` is [XYZ, min/max], shape [3,2]. ``grid_size`` is either a scalar
    or (D,H,W), including for noncubic grids. Returned ``flow_field`` is float32
    [XYZ,D,H,W]. Raw nearest labels and all masks are [D,H,W]. ``grid_points`` is
    [D,H,W,XYZ]; ``grid_axes`` contains separate x/y/z coordinate vectors.
    See :func:`interpolate_particle_velocity` for donor selection and support.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds must be finite XYZ [3,2] with max > min")
    if np.isscalar(grid_size):
        shape = (int(grid_size),) * 3
    else:
        shape = tuple(int(size) for size in grid_size)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("grid_size must be >=2 or a (D,H,W) tuple with each >=2")
    depth, height, width = shape
    x = np.linspace(*bounds[0], width)
    y = np.linspace(*bounds[1], height)
    z = np.linspace(*bounds[2], depth)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    queries = np.stack((xx, yy, zz), axis=-1)
    fit = interpolate_particle_velocity(points, velocities, queries.reshape(-1, 3), spacing, **kwargs)
    velocity = fit.pop("velocity").reshape(*shape, 3)
    output = {
        "flow_field": np.moveaxis(velocity, -1, 0).astype(np.float32),
        "grid_points": queries,
        "grid_axes": {"x": x, "y": y, "z": z},
        "domain_bounds": bounds,
    }
    for name, value in fit.items():
        output[name] = value.reshape(shape) if isinstance(value, np.ndarray) else value
    return output


def _self_test():
    """Small analytic checks independent of simulation output; run this file."""
    import tempfile

    rng = np.random.default_rng(87)
    points = rng.uniform(-1, 1, (600, 3))
    queries = rng.uniform(-0.8, 0.8, (50, 3))
    constant = np.broadcast_to([2.0, -3.0, 4.0], points.shape)
    fit = interpolate_particle_velocity(points, constant * 10, queries, spacing=0.5)
    np.testing.assert_allclose(fit["velocity"], constant[:len(queries)], atol=1e-11)
    matrix = np.array([[2.0, -1.0, 0.5], [0.2, 1.0, 3.0], [-0.5, 0.1, 1.0]])
    velocity = points @ matrix.T + [0.7, -0.3, 0.2]
    fit = interpolate_particle_velocity(points, velocity * 10, queries, spacing=0.5)
    np.testing.assert_allclose(fit["velocity"], queries @ matrix.T + [0.7, -0.3, 0.2], atol=1e-8)
    grid = grid_particle_velocity(points, velocity * 10, [[-0.5, 0.5]] * 3,
                                  grid_size=(3, 4, 5), spacing=0.5)
    assert grid["flow_field"].shape == (3, 3, 4, 5)
    np.testing.assert_allclose(grid["grid_points"][2, 1, 3], [0.25, -1 / 6, 0.5])
    expected = grid["grid_points"] @ matrix.T + [0.7, -0.3, 0.2]
    np.testing.assert_allclose(np.moveaxis(grid["flow_field"], 0, -1), expected, atol=2e-7)
    far_fit = interpolate_particle_velocity(points, velocity * 10, [[9, 9, 9]], spacing=0.5)
    assert not far_fit["support_mask"].any() and np.isnan(far_fit["velocity"]).all()
    # Excluding a deliberately poisoned particle must eliminate it from both fit and labels.
    poisoned = np.concatenate(([[0, 0, 0]], points))
    poisoned_velocity = np.concatenate(([[1e9, 1e9, 1e9]], velocity * 10))
    fit = interpolate_particle_velocity(poisoned, poisoned_velocity, [[0, 0, 0]], spacing=0.5,
                                        particle_ids=np.arange(601), excluded_ids=[0],
                                        fixed=np.arange(601))
    assert fit["nearest_id"][0] != 0 and fit["nearest_fixed"][0] != 0
    assert fit["nearest_distance"][0] > 0
    np.testing.assert_allclose(fit["velocity"][0], [0.7, -0.3, 0.2], atol=1e-8)
    # Distinguish material occupancy from a local extension into nearby void.
    fit = interpolate_particle_velocity(points, velocity * 10, points[:1], spacing=0.5)
    assert fit["material_mask"][0] and not fit["extension_mask"][0]
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    extension = interpolate_particle_velocity(cube, cube * 10, [[1.5, 1, 1]], spacing=0.5)
    assert extension["support_mask"][0] and extension["extension_mask"][0]
    assert not extension["material_mask"][0]
    vtk = """# vtk DataFile Version 2.0
test
ASCII
DATASET POLYDATA
POINTS 2 float
0 0 0 1
2 3
POINT_DATA 2
SCALARS unique_idx int 1
LOOKUP_TABLE default
10 20
SCALARS Fixed float
LOOKUP_TABLE default
0 2
VECTORS velocity float
1 2 3 4 5 6
SCALARS Jointed float 1
LOOKUP_TABLE default
0 1
CELL_DATA 1
SCALARS Fixed float 1
LOOKUP_TABLE default
99
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "frame.vtk"
        path.write_text(vtk)
        data = read_vtk(path)
        np.testing.assert_array_equal(data["points"], [[0, 0, 0], [1, 2, 3]])
        np.testing.assert_array_equal(data["fields"]["unique_idx"], [10, 20])
        np.testing.assert_array_equal(data["fields"]["velocity"], [[1, 2, 3], [4, 5, 6]])
        assert set(read_vtk(path, {"Fixed"})["fields"]) == {"Fixed"}
    print("fs_grid analytic tests passed: affine/constant fields, XYZ-ZYX, scaling, masks, held-out IDs, ASCII VTK")


if __name__ == "__main__":
    _self_test()
