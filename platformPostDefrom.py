import csv
import json
from pathlib import Path
import warnings

import itasca as it
import numpy as np


# ============================================================
# User inputs -- relative paths use FLAC3D's current directory
# ============================================================
BEFORE_SAVE = r"[4]_.sav"
AFTER_SAVE = r"[5]DynamicRun3D_T_Via4_5GX20MPa.sav"

OUTFILE_BEFORE = r"platformShellBefore.vtk"
OUTFILE_AFTER = r"platformShellAfterPuredef.vtk"
OUTFILE_BOUNDARIES = r"platformShellBoundaryGridpoints.csv"
OUTFILE_SUMMARY = r"platformShellDriftSummary.json"

TARGET_SHELL_IDS = {703}  # Shell structural IDs, not unique component IDs
SCALE = 20.0
REMOVE_X = True
REMOVE_Y = True
REMOVE_Z = False

DISPLACEMENT_SOURCE = "position"  # change to "displacement" for small strain

# Explicit absolute tolerance, or None to calculate automatically from height.
# Effective tolerance includes a floating-point roundoff floor; no np.isclose rtol.
Z_TOL_ABS = None                 # e.g. 1.0e-4 in model length units
Z_TOL_REL = 1.0e-6               # automatic: height * Z_TOL_REL
Z_TOL_MIN = 1.0e-8               # automatic absolute minimum


# ============================================================
# Snapshot helpers: never return live Itasca object references
# ============================================================
def vnp(vector):
    return np.array([vector[0], vector[1], vector[2]], dtype=float)


def restore_model(filename):
    # Forward slashes also work for Windows paths in Itasca commands.
    filename = str(filename).replace("\\", "/")
    if any(char in filename for char in ('"', "\n", "\r")):
        raise ValueError("Save filenames cannot contain quotes or line breaks.")
    it.command('model restore "{}"'.format(filename))


def read_all_gridpoints():
    ids, xyz, disp = [], [], []
    for gp in it.gridpoint.list():
        ids.append(int(gp.id()))
        xyz.append(vnp(gp.pos()))
        disp.append(vnp(gp.disp()))
    if not ids:
        raise RuntimeError("The restored model has no gridpoints.")
    ids = np.asarray(ids, dtype=np.int64)
    order = np.argsort(ids)
    state = {
        "ids": ids[order],
        "xyz": np.asarray(xyz, dtype=float)[order],
        "disp": np.asarray(disp, dtype=float)[order],
    }
    if np.any(np.diff(state["ids"]) == 0):
        raise RuntimeError("Duplicate gridpoint IDs were encountered.")
    if not (np.isfinite(state["xyz"]).all() and np.isfinite(state["disp"]).all()):
        raise RuntimeError("Gridpoint positions/displacements contain NaN or infinity.")
    return state


def bounding_box(state, label):
    xyz, ids = state["xyz"], state["ids"]
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    # One representative ID per exact extremum; ties can contain many IDs.
    low_ids = ids[np.argmin(xyz, axis=0)]
    high_ids = ids[np.argmax(xyz, axis=0)]
    print("\n{}: {} gridpoints".format(label, len(ids)))
    for axis, name in enumerate("xyz"):
        print("  {}: min={:.12g} (GP {}), max={:.12g} (GP {})".format(
            name, low[axis], low_ids[axis], high[axis], high_ids[axis]))
    return {
        "min_xyz": low.tolist(), "max_xyz": high.tolist(),
        "representative_min_gp_ids_xyz": low_ids.tolist(),
        "representative_max_gp_ids_xyz": high_ids.tolist(),
    }


def select_boundary_rows(state, bbox):
    z_min, z_max = bbox["min_xyz"][2], bbox["max_xyz"][2]
    height = z_max - z_min
    if not np.isfinite(height) or height <= 0.0:
        raise RuntimeError("The model has no positive z extent.")
    roundoff = 32.0 * np.finfo(float).eps * max(abs(z_min), abs(z_max), height)
    if Z_TOL_ABS is None:
        tolerance = max(Z_TOL_MIN, Z_TOL_REL * height, roundoff)
    else:
        tolerance = max(float(Z_TOL_ABS), roundoff)
    if not np.isfinite(tolerance) or tolerance < 0.0 or 2.0 * tolerance >= height:
        raise ValueError("z tolerance must be finite, nonnegative and less than half the height.")
    z = state["xyz"][:, 2]
    bottom = np.flatnonzero(np.abs(z - z_min) <= tolerance)
    top = np.flatnonzero(np.abs(z - z_max) <= tolerance)
    if not len(bottom) or not len(top):
        raise RuntimeError("An automatically selected boundary is empty.")
    print("  z tolerance: {:.12g}".format(tolerance))
    print("  Selected BEFORE bottom/top gridpoints: {} / {}".format(len(bottom), len(top)))
    return {"bottom": bottom, "top": top}, tolerance


def read_shell_state(label):
    # Unique element component ID -> (structural ID, ordered node component IDs).
    elements, nodes, counts, available = {}, {}, {}, {}
    for shell in it.structure.list():
        if type(shell) is not it.structure.Shell:
            continue
        sid = int(shell.id())
        available[sid] = available.get(sid, 0) + 1
        if sid not in TARGET_SHELL_IDS:
            continue
        eid = int(shell.component_id())
        conn = []
        for node in shell.nodes():
            nid = int(node.component_id())
            conn.append(nid)
            if nid not in nodes:
                # disp_global() returns (translation vec3, rotation vec3).
                nodes[nid] = (vnp(node.pos()), vnp(node.disp_global()[0]))
        if len(conn) < 3 or len(set(conn)) != len(conn):
            raise RuntimeError("Shell component {} has invalid polygon connectivity.".format(eid))
        if eid in elements:
            raise RuntimeError("Duplicate Shell component ID {}.".format(eid))
        elements[eid] = (sid, tuple(conn))
        counts[sid] = counts.get(sid, 0) + 1
    missing = sorted(set(TARGET_SHELL_IDS) - set(counts))
    if missing:
        raise RuntimeError("{}: Shell structural IDs {} were not found. Available: {}".format(
            label, missing, sorted(available.items())))
    if not nodes:
        raise RuntimeError("No target Shell nodes were found.")
    ids = np.asarray(sorted(nodes), dtype=np.int64)
    xyz = np.asarray([nodes[int(nid)][0] for nid in ids], dtype=float)
    disp = np.asarray([nodes[int(nid)][1] for nid in ids], dtype=float)
    if not (np.isfinite(xyz).all() and np.isfinite(disp).all()):
        raise RuntimeError("Shell data contain NaN or infinity.")
    print("{} Shells: {} elements, {} unique nodes; structural ID counts {}".format(
        label, len(elements), len(ids), sorted(counts.items())))
    return {"ids": ids, "xyz": xyz, "disp": disp, "elements": elements}


def matching_rows(sorted_ids, wanted_ids, label):
    rows = np.searchsorted(sorted_ids, wanted_ids)
    found = rows < len(sorted_ids)
    found[found] = sorted_ids[rows[found]] == wanted_ids[found]
    if not found.all():
        missing = wanted_ids[~found]
        raise RuntimeError("{}: {} tracked IDs missing; first 20: {}. "
                           "No boundary points were silently discarded.".format(
                               label, len(missing), missing[:20].tolist()))
    return rows


def validate_shell_match(before, after):
    # Both node arrays are sorted by component ID, independent of iterator order.
    if not np.array_equal(before["ids"], after["ids"]):
        missing = np.setdiff1d(before["ids"], after["ids"])
        added = np.setdiff1d(after["ids"], before["ids"])
        raise RuntimeError("Shell node IDs changed: {} missing, {} added; first missing/added: {} / {}".format(
            len(missing), len(added), missing[:20].tolist(), added[:20].tolist()))
    if before["elements"] != after["elements"]:
        raise RuntimeError("Shell element component IDs, structural IDs or ordered connectivity changed.")


# ============================================================
# Outputs: legacy VTK POLYDATA; no vtk/meshio installation needed
# ============================================================
def output_path(filename):
    path = Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_vtk(filename, shell, xyz, u_shell, u_pure, u_drift, title):
    path = output_path(filename)
    ids = shell["ids"]
    index = {int(nid): row for row, nid in enumerate(ids)}
    eids = sorted(shell["elements"])
    connectivity = [shell["elements"][eid][1] for eid in eids]
    vectors = {"actual_displacement": u_shell, "pure_deformation": u_pure,
               "removed_global_drift": u_drift}
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# vtk DataFile Version 3.0\n{}\nASCII\nDATASET POLYDATA\n".format(title))
        f.write("POINTS {} double\n".format(len(ids)))
        np.savetxt(f, xyz, fmt="%.16e")
        f.write("POLYGONS {} {}\n".format(
            len(connectivity), sum(1 + len(conn) for conn in connectivity)))
        for conn in connectivity:
            f.write("{} {}\n".format(len(conn), " ".join(str(index[nid]) for nid in conn)))
        f.write("POINT_DATA {}\n".format(len(ids)))
        # IDs stored as double remain exact through 2**53; avoid platform int widths.
        f.write("SCALARS node_component_id double 1\nLOOKUP_TABLE default\n")
        np.savetxt(f, ids, fmt="%d")
        for name, values in vectors.items():
            f.write("VECTORS {} double\n".format(name))
            np.savetxt(f, values, fmt="%.16e")
        f.write("SCALARS pure_deformation_magnitude double 1\nLOOKUP_TABLE default\n")
        np.savetxt(f, np.linalg.norm(u_pure, axis=1), fmt="%.16e")
        f.write("CELL_DATA {}\n".format(len(eids)))
        for name, values in (
                ("shell_component_id", eids),
                ("shell_structural_id", [shell["elements"][eid][0] for eid in eids])):
            f.write("SCALARS {} double 1\nLOOKUP_TABLE default\n".format(name))
            np.savetxt(f, values, fmt="%d")
    return str(path)


def write_boundary_csv(filename, before, after, boundary_rows, after_rows, delta):
    path = output_path(filename)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["boundary", "gridpoint_id", "before_x", "before_y", "before_z",
                         "after_x", "after_y", "after_z", "movement_x", "movement_y", "movement_z"])
        for name in ("bottom", "top"):
            for rb, ra, movement in zip(boundary_rows[name], after_rows[name], delta[name]):
                writer.writerow([name, int(before["ids"][rb]), *before["xyz"][rb],
                                 *after["xyz"][ra], *movement])
    return str(path)


# ============================================================
# Main process
# ============================================================
def main():
    if DISPLACEMENT_SOURCE not in ("position", "displacement"):
        raise ValueError('DISPLACEMENT_SOURCE must be "position" or "displacement".')
    if not TARGET_SHELL_IDS:
        raise ValueError("TARGET_SHELL_IDS cannot be empty.")
    if not np.isfinite(SCALE) or SCALE < 0.0:
        raise ValueError("SCALE must be finite and nonnegative.")
    for value in (Z_TOL_ABS, Z_TOL_REL, Z_TOL_MIN):
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError("Tolerance settings must be finite and nonnegative.")
    output_files = [OUTFILE_BEFORE, OUTFILE_AFTER, OUTFILE_BOUNDARIES, OUTFILE_SUMMARY]
    output_names = [Path(name).expanduser().resolve() for name in output_files]
    if len(set(output_names)) != len(output_names):
        raise ValueError("Output filenames must all be different.")
    save_names = {Path(name).expanduser().resolve() for name in (BEFORE_SAVE, AFTER_SAVE)}
    if any(name in save_names for name in output_names):
        raise ValueError("An output filename would overwrite an input save.")

    it.command("python-reset-state false")
    restore_model(BEFORE_SAVE)
    gp_before = read_all_gridpoints()
    bbox_before = bounding_box(gp_before, "BEFORE")
    boundary_rows, z_tolerance = select_boundary_rows(gp_before, bbox_before)
    shell_before = read_shell_state("BEFORE")
    z_bottom = float(gp_before["xyz"][boundary_rows["bottom"], 2].mean())
    z_top = float(gp_before["xyz"][boundary_rows["top"], 2].mean())
    if z_top <= z_bottom:
        raise RuntimeError("Top mean elevation must exceed bottom mean elevation.")

    # All BEFORE snapshots contain only arrays, integers, tuples and dictionaries.
    restore_model(AFTER_SAVE)
    gp_after = read_all_gridpoints()
    bbox_after = bounding_box(gp_after, "AFTER")
    shell_after = read_shell_state("AFTER")
    validate_shell_match(shell_before, shell_after)

    key = "xyz" if DISPLACEMENT_SOURCE == "position" else "disp"
    after_rows, delta, means = {}, {}, {}
    position_changes, displacement_changes = [], []
    for name in ("bottom", "top"):
        rb = boundary_rows[name]
        # Never reselect AFTER by its z extrema: follow the original material IDs.
        ra = matching_rows(gp_after["ids"], gp_before["ids"][rb], "AFTER " + name)
        after_rows[name] = ra
        delta[name] = gp_after[key][ra] - gp_before[key][rb]
        means[name] = delta[name].mean(axis=0)  # equal weight for every matched GP
        position_changes.append(gp_after["xyz"][ra] - gp_before["xyz"][rb])
        displacement_changes.append(gp_after["disp"][ra] - gp_before["disp"][rb])
        print("{}: matched {} / {} IDs; mean movement {}".format(
            name.upper(), len(ra), len(rb), means[name]))

    pos_max = max(float(np.linalg.norm(v, axis=1).max()) for v in position_changes)
    disp_max = max(float(np.linalg.norm(v, axis=1).max()) for v in displacement_changes)
    noise = 1.0e-12 * max(1.0, bbox_before["max_xyz"][2] - bbox_before["min_xyz"][2])
    if DISPLACEMENT_SOURCE == "position" and pos_max <= noise and disp_max > 100.0 * noise:
        warnings.warn('Boundary GP positions are unchanged but displacement fields changed. '
                      'For small strain, set DISPLACEMENT_SOURCE = "displacement" and rerun. '
                      'Check that displacement references were not reset between saves.', RuntimeWarning)

    xyz_before = shell_before["xyz"]
    u_shell = shell_after[key] - shell_before[key]
    eta = (xyz_before[:, 2] - z_bottom) / (z_top - z_bottom)
    remove = np.array([REMOVE_X, REMOVE_Y, REMOVE_Z], dtype=bool)
    u_drift = (means["bottom"][None, :] + eta[:, None] *
               (means["top"] - means["bottom"])[None, :]) * remove[None, :]
    u_pure = u_shell - u_drift
    xyz_after_plot = xyz_before + SCALE * u_pure
    if not all(np.isfinite(v).all() for v in (u_shell, u_drift, u_pure, xyz_after_plot)):
        raise RuntimeError("Calculated displacement or plot geometry is non-finite.")
    outside = int(np.count_nonzero((eta < 0.0) | (eta > 1.0)))
    if outside:
        print("{} Shell nodes lie outside the boundary mean elevations; drift is linearly extrapolated.".format(outside))

    # Both geometries carry the same BEFORE-to-AFTER increment fields for comparison.
    # Geometry is already scaled: do not apply Warp By Vector again in ParaView.
    paths = {
        "before_vtk": write_vtk(OUTFILE_BEFORE, shell_before, xyz_before, u_shell, u_pure,
                                u_drift, "FLAC3D BEFORE geometry; comparison increment fields"),
        "after_vtk": write_vtk(OUTFILE_AFTER, shell_before, xyz_after_plot, u_shell, u_pure,
                               u_drift, "FLAC3D BEFORE plus scaled drift-corrected displacement"),
        "boundary_csv": write_boundary_csv(OUTFILE_BOUNDARIES, gp_before, gp_after,
                                            boundary_rows, after_rows, delta),
    }
    summary = {
        "before_save": str(BEFORE_SAVE), "after_save": str(AFTER_SAVE),
        "displacement_source": DISPLACEMENT_SOURCE,
        "before_bbox": bbox_before, "after_bbox": bbox_after,
        "before_gridpoint_count": len(gp_before["ids"]),
        "after_gridpoint_count": len(gp_after["ids"]),
        "z_tolerance": z_tolerance,
        "boundary_selection": "BEFORE abs(z-z_min/max) <= tolerance; same IDs tracked in AFTER",
        "bottom_count": len(boundary_rows["bottom"]), "top_count": len(boundary_rows["top"]),
        "z_bottom_mean_before": z_bottom, "z_top_mean_before": z_top,
        "mean_bottom_movement_xyz": means["bottom"].tolist(),
        "mean_top_movement_xyz": means["top"].tolist(),
        "mean_weighting": "equal weight per matched boundary gridpoint",
        "remove_xyz": remove.tolist(), "scale": float(SCALE),
        "shell_structural_ids": sorted(int(sid) for sid in TARGET_SHELL_IDS),
        "shell_element_count": len(shell_before["elements"]),
        "shell_node_count": len(shell_before["ids"]),
        "shell_nodes_outside_boundary_mean_elevations": outside,
        "max_pure_deformation_magnitude": float(np.linalg.norm(u_pure, axis=1).max()),
        "outputs": dict(paths),
    }
    summary_path = output_path(OUTFILE_SUMMARY)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    paths["summary_json"] = str(summary_path)
    print("\nCompleted. Displacement source: {}; geometry scale: {}".format(DISPLACEMENT_SOURCE, SCALE))
    print("VTK displacement vectors are unscaled; AFTER geometry is already scaled.")
    for name, path in paths.items():
        print("{}: {}".format(name, path))
    return summary


if __name__ == "__main__":
    main()