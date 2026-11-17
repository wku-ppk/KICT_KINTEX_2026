import itasca as it
import numpy as np

# ============================================================
# User inputs
# ============================================================

BEFORE_SAVE = r"[4]_.sav"
AFTER_SAVE = r"[5]DynamicRun3D_T_Via1.sav"

OUTFILE_BEFORE = r"mainshaft_shell_before.vtk"
OUTFILE_AFTER = r"mainshaft_shell_after_puredef.vtk"

TARGET_SHELL_IDS = {401, 411, 412}

BOTTOM_GPS = [2271, 214, 92796]
TOP_GPS = [2295, 238, 92820]

SCALE = 20.0

REMOVE_X = True
REMOVE_Y = True
REMOVE_Z = False


# ============================================================
# Helper functions
# ============================================================

def vnp(v):
    return np.array([v[0], v[1], v[2]], dtype=float)


def restore_model(filename):
    it.command('model restore "{}"'.format(filename))


def print_shell_id_summary():
    shell_counts = {}

    for shell in it.structure.list():
        if type(shell) is it.structure.Shell:
            sid = shell.id()
            shell_counts[sid] = shell_counts.get(sid, 0) + 1

    print("------------------------------------------------")
    print("Available Shell IDs between 400 and 499:")

    for sid in sorted(shell_counts):
        if 400 <= sid <= 499:
            print(
                "Shell ID {}: {} elements".format(
                    sid,
                    shell_counts[sid]
                )
            )

    print("------------------------------------------------")


def read_shell_state():
    shell_node_component_ids = []
    shell_connectivity = []
    id_counts = {sid: 0 for sid in TARGET_SHELL_IDS}

    for shell in it.structure.list():
        if type(shell) is not it.structure.Shell:
            continue

        if shell.id() in TARGET_SHELL_IDS:
            conn = [
                node.component_id()
                for node in shell.nodes()
            ]

            shell_connectivity.append(conn)
            shell_node_component_ids.extend(conn)
            id_counts[shell.id()] += 1

    shell_node_component_ids = sorted(
        set(shell_node_component_ids)
    )

    if not shell_node_component_ids:
        print_shell_id_summary()
        raise RuntimeError(
            "No target Shell elements were found."
        )

    node_map = {
        node.component_id(): node
        for node in it.structure.node.list()
    }

    xyz = np.array([
        vnp(node_map[nid].pos())
        for nid in shell_node_component_ids
    ])

    print("------------------------------------------------")
    print("Selected Shell element count:", len(shell_connectivity))
    print("Selected unique node count:", len(shell_node_component_ids))

    for sid in sorted(TARGET_SHELL_IDS):
        print(
            "Shell ID {}: {} selected elements".format(
                sid,
                id_counts[sid]
            )
        )

    print("------------------------------------------------")

    return shell_node_component_ids, shell_connectivity, xyz


def read_gp_state(gp_ids):
    xyz = {}

    for gid in gp_ids:
        gp = it.gridpoint.find(gid)

        if gp is None:
            raise RuntimeError(
                "Grid point {} was not found.".format(gid)
            )

        xyz[gid] = vnp(gp.pos())

    return xyz


def write_vtk(filename, node_ids, connectivity, xyz, u_pure, u_drift):
    node_index = {
        nid: i
        for i, nid in enumerate(node_ids)
    }

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FLAC3D main shaft shell geometry\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write("POINTS {} float\n".format(len(xyz)))

        for p in xyz:
            f.write(
                "{:.8e} {:.8e} {:.8e}\n".format(
                    p[0],
                    p[1],
                    p[2]
                )
            )

        polygon_value_count = sum(
            1 + len(conn)
            for conn in connectivity
        )

        f.write(
            "POLYGONS {} {}\n".format(
                len(connectivity),
                polygon_value_count
            )
        )

        for conn in connectivity:
            indices = [
                node_index[nid]
                for nid in conn
            ]

            f.write(
                "{} {}\n".format(
                    len(indices),
                    " ".join(str(i) for i in indices)
                )
            )

        f.write("POINT_DATA {}\n".format(len(xyz)))

        f.write("VECTORS pure_deformation float\n")

        for u in u_pure:
            f.write(
                "{:.8e} {:.8e} {:.8e}\n".format(
                    u[0],
                    u[1],
                    u[2]
                )
            )

        f.write("VECTORS removed_global_drift float\n")

        for u in u_drift:
            f.write(
                "{:.8e} {:.8e} {:.8e}\n".format(
                    u[0],
                    u[1],
                    u[2]
                )
            )

        f.write("SCALARS pure_deformation_magnitude float 1\n")
        f.write("LOOKUP_TABLE default\n")

        for u in u_pure:
            f.write(
                "{:.8e}\n".format(
                    np.linalg.norm(u)
                )
            )


# ============================================================
# Main process
# ============================================================

it.command("python-reset-state false")

# ------------------------------------------------------------
# 1. Read static equilibrium state
# ------------------------------------------------------------

restore_model(BEFORE_SAVE)

node_ids, connectivity, xyz_before = read_shell_state()

all_gp_ids = BOTTOM_GPS + TOP_GPS

gp_xyz_before = read_gp_state(all_gp_ids)

z_bottom = np.mean([
    gp_xyz_before[gid][2]
    for gid in BOTTOM_GPS
])

z_top = np.mean([
    gp_xyz_before[gid][2]
    for gid in TOP_GPS
])

if abs(z_top - z_bottom) < 1.0e-10:
    raise RuntimeError(
        "Top and bottom elevations are identical."
    )

# ------------------------------------------------------------
# 2. Read dynamic state
# ------------------------------------------------------------

restore_model(AFTER_SAVE)

node_ids_after, connectivity_after, xyz_after = read_shell_state()

gp_xyz_after = read_gp_state(all_gp_ids)

if node_ids != node_ids_after:
    raise RuntimeError(
        "Shell node IDs differ between before and after save files."
    )

# ------------------------------------------------------------
# 3. Calculate actual shell displacement
# ------------------------------------------------------------

u_shell = xyz_after - xyz_before

# ------------------------------------------------------------
# 4. Calculate global boundary drift
# ------------------------------------------------------------

u_bottom = np.mean(
    [
        gp_xyz_after[gid] - gp_xyz_before[gid]
        for gid in BOTTOM_GPS
    ],
    axis=0
)

u_top = np.mean(
    [
        gp_xyz_after[gid] - gp_xyz_before[gid]
        for gid in TOP_GPS
    ],
    axis=0
)

eta = (
    (xyz_before[:, 2] - z_bottom) /
    (z_top - z_bottom)
)

u_drift = np.zeros_like(u_shell)

if REMOVE_X:
    u_drift[:, 0] = (
        u_bottom[0] +
        eta * (u_top[0] - u_bottom[0])
    )

if REMOVE_Y:
    u_drift[:, 1] = (
        u_bottom[1] +
        eta * (u_top[1] - u_bottom[1])
    )

if REMOVE_Z:
    u_drift[:, 2] = (
        u_bottom[2] +
        eta * (u_top[2] - u_bottom[2])
    )

# ------------------------------------------------------------
# 5. Calculate pure local deformation
# ------------------------------------------------------------

u_pure = u_shell - u_drift

# ------------------------------------------------------------
# 6. Generate two VTK geometries
# ------------------------------------------------------------

xyz_before_plot = xyz_before
xyz_after_plot = xyz_before + SCALE * u_pure

write_vtk(
    OUTFILE_BEFORE,
    node_ids,
    connectivity,
    xyz_before_plot,
    u_pure,
    u_drift
)

write_vtk(
    OUTFILE_AFTER,
    node_ids,
    connectivity,
    xyz_after_plot,
    u_pure,
    u_drift
)

print("Before save file:", BEFORE_SAVE)
print("After save file :", AFTER_SAVE)
print("Before VTK file :", OUTFILE_BEFORE)
print("After VTK file  :", OUTFILE_AFTER)
print("Scale factor    :", SCALE)