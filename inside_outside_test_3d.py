from timeit import default_timer

import numpy as np
from netgen.read_meshio import ReadViaMeshIO
from pykdtree.kdtree import KDTree


def is_inside_tet(query_point, v1, v2, v3, v4):
    """Reference https://web.archive.org/web/20220417181453/https://steve.hollasch.net/cgindex/geometry/ptintet.html"""
    d0 = np.linalg.det(
        [
            [*v1, 1],
            [*v2, 1],
            [*v3, 1],
            [*v4, 1],
        ]
    )
    d1 = np.linalg.det(
        [
            [*query_point, 1],
            [*v2, 1],
            [*v3, 1],
            [*v4, 1],
        ]
    )
    d2 = np.linalg.det(
        [
            [*v1, 1],
            [*query_point, 1],
            [*v3, 1],
            [*v4, 1],
        ]
    )
    d3 = np.linalg.det(
        [
            [*v1, 1],
            [*v2, 1],
            [*query_point, 1],
            [*v4, 1],
        ]
    )
    d4 = np.linalg.det(
        [
            [*v1, 1],
            [*v2, 1],
            [*v3, 1],
            [*query_point, 1],
        ]
    )

    assert np.isclose(d0, d1 + d2 + d3 + d4)
    if np.isclose(d0, 0.0):
        print("WARNING: degenerate tetrahedron")

    if np.isclose((d1, d2, d3, d4), 0.0).any():
        # Point on boundary
        return True

    return np.sign(d0) == np.sign(d1) == np.sign(d2) == np.sign(d3) == np.sign(d4)


mesh = ReadViaMeshIO("./Suzanne.stl")

t0 = default_timer()
mesh.GenerateVolumeMesh()
t1 = default_timer()
print(f"Volume Mesh generated in {t1-t0:.2f} seconds")


tet_centers = []
tets = []
for elem3d in mesh.Elements3D():
    tet_verts = [[v for v in mesh[eid]] for eid in elem3d.vertices]
    tets.append(tet_verts)
    center = np.mean(tet_verts, axis=0)
    tet_centers.append(center)

kd_tree = KDTree(np.asarray(tet_centers))

query_points = np.array([(0, 0, 0)])

dist, idx = kd_tree.query(query_points, k=1)

is_inside = is_inside_tet(query_points[0], *tets[idx[0]])
print(is_inside)
