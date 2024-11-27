import copy
import heapq
import math
from itertools import pairwise

import numpy as np
import open3d
from scipy.spatial import Delaunay
from skspatial.objects import Plane, Vector, Triangle

import trimesh

BRIGHT_GREEN = np.array([106, 13, 173.0]) / 255
DARK_GREEN = np.array([120.0, 240, 10.0]) / 255
MAX_DEGREE = 8


def calc_normal(triangles):
    return np.cross(triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0], axis=1)


def calc_area(triangles):
    return np.linalg.norm(calc_normal(triangles), axis=1) / 2


class Sentence:
    def __init__(self):
        self.vertices = []
        self.parent = []
        self.lines = []
        self.colors = []
        self.last_added = 0

    @classmethod
    def create_axiom(cls):
        axiom = cls()
        axiom.vertices = [
            np.array([0.0, 0.0, 0.0])[..., None],
            np.array([0.0, 1.0, 0.0])[..., None],
        ]
        axiom.lines = [
            (0, 1),
        ]
        axiom.colors = [BRIGHT_GREEN] * len(axiom.vertices)
        axiom.last_added = 3
        return axiom

    def triangulate_delaunay(self):
        plane = Plane(point=(0, 0, 0), normal=(0, 0, 1))
        plane_vec1 = Vector([1, 0, 0])
        plane_vec2 = Vector([0, 1, 0])
        all_points = self.vertices
        prj_points = [plane.project_point(p.squeeze()) for p in all_points]
        prj_points_2d = []
        for p in prj_points:
            p_x = np.linalg.norm(plane_vec1.project_vector(p) - plane.point)
            p_y = np.linalg.norm(plane_vec2.project_vector(p) - plane.point)
            prj_points_2d.append(np.array([p_x, p_y]))

        prj_points_2d = np.stack(prj_points_2d).squeeze()
        tri = Delaunay(prj_points_2d)
        return self.vertices, tri.simplices, self.colors

    def triangulate(self):

        def parent(idx): return idx // 2
        def r_child(idx): return idx * 2
        def l_child(idx): return 1 + idx * 2
        def is_leaf(idx): return idx >= len(self.vertices) - 2**MAX_DEGREE

        curr_idx = len(self.vertices) - 1
        contours = []
        visited = [False] * len(self.vertices)
        visited_cnt = 0
        while True:
            if is_leaf(curr_idx):
                if len(contours) != 0:
                    contours[-1].append(curr_idx)
                contours.append([])
            visited[curr_idx] = True
            visited_cnt += 1
            contours[-1].append(curr_idx)
            if not is_leaf(curr_idx) and not visited[l_child(curr_idx)]:
                curr_idx = l_child(curr_idx)
            elif not is_leaf(curr_idx) and not visited[r_child(curr_idx)]:
                curr_idx = r_child(curr_idx)
            elif curr_idx != 0:
                curr_idx = parent(curr_idx)
            else:
                break
        contours[-1].append(0)
        curr_idx = len(self.vertices) - 1
        contours.append([])
        while curr_idx != 0:
            contours[-1].append(curr_idx)
            curr_idx = parent(curr_idx)
        contours[-1].append(0)

        points = np.stack(self.vertices).squeeze()
        vertices = copy.deepcopy(self.vertices)
        colors = copy.deepcopy(self.colors)
        tris = []

        for cont in contours:
            centroid = points[cont].mean(axis=0)
            tris.extend([(vi1, vi2, len(vertices)) for vi1, vi2 in pairwise(cont)])
            colors.append(DARK_GREEN)
            vertices.append(centroid)
        vertices = [vert.squeeze() for vert in vertices]

        tris_pq = []
        for tri in tris:
            try:
                area = -Triangle(*[vertices[v] for v in tri]).area()
            except ValueError:
                area = 0
            heapq.heappush(tris_pq, (area, tri))

        for _ in range(1_000):
            _, tri = heapq.heappop(tris_pq)
            tri_pts = [vertices[v] for v in tri]
            center = np.mean(tri_pts, axis=0) + np.random.normal(scale=np.array([0.01, 0.01, 0.01]))
            center_idx = len(vertices)
            for idx1, idx2 in pairwise([*tri, tri[0]]):
                new_tri = (center_idx, idx1, idx2)
                try:
                    area = -Triangle(vertices[idx1], vertices[idx2], center).area()
                except ValueError:
                    area = 0
                heapq.heappush(tris_pq, (area, new_tri))
            vertices.append(center)
            colors.append(colors[tri[0]])

        tris = [tri for _, tri in tris_pq]
        return vertices, tris, colors


def generate_leaf():
    axiom = Sentence.create_axiom()

    def apply_rule(curr: Sentence):
        curr_new = copy.deepcopy(curr)

        for idx1, idx2 in curr.lines[len(curr.lines) - curr.last_added:]:
            rot_r = open3d.geometry.get_rotation_matrix_from_axis_angle(-np.array([-0.1, -0.1, 1.0]) * math.pi / 7)
            rot_l = open3d.geometry.get_rotation_matrix_from_axis_angle(np.array([-0.1, -0.1, 1.0]) * math.pi / 7)
            vert1 = curr.vertices[idx1]
            vert2 = curr.vertices[idx2]

            alpha = 0.35
            branch_center = alpha * vert2 + (1.0 - alpha) * vert1

            rot_r += np.random.normal(scale=np.array([0.1, 0.1, 0.1]))[..., None]
            rot_l += np.random.normal(scale=np.array([0.1, 0.1, 0.1]))[..., None]

            new_idx1 = len(curr_new.vertices)
            vert_r = rot_r @ (vert2 - branch_center) + vert2
            curr_new.vertices.append(vert_r)

            new_idx2 = len(curr_new.vertices)
            vert_l = rot_l @ (vert2 - branch_center) + vert2
            curr_new.vertices.append(vert_l)

            curr_new.lines.append((idx2, new_idx1))
            curr_new.lines.append((idx2, new_idx2))

            alpha = 0.8
            color = alpha * curr.colors[idx1] + (1.0 - alpha) * DARK_GREEN
            curr_new.colors.append(color)
            curr_new.colors.append(color)

        curr_new.last_added = len(curr_new.lines) - len(curr.lines)
        return curr_new

    population = axiom
    for degree in range(MAX_DEGREE):
        population = apply_rule(population)

    vertices, tris, colors = population.triangulate()
    triangles = []
    for tri in tris:  # kind of face culling
        triangles.append(tri)
        triangles.append(tri[::-1])

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    mesh.triangles = open3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = open3d.utility.Vector3dVector(colors)
    return mesh


def main():
    mesh = generate_leaf()
    open3d.visualization.draw_geometries([mesh])
    open3d.io.write_triangle_mesh('generated_leaf.obj', mesh)


if __name__ == '__main__':
    main()
