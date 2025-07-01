from typing import Optional
from pathlib import Path
import logging
import open3d
import numpy as np
import laspy


def find_rotation(a: np.array, b: np.array) -> np.array:
    """
    Compute rotation matrix to rotate a onto b

    :ref: https://math.stackexchange.com/a/476311 and
      https://stackoverflow.com/a/56505181/30470547

    :param a: Vector to rotate
    :type a: np.array
    :param b: Target vector
    :type b: np.array
    :raises ValueError: Vectors can't point in opposite directions
    :return: Rotation matrix [3x3]
    :rtype: np.array
    """
    if np.array_equal(a, -b):
        raise ValueError("Cannot rotate vectors when a == -b")

    a_n = a / np.linalg.norm(a)
    b_n = b / np.linalg.norm(b) 
    v: np.array = np.cross(a_n, b_n)
    # s: np.floating = np.linalg.norm(v)  # simplified aways (1-c)/s^2 == (1 / (1 + c))
    c: np.floating = np.dot(a_n, b_n)

    vx: np.array = np.array(
        [
            [0    , -v[2],    v[1]],
            [v[2] ,     0,   -v[0]],
            [-v[1],  v[0],    0]
        ]
    )

    R: np.array = np.eye(3) + vx + vx.dot(vx) * (1 / (1 + c))

    return R    

class PointCloud:
    """
    Wrapper around open3d.geometry.PointCloud for convenient access
    """
    def __init__(self):
        self.pc: open3d.geometry.PointCloud = open3d.geometry.PointCloud()
        self.original_header: laspy.LasHeader = None
        self.mesh: open3d.geometry.TriangleMesh = None
        self.mesh_densities: open3d.utility.DoubleVector = None
        self.smoothed_mesh: open3d.geometry.TriangleMesh = None
        self.logger = logging.Logger("PointCloud")

    def load_from_laz(self, model_path: str):
        # TODO: what about all other point cloud atributes?
        laspy_pc: laspy.LasData = laspy.read(Path(model_path))
        self.original_header = laspy_pc.header
        
        xyz: np.array = np.array(laspy_pc.xyz)
        self.pc.points = open3d.utility.Vector3dVector(xyz)

        try:
            rgb: np.array = np.vstack([laspy_pc.red, laspy_pc.green, laspy_pc.blue]).transpose() / np.iinfo(np.uint16).max
            self.pc.colors = open3d.utility.Vector3dVector(rgb)
            del rgb
        except AttributeError:
            self.logger.warn("Supplied point cloud does not contain color information")
        
        self.pc.estimate_normals()        
        self.pc.orient_normals_to_align_with_direction()
        self.pc.normalize_normals()

        del laspy_pc
        del xyz

    def write_to_laz(self, model_path: Path):
        if Path(model_path).exists():
            self.logger.error("File already exists, not overwriting it")
            return

        out = laspy.LasData(self.original_header)
        out.xyz = np.array(self.pc.points)
        out.write(model_path)
    
    def construct_mesh(self, *args, **kwargs):
        self.mesh, self.mesh_densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pc, *args, **kwargs)
        open3d.geometry.TriangleMesh.compute_vertex_normals(self.mesh)
        open3d.geometry.TriangleMesh.compute_triangle_normals(self.mesh)
        self.mesh.normalize_normals()
        self.mesh.orient_triangles()
    
    def smooth_mesh(self, iterations: int):
        """
        Smooth mesh with Laplacian filter for `iterations` iterations

        :param iterations: Number of iterations
        :type iterations: int
        """
        self.smooth_mesh = self.mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    
    def write_mesh(self, model_path: str):
        """
        Thin wrapper aroumd `write_triangle_mesh`

        :param model_path: Output file path, including file name
        :type model_path: Path
        """
        if Path(model_path).exists():
            self.logger.error("File already exists, not overwriting it")
            return
        
        open3d.io.write_triangle_mesh(model_path, self.mesh, compressed=True)

    def rotate_upwards(self, k: int = 10, source: Optional[np.array] = None):
        """
        Rotate the point cloud upwards.

        :note: This assumes convex/concave model.

        :param k: Number of nearest neighbors for normal orientation, defaults to 10
        :type k: int, optional
        :param source: Instead of estimating normals, give a 'source' vector which should
        rotated
        :type source: np.array, optional
        """
        if source is not None:
            R: np.array = find_rotation(source, np.array([0, 0, 1]))
        
            self.pc = self.pc.rotate(R)
        else:
            if not self.pc.has_normals():
                self.pc.estimate_normals()
                self.pc.orient_normals_consistent_tangent_plane(k)
                self.pc.normalize_normals()

            mean_normal: np.array = np.array(self.pc.normals).mean(axis=0)

            R: np.array = find_rotation(mean_normal, np.array([0, 0, 1]))

            self.pc = self.pc.rotate(R)
