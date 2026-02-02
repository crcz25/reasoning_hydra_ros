import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import open3d as o3d
from tqdm import trange


class TSDFReconstructor:
    def __init__(self, dataset_path, scene_name, voxel_length=0.005, sdf_trunc=0.04):
        self.dataset_path = Path(dataset_path) / scene_name / "sequence"
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc

        (
            self.intrinsics,
            self.H,
            self.W,
            self.depth_scale,
            self.num_frames,
        ) = self._load_intrinsics()
        self.data_list = self._get_data_list()

        # Initialize TSDF volume
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def _load_intrinsics(self):
        with (self.dataset_path / "_info.txt").open() as f:
            lines = f.readlines()
            for line in lines:
                if "m_colorWidth" in line:
                    W = int(line.split("=")[-1])
                if "m_colorHeight" in line:
                    H = int(line.split("=")[-1])
                if "m_calibrationColorIntrinsic" in line:
                    intrinsics = (
                        np.array(line.split("= ")[-1].split(" ")[:-1])
                        .astype(float)
                        .reshape(4, 4)
                    )
                if "m_frames.size" in line:
                    num_frames = int(line.split("=")[-1])
                if "m_depthShift" in line:
                    scale = float(line.split("=")[-1])

        return intrinsics[:3, :3], H, W, scale, num_frames

    def _get_data_list(self):
        rgb_data_list = []
        depth_data_list = []
        pose_data_list = []
        for i in range(self.num_frames):
            rgb_data_list.append(self.dataset_path / f"frame-{i:06d}.color.jpg")
            depth_data_list.append(self.dataset_path / f"frame-{i:06d}.depth.pgm")
            pose_data_list.append(self.dataset_path / f"frame-{i:06d}.pose.txt")

        return list(zip(rgb_data_list, depth_data_list, pose_data_list))

    def _load_rgb_image(self, path):
        return np.asarray(Image.open(path))

    def _load_depth_image(self, path):
        depth_image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return (
            cv2.resize(depth_image, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            / self.depth_scale
        ).astype(np.float32)

    def _load_pose(self, path):
        return np.loadtxt(path)

    def _get_intrinsic_o3d(self):
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            self.W,
            self.H,
            self.intrinsics[0, 0],
            self.intrinsics[1, 1],
            self.intrinsics[0, 2],
            self.intrinsics[1, 2],
        )
        return intrinsic

    def integrate(self):
        intrinsic = self._get_intrinsic_o3d()

        for idx in trange(len(self.data_list), desc="Integrating"):
            rgb_path, depth_path, pose_path = self.data_list[idx]

            if not (rgb_path.exists() and depth_path.exists() and pose_path.exists()):
                continue

            rgb = self._load_rgb_image(rgb_path)
            depth = self._load_depth_image(depth_path)
            pose = self._load_pose(pose_path)

            # Convert to Open3D RGBD image
            rgb_o3d = o3d.geometry.Image(rgb)
            depth_o3d = o3d.geometry.Image(depth)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d,
                depth_o3d,
                convert_rgb_to_intensity=False,
                depth_trunc=8.0,
                depth_scale=1.0,
            )

            extrinsic = np.linalg.inv(pose)  # Open3D expects camera-to-world
            self.tsdf_volume.integrate(rgbd_image, intrinsic, extrinsic)

    def save_mesh(self, output_path="mesh.ply"):
        mesh = self.tsdf_volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Saved mesh to {output_path}")


def main():
    dataset_path = "/home/albert/Desktop/tmp_results/dataset/data/3RScan"
    scene_name = "4fbad331-465b-2a5d-8488-852fcda9513c"
    reconstructor = TSDFReconstructor(dataset_path, scene_name)
    reconstructor.integrate()
    reconstructor.save_mesh(
        "/home/albert/Desktop/tmp_results/results/3rscan/4fbad331-465b-2a5d-8488-852fcda9513c/hovsg_format/reconstructed_mesh.ply"
    )


if __name__ == "__main__":
    main()
