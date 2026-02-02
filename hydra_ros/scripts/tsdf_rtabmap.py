import argparse
from pathlib import Path
import open3d as o3d
import numpy as np
import cv2
import yaml
import os


class TSDF:
    def __init__(self, input_path: Path, output_path: Path):
        assert input_path.is_dir(), f"Input path {input_path} is not a directory."
        assert (
            output_path.parent.exists()
        ), f"Output path {output_path.parent} does not exist."
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self.poses = np.loadtxt(input_path / "poses_cam.txt").reshape(-1, 3, 4)
        # Convert the poses to 4x4 matrices.
        self.poses = np.array([np.vstack((pose, [0, 0, 0, 1])) for pose in self.poses])

        with (input_path / "calibration.yaml").open("r") as f:
            data = yaml.safe_load(f)
        camera_intrinsics = np.asarray(data["camera_matrix"]["data"]).reshape(3, 3)
        width = data["image_width"]
        height = data["image_height"]
        # Get the camera intrinsics.
        self.intrinshics = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            camera_intrinsics[0, 0],
            camera_intrinsics[1, 1],
            camera_intrinsics[0, 2],
            camera_intrinsics[1, 2],
        )

        numbers = [int(num[:-4]) for num in os.listdir(input_path / "rgb")]
        numbers.sort()
        self.images_paths = [input_path / f"rgb/{num}.png" for num in numbers]
        self.depth_paths = [input_path / f"depth/{num}.png" for num in numbers]
        self.output_path = output_path
        self.input_path = input_path

    def run(self):
        for pose, image_path, depth_path in zip(
            self.poses, self.images_paths, self.depth_paths
        ):
            color_image = cv2.imread(str(self.input_path / "rgb" / image_path))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            color_image = np.array(color_image, dtype=np.uint8)
            depth_image = cv2.imread(
                str(self.input_path / "depth" / depth_path), cv2.IMREAD_UNCHANGED
            )

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image),
                o3d.geometry.Image(depth_image),
                convert_rgb_to_intensity=False,
                depth_trunc=10.0,
            )
            pose = np.linalg.inv(pose)
            self.volume.integrate(rgbd_image, self.intrinshics, pose)

        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(self.output_path), mesh)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a mesh from a TSDF volume.")
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to the input directory containing the RGB and depth images.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to the output file where the mesh will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tsdf = TSDF(args.input_path, args.output_path)
    tsdf.run()


if __name__ == "__main__":
    main()
