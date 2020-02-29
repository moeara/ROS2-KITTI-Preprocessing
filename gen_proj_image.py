"""
Converting lidar pointclouds to a range view image
"""

import sys
import numpy as np
import cv2
import time
from sparsify_pointclouds import vis_mlab, sparsify_pointclouds, load_pc

#sys.path.append('/home/moemen/Work/3d_detection_utils/3d_detection_kit')
import projection_utils_kitti


def main(args=None):

    width = 1242
    height = 375
    img = np.zeros((height, width))

    # Create a conversion object and pass the calibration file to the constructor
    converter = projection_utils_kitti.Calibration('/home/moemen/Work/preprocessing/data/testing/calib/000000.txt')

    # Load the pc for kitti --read the bin file for the lidar bag read converted xyz PC
    points = load_pc('/home/moemen/Work/preprocessing/data/testing/velodyne/000000.bin')[:, :3]
    print("Number of points in PC ", points.shape)

    points = sparsify_pointclouds(points, True)
    print("Number of points in PC after sparsification ", points.shape)

    # Project the PC using converter.pc_to_image
    pixel_positions = np.around(converter.project_velo_to_image(points))
    print("Returned Pixel Locations shape ", pixel_positions.shape)

    start = time.time()

    valid_positions = np.logical_and(
                                     points[:, 0] > 0,
                                     np.logical_and(
                                                    np.logical_and(pixel_positions[:, 0] > 0, pixel_positions[:, 0] < width),
                                                    np.logical_and(pixel_positions[:, 1] > 0, pixel_positions[:, 1] < height)
                                                    )
                                    )

    valid_pixel_depths = points[valid_positions, 0]  # Original (x) depth values from the PC
    valid_pixel_positions = pixel_positions[valid_positions].astype(int)

    # Depth value calculations
    max_depth = np.array([70])  # For normalization, preferably use a constant value though
    normalised_depth = (valid_pixel_depths / max_depth)*255

    normalised_depth[normalised_depth > 255] = 255

    img[valid_pixel_positions[:, 1], valid_pixel_positions[:, 0]] = normalised_depth.astype(int)

    end = time.time()

    print("Computation time ", str((end-start)*1000), " ms")
    print("Valid Pixel Locations ", valid_pixel_positions.shape)
    print("Valid Depth Indices ", valid_pixel_depths.shape)
    print("Max x value ", max(valid_pixel_positions[:, 0]))
    print("Max y value ", max(valid_pixel_positions[:, 1]))

    cv2.imwrite("my_kitti_projections/0-1-1.png", img)


if __name__ == '__main__':
    main()
