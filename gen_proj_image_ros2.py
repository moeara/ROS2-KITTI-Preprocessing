"""
Converting lidar pointclouds from a ros2 topic to a range view image
"""

import sys
import numpy as np
import cv2
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

#sys.path.append('/home/moemen/Work/preprocessing')
from sparsify_pointclouds import vis_mlab, sparsify_pointclouds, load_pc
import pointcloud2_to_numpy as np_pc2
import projection_utils

cloud_msg = PointCloud2()

width = 1280
height = 720
img = np.zeros((height, width))


class PC_Subscriber(Node):

    def __init__(self):
        super().__init__('pointcloud_subscriber')
        self.subscription = self.create_subscription(PointCloud2, '/velodyne_points', self.pointcloud_callback, 1)
        # self.subscription  #to prevent the unused variable warning

    def pointcloud_callback(self, msg):
        global cloud_msg
        cloud_msg = msg


def gen_rv_image(full_pc_data, converter, print_data=False):
    points = full_pc_data[:, :3]
    #intensities = full_pc_data[:,3]

    # Project the PC using converter.pc_to_image
    pixel_positions = np.around(converter.project_velo_to_image(points))

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
    normalised_depth = (valid_pixel_depths / max_depth) * 255

    normalised_depth[normalised_depth > 255] = 255 # use np.clip

    global img
    img[valid_pixel_positions[:, 1], valid_pixel_positions[:, 0]] = normalised_depth.astype(int)

    end = time.time()

    if print_data:
        print("Number of points in PC : ", points.shape)
        print("Returned Pixel Locations shape ", pixel_positions.shape)
        print("Valid Pixel Locations", valid_pixel_positions.shape)
        print("Valid Depth Indices", valid_pixel_depths.shape)
        print("Computation time ", str((end - start) * 1000), " ms")
        print('Max x value ', max(valid_pixel_positions[:, 0]))
        print('Max y value ', max(valid_pixel_positions[:, 1]))

    return


def main(args=None):

    rclpy.init(args=None)
    pointcloud_subscriber = PC_Subscriber()

    # Create a conversion object and pass the calibration file to the constructor
    converter = projection_utils.Calibration('calibration_full.npy')

    i = 0

    try:
        while True:
            rclpy.spin_once(pointcloud_subscriber)

            # Convert the ros2 pointcloud2 msg to a numpy array
            full_pc_data = np_pc2.pointcloud2_to_xyz_array(cloud_msg, remove_nans=True)

            # Project the pointcloud
            gen_rv_image(full_pc_data, converter, True)

            # Visualise and save
            #vis_mlab(points, i)
            cv2.imwrite("my_projections/sequence/sequence" + str(i) + ".png", img)
            i += 1

    except KeyboardInterrupt:
        # Clean up after node exists
        pointcloud_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
