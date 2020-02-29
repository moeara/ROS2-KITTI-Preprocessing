"""
to generate bird eye view, we have to filter point cloud
first. which means we have to limit coordinates

"""
import numpy as np
import cv2
import pointcloud2_to_numpy as np_pc2

from sparsify_pointclouds import vis_mlab, sparsify_pointclouds, load_pc

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import time

res = 0.05
# image size would be 400x800
side_range = (-30, 30)
fwd_range = (-20, 20)

cloud_msg = PointCloud2()

# Save received PC


class PC_Subscriber(Node):

    def __init__(self):
        super().__init__('pointcloud_subscriber')
        self.subscription = self.create_subscription(PointCloud2, '/velodyne_points', self.pointcloud_callback, 1)
        # self.subscription  #to prevent the unused variable warning

    def pointcloud_callback(self, msg):
        global cloud_msg
        cloud_msg = msg


def gen_bev_map(pc, lr_range=[-10, 10], bf_range=[-20, 20], res=0.05):
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    # filter point cloud
    f_filt = np.logical_and((x>bf_range[0]), (x<bf_range[1]))
    s_filt = np.logical_and((y>-lr_range[1]), (y<-lr_range[0]))
    filt = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filt).flatten()
    x = x[indices]
    y = y[indices]
    z = z[indices]

    # convert coordinates to grid cell locations
    x_img = (-y/res).astype(np.int32)
    y_img = (-x/res).astype(np.int32)
    # shifting image, make min pixel is 0,0
    x_img -= int(np.floor(lr_range[0]/res))
    y_img += int(np.ceil(bf_range[1]/res))

    # crop y to make it not bigger than 255
    height_range = (-2, 0.5)
    pixel_values = np.clip(a=z, a_min=height_range[0], a_max=height_range[1])
    def scale_to_255(a, min, max, dtype=np.uint8):
        return (((a - min) / float(max - min)) * 255).astype(dtype)
    pixel_values = scale_to_255(pixel_values, min=height_range[0], max=height_range[1])

    # according to width and height generate image
    w = 1+int((lr_range[1] - lr_range[0])/res)
    h = 1+int((bf_range[1] - bf_range[0])/res)
    im = np.zeros([h, w], dtype=np.uint8)
    im[y_img, x_img] = pixel_values
    cropped_cloud = np.vstack([x, y, z]).transpose()
    return im, cropped_cloud


def main(args=None):
    # Initialise ros2 node and the subscriber
    rclpy.init(args=None)
    pointcloud_subscriber = PC_Subscriber()

    i = 0

    try:
        while True:
            rclpy.spin_once(pointcloud_subscriber)

            # Convert the ros2 pointcloud2 msg to a numpy array
            points = np_pc2.pointcloud2_to_xyz_array(cloud_msg, remove_nans=True)

            # Visualise the generated BEV
            im, cropped_cloud = gen_bev_map(points)

            # Visualise the original pointcloud
            #vis_mlab(points, i)
            cv2.imwrite('my_bev/sequence/bev' + str(i) + '.png', im)

            i += 1

    except KeyboardInterrupt:
        # Clean up after node exists
        pointcloud_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
