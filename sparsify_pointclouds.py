import numpy as np
import mayavi.mlab as mlab

def load_pc(f):
    b = np.fromfile(f, dtype=np.float32)
    return b.reshape((-1, 4))


def sparsify_pointclouds(points, print_distribution=False):
    # Converting to sparser input
    print("Number of points in a HDL64 PC: ", points.shape)

    # Calculate the original phi angles in degrees
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    phi = np.degrees(np.arccos(z / r))
    phi = -phi+90

    if print_distribution:
        print("Phi angle range: ", np.ptp(phi))
        print("Phi angle min: ", np.min(phi))
        print("Phi angle max: ", np.max(phi))
        print("Phi shape: ", phi.shape)

    # Only keep values which belong to specific channels (those in the vlp 16)
    vlp16_angles = [15,13,11,9,7,5,3,1,-1,-3,-5,-7,-9,-11,-13,-15]

    upper_threshold = 12 # mid point is 11.45 degrees
    lower_threshold = 11

    previous_mask = np.logical_and((phi <= (15 - lower_threshold)), (phi >= (15 - upper_threshold)))
    for i in vlp16_angles:
        mask = np.logical_and((phi <= (i - lower_threshold)), (phi >= (i - upper_threshold)))
        previous_mask = np.logical_or(mask, previous_mask)

    sparser_points = points[previous_mask]
    return sparser_points


def vis_mlab(points, i):
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 480))
    mlab.points3d(points[:, 0],  # x
                  points[:, 1],  # y
                  points[:, 2],  # z
                  points[:, 2],  # Height data used for shading
                  mode="point",  # How to render each point {'point', 'sphere' , 'cube' }
                  colormap='spectral',  # 'bone', 'copper',
                  # color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
                  scale_factor=100,  # scale of the points
                  line_width=10,  # Scale of the line, if any
                  figure=fig,
                  )
    mlab.savefig(filename='my_bev/test/test'+str(i)+'.png')
