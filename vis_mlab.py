import numpy as np
import mayavi.mlab as mlab
import math

def load_pc(f):
    b = np.fromfile(f, dtype=np.float32)
    return b.reshape((-1, 4))

pc = load_pc('/home/moemen/Work/3d_detection_utils/3d_detection_kit/data/testing/velodyne/000003.bin')
hdl_velo = pc[:, :3]
print("Number of points in a HDL64 PC: ", hdl_velo.shape)

# Visualise the original pointcloud
fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 480))
mlab.points3d(  hdl_velo[:, 0],   # x
                hdl_velo[:, 1],   # y
                hdl_velo[:, 2],   # z
                pc[:, 3],   # Height data used for shading
                mode="point", # How to render each point {'point', 'sphere' , 'cube' }
                colormap='spectral',  # 'bone', 'copper',
                #color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
                scale_factor=100,     # scale of the points
                line_width=10,        # Scale of the line, if any
                figure=fig,
            )

# Convert to sparser input
x = hdl_velo[:, 0]
y = hdl_velo[:, 1]
z = hdl_velo[:, 2]

# Calculate the original phi angles in degrees
r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
phi = np.degrees(np.arccos(z / r))
phi = -phi+90

print("Phi angle range", np.ptp(phi))
print("Phi angle min", np.min(phi))
print("Phi angle max", np.max(phi))
print("Phi shape: ", phi.shape)

# Only keep values which belong to specific channels (those in the vlp 16)
vlp16_angles = [15,13,11,9,7,5,3,1,-1,-3,-5,-7,-9,-11,-13,-15]

previous_mask = np.logical_and((phi <= (15 - 17.7)), (phi >= (15 - 18.3)))
for i in vlp16_angles:
    mask = np.logical_and((phi <= (i - 17.7)), (phi >= (i - 18.3)))
    previous_mask = np.logical_or(mask, previous_mask)

pc1 = pc[previous_mask]
vlp16_velo = hdl_velo[previous_mask]
print("Number of points in a Pseudo-VLP16 PC: ", vlp16_velo.shape)

# Visualise the corrected pointcloud
fig1 = mlab.figure(bgcolor=(0, 0, 0), size=(640, 480))
mlab.points3d(  vlp16_velo[:, 0],   # x
                vlp16_velo[:, 1],   # y
                vlp16_velo[:, 2],   # z
                pc1[:, 3],   # Height data used for shading
                mode="point", # How to render each point {'point', 'sphere' , 'cube' }
                colormap='spectral',  # 'bone', 'copper',
                #color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
                scale_factor=100,     # scale of the points
                line_width=10,        # Scale of the line, if any
                figure=fig1,
             )

# velo[:, 3], # reflectance values
mlab.show()