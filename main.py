import numpy as np
import open3d as o3d
import plotly.express as plt
import math

# https://www.arnoldfw.com/pdf/3d_curves.pdf

# some of those might need swapping around
# for t [0, 6pi]
def make_spiral (t, a) :
    x = 4 * np.sin(t)
    y = 4 * np.cos(t)
    z = a ** (- t/10)
    return np.column_stack([x, y, z])

def make_wrapped_circle (t) :
    x = 2 * np.cos(t)
    y = 2 * np.sin(t)
    z = np.cos(2 * t)
    return np.column_stack([x, y, z])

def make_curve (t) :
    x = 2 * t**3 - 3 * t
    y = t**2 + 4 * t
    z = -t**3 + 2 * t**2 - t
    return np.column_stack([x, y, z])

def main () :
    ts = np.linspace(0, 6 * np.pi, 100)
    pts = make_spiral(ts, 10)
    x_ = [p[0] for p in pts]
    y_ = [p[1] for p in pts]
    z_ = [p[2] for p in pts]
    

    fig = plt.line_3d(x = x_, y = y_, z = z_, title = "squiggle")
    #fig.show()

    tss = np.linspace(0, 20, 100)
    ptss = make_curve(tss)
    xx_ = [p[0] for p in ptss]
    yy_ = [p[1] for p in ptss]
    zz_ = [p[2] for p in ptss]
    figg = plt.line_3d(x = xx_, y = yy_, z = zz_, title = "wave")
    #figg.show()

    # read in and show the skeleton made by the L1 thing
    l1_skel_pts = np.load("./skeleton_i4.npy")
    print(l1_skel_pts)
    # tube_pcd = o3d.io.read_point_cloud("./cropped_downsampled.pcd")
    # #print(type(l1_skel_pts))
    # #print(np.isnan(l1_skel_pts))
    # for p in l1_skel_pts :
    #     #print(p[0])
    #     if np.isnan(p[0]):
    #         print("found a nan")
    # print("now from the suposedly removed")
    # #l1_skel_pts = l1_skel_pts[~np.isnan(l1_skel_pts)]
    # skel_pcd = o3d.geometry.PointCloud()
    # skel_pcd.points = o3d.utility.Vector3dVector(l1_skel_pts)
    # o3d.visualization.draw_geometries([skel_pcd, tube_pcd])
if __name__ == "__main__" :
    main()
