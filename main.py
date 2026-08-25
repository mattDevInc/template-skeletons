import numpy as np
import open3d as o3d
import plotly.express as plt
import math
import point_cloud_utils as pcu

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
    x = t
    y = t**4
    z = t**3
    return np.column_stack([x, y, z])

def make_circle (r, t, c, v1, v2) :
    x = c[0] + r * np.cos(t) * v1[0] + r * np.sin(t) * v2[0]
    y = c[1] + r * np.cos(t) * v1[1] + r * np.sin(t) * v2[1]
    z = c[2] + r * np.cos(t) * v1[2] + r * np.sin(t) * v2[2]
    return np.column_stack([x, y, z])

def define_direction (curve_pts) :
    directed = []
    for p in range(len(curve_pts)) :
        if p == len(curve_pts) - 1 :
            v1 = curve_pts[p - 1]
            v2 = curve_pts[p]
        else :
            v1 = curve_pts[p]
            v2 = curve_pts[p + 1]

        direction_of_p = v1 - v2
        # note that if dir vector is not normalised we get ellipses rather than circles which could be useful in some cases
        normalised_dir = direction_of_p / np.linalg.norm(direction_of_p)
        directed.append(normalised_dir)

    return np.asarray(directed, dtype="float32")


def mock_dataset (skel_pts, dset_size) :
    dir_of_pts = define_direction(skel_pts)
    sample = np.random.choice(skel_pts.shape[0], dset_size, replace = False)
    circle_centers = skel_pts[sample]
    centers_direction = dir_of_pts[sample]
    # find pairs of orthogonal vectors for each of the direction vectors corresponding to the centers
    orthogonals = []
    for d in centers_direction :
        orthogonals.append(np.array([d[1], - d[0], 0]))

    v1s = np.asarray(orthogonals, dtype="float32")
    orthogonal_orthogonals = []
    for i in range(len(centers_direction)) :
        orthogonal_orthogonals.append(np.cross(centers_direction[i], v1s[i]))

    v2s = np.asarray(orthogonal_orthogonals, dtype="float32")

    # for random radiai create points of circles
    circs = np.empty([1, 3])
    for n in range(len(circle_centers)) :
        tsss = np.linspace(0, 2 * np.pi, 20)
        circ_radious = (2 - 0.3) * np.random.random_sample() + 0.3
        circle_pts = make_circle(circ_radious, tsss, circle_centers[n], v1s[n], v2s[n])
        #print(circle_pts)
        circs = np.concatenate((circs, circle_pts), axis = 0)
        
    circs = np.delete(circs, (0), axis = 0)

    # x_ = [p[0] for p in circs]
    # y_ = [p[1] for p in circs]
    # z_ = [p[2] for p in circs]
    
    # plt.line_3d(x = x_, y = y_, z = z_).show()
    return circs

def main () :
    ts = np.linspace(0, 6 * np.pi, 50)
    spiral_pts = make_spiral(ts, 1.8)
    x_ = [p[0] for p in spiral_pts]
    y_ = [p[1] for p in spiral_pts]
    z_ = [p[2] for p in spiral_pts]
    # pts is the skeleton, all the plt does is connect them and plot
    

    fig = plt.line_3d(x = x_, y = y_, z = z_, title = "squiggle")
    #fig.show()

    tss = np.linspace(0, 20, 20)
    curve_pts = make_curve(tss)
    xx_ = [p[0] for p in curve_pts]
    yy_ = [p[1] for p in curve_pts]
    zz_ = [p[2] for p in curve_pts]
    figg = plt.line_3d(x = xx_, y = yy_, z = zz_, title = "wave")
    #figg.show()

    tsss = np.linspace(0, 2 * np.pi, 20)
    circle_pts = make_circle(4, tsss, np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]))
    xxx_ = [p[0] for p in circle_pts]
    yyy_ = [p[1] for p in circle_pts]
    zzz_ = [p[2] for p in circle_pts]
    figgg = plt.line_3d(x = xxx_, y = yyy_, z = zzz_, title = "circle")
    #figgg.show()

    mck = mock_dataset(curve_pts, 20)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mck)
    #o3d.io.write_point_cloud("./mock_data.ply", pcd)
    np.save(r"C:\Users\Cobra Kai\python\skeletons-from-poincloud-working-copy\Data\mock_dataset_curve", mck)
    np.save(r"C:\Users\Cobra Kai\python\skeletons-from-poincloud-working-copy\Data\g_truth_skeleton_curve", curve_pts)


    # read in and show the skeleton made by the L1 thing
    # l1_skel_pts = np.load("./skeleton_i4.npy")
    # print(l1_skel_pts)
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

    # importing a point cloud via pcu saved as a .ply
    points = pcu.load_mesh_v("./apple_L1_skeleton.ply") # this is a numpy array shape (500,3)
    # for comparision all I need is two numpy arrays (N, 3)

    # chamfer distace calculation - average distance between points of two sets
    generated_L1_skeleton_of_spiral = np.load("./L1_skeletons/L1_skel_spiral_i4.npy")
    chamfer = pcu.chamfer_distance(spiral_pts, generated_L1_skeleton_of_spiral)
    #print(chamfer)
    # so Chamfer distance between set itself is 0, as expected
    #print(pcu.chamfer_distance(spiral_pts, spiral_pts))

if __name__ == "__main__" :
    main()
