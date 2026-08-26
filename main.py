import numpy as np
import open3d as o3d
import plotly.express as plt
import point_cloud_utils as pcu
import os
import skeletor as sk

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

def show_coded_dataset () :
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

def obj_to_npy (obj_in, path_out, npy_out) :
    pts = np.asarray(obj_in.vertices)
    np.save(f"{path_out}/{npy_out}", pts)

def normalise_pts (pts_in) :
    mean_ = np.mean(pts_in, axis = 0)
    pts_in -= mean_
    furthest_distance = np.max(np.sqrt(np.sum(abs(pts_in)**2,axis =-1)))
    pts_in /= furthest_distance
    return pts_in

def batch_normalise_objs (dir_in) :
    objs = os.listdir(dir_in)

    for fname in objs :
        if not fname.endswith(".obj") :
            continue
        v, f = pcu.load_mesh_vf(f"{dir_in}/{fname}")
        fname = fname[: -4]
        v = normalise_pts(v)
        fname += "_norm"
        pcu.save_mesh_vf(f"{dir_in}/{fname}.ply", v, f)

def batch_export_objs (dir_in, dir_out, normalise = False) :
    objs = os.listdir(dir_in)

    for fname in objs :
        if not fname.endswith(".obj") :
            continue
        obj = pcu.load_mesh_v(f"{dir_in}/{fname}")
        fname = fname[: -4]
        if normalise :
            obj = normalise_pts(obj)
            fname += "_norm"

        np.save(f"{dir_out}/{fname}", obj)

def batch_load_datasets (dir_in) -> list :
    files = os.listdir(dir_in)
    out_list = []
    for fname in files:
        if not fname.endswith("norm.ply") or not fname.startswith("mock") :
            continue
        ply = pcu.load_mesh_vf(f"{dir_in}/{fname}")
        out_list.append(ply)

    return out_list

def batch_load_truth_skels (dir_in) -> list :
    files = os.listdir(dir_in)
    out_list = []
    for fname in files:
        if not fname.endswith("norm.ply") or not fname.startswith("skel") :
            continue
        ply = pcu.load_mesh_v(f"{dir_in}/{fname}")
        out_list.append(ply)

    return out_list

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

def extract_max_min_haus_chamf (res_, verbose = False) -> dict :
    max_h_key = "0"
    max_c_key = "0"
    min_h_key = "0"
    min_c_key = "0"
    max_h = res_[max_h_key]["Hausdorf"]
    min_h = res_[min_h_key]["Hausdorf"]
    max_c = res_[max_c_key]["Chamfer"]
    min_c = res_[min_c_key]["Chamfer"]
    for k in res_ :
        haus = res_[k]["Hausdorf"]
        chamf = res_[k]["Chamfer"]
        if haus > max_h:
            max_h = haus
            max_h_key = k
        elif haus < min_h :
            min_h = haus
            min_h_key = k    
        if chamf > max_c :
            max_c = chamf
            max_c_key = k
        elif chamf < min_c :
            min_c = chamf
            min_c_key = k

    if verbose :
        print(f'''max Hausdorf distance was achived by {res_[max_h_key]}\nmin Hausdorf distance was achieved by {res_[min_h_key]}
                \nmax chamfer distance was achieved by {res_[max_c_key]}\nmin chamfer distance was achieved by {res_[min_c_key]}''')
        
    return {"max_h" : max_h_key, "min_h" : min_h_key, "max_c" : max_c_key, "min_c" : min_c_key}

def visualise_skel_dataset (dataset_v, skel = None, g_truth_v = np.array([]), connect_skel = False) :
    pcd_dataset = o3d.geometry.PointCloud()
    pcd_dataset.points = o3d.utility.Vector3dVector(dataset_v)
    pcd_dataset.paint_uniform_color([0.7, 0.7, 0.7])
    to_vis = [pcd_dataset]

    if skel != None :
        pcd_skel = o3d.geometry.PointCloud()
        pcd_skel.points = o3d.utility.Vector3dVector(skel.vertices)
        pcd_skel.paint_uniform_color([1, 0, 0])
        to_vis.append(pcd_skel)

    if g_truth_v.size != 0 :
        pcd_gtruth = o3d.geometry.PointCloud()
        pcd_gtruth.points = o3d.utility.Vector3dVector(g_truth_v)
        pcd_gtruth.paint_uniform_color([0, 0, 1])
        to_vis.append(pcd_gtruth)

    if connect_skel :
        skel_lines = o3d.geometry.LineSet()
        skel_lines.points = o3d.utility.Vector3dVector(skel.vertices)
        skel_lines.lines = o3d.utility.Vector2iVector(skel.edges)
        cols = [[1, 0, 0]] * (len(skel.vertices) - 1)
        skel_lines.colors = o3d.utility.Vector3dVector(cols)
        to_vis.append(skel_lines)

    o3d.visualization.draw_geometries(to_vis)

def wavefront_pipeline_one_dataset (dataset_to_skeletonise, truth, wave_vals_to_try, step_size_vals_to_try) :
    fixed = sk.pre.fix_mesh(dataset_to_skeletonise, remove_disconnected = 5, inplace = False)
    # ___!!___
    truth = np.asfortranarray(truth)    # so pcu can calculate NNs, whatever skeletor returns
    # is also F_CONIGOUS : True so making this match
    truth = np.asarray(truth, dtype=np.float32)
    # by wavefront works well
    # edge collapse not so much
    # mean curvature look good too
    # tangent ball looks a bit better than mean curvature
    # teasar not good
    # vertex clusters giove similiar results to teasar
    skel = None
    n = 0
    res = dict()
    for i in range(1, wave_vals_to_try) :
        for j in range(1, step_size_vals_to_try) :
            skel = sk.skeletonize.by_wavefront(fixed, waves = i, step_size  = j)
            skel = sk.post.clean_up(skel)   # this one seems to lower the chamfer distance by a little bit
            skel = sk.post.smooth(skel) # this one not so much
            verts = skel.vertices
            verts = np.asarray(skel.vertices, dtype=np.float32)
            chamf = pcu.chamfer_distance(verts, truth)
            chamf = float(chamf)
            haus = pcu.hausdorff_distance(verts, truth)
            # print(f"Iteration {n + 1}: waves param = {i}, step size param = {j}")
            # print("Chamfer Distance:", chamf)
            # print("Hausdorf Distance:", haus)
            # print("------")
            res[str(n)] = {"Chamfer" : chamf, "Hausdorf" : haus, "no. waves" : i, "step size" : j, "skeleton" : skel}
            n += 1

    # extract max and min chamfer and hausdorf values
    min_max_c_h = extract_max_min_haus_chamf(res)
    skel_with_max_hausdorff = min_max_c_h["max_h"]
    skel_with_min_hausdorff = min_max_c_h["min_h"]
    skel_with_max_chamfer = min_max_c_h["max_c"]
    skel_with_min_chamfer = min_max_c_h["min_c"]
    print(res[skel_with_min_chamfer])

    # plot max and min hausdorf and max and min chamfer
    visualise_skel_dataset(dataset_to_skeletonise[0], skel=res[skel_with_min_chamfer]["skeleton"], g_truth_v=truth)
    print("------")

def tangent_ball_pipeline_one_dataset (dataset_to_skeletonise, truth) :
    fixed = sk.pre.fix_mesh(dataset_to_skeletonise, fix_normals=True, remove_disconnected = 5, inplace = False)
    # ___!!___
    truth = np.asfortranarray(truth)    # so pcu can calculate NNs, whatever skeletor returns
    # is also F_CONIGOUS : True so making this match
    truth = np.asarray(truth, dtype=np.float32)
    # by wavefront works well
    # edge collapse not so much
    # mean curvature look good too
    # tangent ball looks a bit better than mean curvature
    # teasar not good
    # vertex clusters giove similiar results to teasar
    skel = sk.skeletonize.by_tangent_ball(fixed)
    skel = sk.post.clean_up(skel)   # this one seems to lower the chamfer distance by a little bit
    skel = sk.post.smooth(skel) # this one not so much
    verts = skel.vertices
    verts = np.asarray(skel.vertices, dtype=np.float32)
    chamf = pcu.chamfer_distance(verts, truth)
    chamf = float(chamf)
    haus = pcu.hausdorff_distance(verts, truth)
    # print(f"Iteration {n + 1}: waves param = {i}, step size param = {j}")
    # print("Chamfer Distance:", chamf)
    # print("Hausdorf Distance:", haus)
    # print("------")
    print("Chamfer", chamf, "Hausdorf", haus, "skeleton", skel)

    # plot max and min hausdorf and max and min chamfer
    visualise_skel_dataset(dataset_to_skeletonise[0], skel=skel, g_truth_v=truth)
    print("------")

def main () :
    # ___exporting all the .objs into .npys and saving so can be saved on OneDrive___
    data_path = "C:/Users/vdwq25/data"
    data_path_out = "C:/Users/vdwq25/data/npy"

    #batch_export_objs(data_path, data_path_out)
    #batch_export_objs(data_path, data_path_out, normalise=True)
    #batch_normalise_objs(data_path)
    ply_datasets = batch_load_datasets(data_path)
    ply_truth_skels = batch_load_truth_skels(data_path)
    to_dup = ply_truth_skels[1]
    ply_truth_skels.insert(2, to_dup)
    #print(ply_truth_skels)
    # those two are now parallel - dataset with index 0 has corresponding skeleton in the 
    # other list at the same index

    #dataset_to_skeletonise = pcu.load_mesh_vf(r"C:\Users\vdwq25\data\mock_dataset_complex_holes_norm.ply")
    truth = pcu.load_mesh_v(r"C:\Users\vdwq25\data\skel_complex_branching_norm.ply")

    # for i in range(len(ply_datasets)) :
    #     wavefront_pipeline_one_dataset(ply_datasets[i], ply_truth_skels[i], 10, 7)

    for i in range(len(ply_datasets)) :
        tangent_ball_pipeline_one_dataset(ply_datasets[i], ply_truth_skels[i])

if __name__ == "__main__" :
    main()
