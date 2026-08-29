import numpy as np
import open3d as o3d
import plotly.express as plt
import point_cloud_utils as pcu
import os
import skeletor as sk
import pandas as pd
import copy
import time

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

def center_pts (pts_in) :
    mean_ = np.mean(pts_in, axis = 0)
    pts_in -= mean_
    return pts_in

def batch_center_objs (dir_in) :
    objs = os.listdir(dir_in)

    for fname in objs :
        if not fname.endswith(".obj") :
            continue
        v, f = pcu.load_mesh_vf(f"{dir_in}/{fname}")
        fname = fname[: -4]
        v = center_pts(v)
        fname += "_centr"
        pcu.save_mesh_vf(f"{dir_in}/{fname}.ply", v, f)

def batch_export_objs (dir_in, dir_out, center = False) :
    objs = os.listdir(dir_in)

    for fname in objs :
        if not fname.endswith(".obj") :
            continue
        obj = pcu.load_mesh_v(f"{dir_in}/{fname}")
        fname = fname[: -4]
        if center :
            obj = center_pts(obj)
            fname += "_centr"

        np.save(f"{dir_out}/{fname}", obj)

def batch_load_datasets (dir_in) -> list :
    files = os.listdir(dir_in)
    out_list = []
    for fname in files:
        if not fname.endswith("centr.ply") or not fname.startswith("mock") :
            continue
        ply = pcu.load_mesh_vf(f"{dir_in}/{fname}")
        out_list.append(ply)

    return out_list

def batch_load_truth_skels (dir_in) -> list :
    files = os.listdir(dir_in)
    out_list = []
    for fname in files:
        if not fname.endswith("centr.ply") or not fname.startswith("skel") :
            continue
        ply = pcu.load_mesh_v(f"{dir_in}/{fname}")
        out_list.append(ply)

    return out_list

def dataset_names (dir_in) :
    files = os.listdir(dir_in)
    out_list = []
    for fname in files:
        if not fname.endswith("centr.ply") or not fname.startswith("mock") :
            continue  
        out_list.append(fname[: -10])

    return out_list

def convert_to_list_of_tuples (l : list, names_ : list) :
    out_list = []
    for i in range(len(l)) :
        out_list.append((l[i], names_[i]))
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

def extract_max_min_haus_chamf_mover (res_, verbose = False) -> dict :
    max_h_key = "0"
    max_c_key = "0"
    max_e_key = "0"
    min_h_key = "0"
    min_c_key = "0"
    min_e_key = "0"
    max_h = res_[max_h_key]["Hausdorff"]
    min_h = res_[min_h_key]["Hausdorff"]
    max_c = res_[max_c_key]["Chamfer"]
    min_c = res_[min_c_key]["Chamfer"]
    max_e = res_[max_e_key]["Earth Mover's"]
    min_e = res_[min_e_key]["Earth Mover's"]
    for k in res_ :
        haus = res_[k]["Hausdorff"]
        chamf = res_[k]["Chamfer"]
        e_move = res_[k]["Earth Mover's"]
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
        if e_move > max_e :
            max_e = e_move
            max_e_key = k
        elif e_move < min_e :
            min_e = e_move
            min_e_key = k

    if verbose :
        print(f'''max Hausdorff distance was achived by {res_[max_h_key]}\nmin Hausdorff distance was achieved by {res_[min_h_key]}
                \nmax Chamfer distance was achieved by {res_[max_c_key]}\nmin Chamfer distance was achieved by {res_[min_c_key]}
                \nmax Earth mover's distance was achieved by {res_[max_e_key]}\nmin Earth mover's distance was achieved by {res_[min_e_key]}''')
        
    return {"max_h" : max_h_key, "min_h" : min_h_key, "max_c" : max_c_key, "min_c" : min_c_key, "max_e" : max_e_key, "min_e" : min_e_key}

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

def save_output_as_visualisation (dataset_v, out_name, skel = None, g_truth_v = np.array([])) :
    pcd_dataset = o3d.geometry.PointCloud()
    pcd_dataset.points = o3d.utility.Vector3dVector(dataset_v)
    pcd_dataset.paint_uniform_color([0.7, 0.7, 0.7])

    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 600, height = 600)
    vis.add_geometry(pcd_dataset)
    v_ctrl = vis.get_view_control()
    R = pcd_dataset.get_rotation_matrix_from_xyz((np.pi / 2,0 ,0))
    pcd_dataset.rotate(R, center = pcd_dataset.get_center())

    if skel != None :
        pcd_skel = o3d.geometry.PointCloud()
        pcd_skel.points = o3d.utility.Vector3dVector(skel.vertices)
        pcd_skel.paint_uniform_color([1, 0, 0])
        R = pcd_skel.get_rotation_matrix_from_xyz((np.pi / 2,0 ,0))
        pcd_skel.rotate(R, center = pcd_skel.get_center())
        vis.add_geometry(pcd_skel)

    if g_truth_v.size != 0 :
        pcd_gtruth = o3d.geometry.PointCloud()
        pcd_gtruth.points = o3d.utility.Vector3dVector(g_truth_v)
        pcd_gtruth.paint_uniform_color([0, 0, 1])
        R = pcd_gtruth.get_rotation_matrix_from_xyz((np.pi / 2,0 ,0))
        pcd_gtruth.rotate(R, center = pcd_gtruth.get_center())
        vis.add_geometry(pcd_gtruth)
    
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"./vis/{out_name}.png")


def rotation (pcu_mesh, euler_angles) :
    msh = o3d.geometry.TriangleMesh()
    msh.vertices = o3d.utils.Vector3dVector(pcu_mesh[0])
    msh.faces = o3d.utils.Vector3dVector(pcu_mesh[1])
    R = msh.get_rotation_matrix_from_xyz(euler_angles)
    msh.rotate(R, center = msh.get_center())
    out = np.asarray(msh.points, dtype=np.float64)
    return np.asfortranarray(out)

# def check_if_invartiant_under_iso_transform (dataset_vf) :
#     rotated

def wavefront_pipeline_one_dataset (dataset_to_skeletonise, truth, wave_vals_to_try, step_size_vals_to_try, show_prog=True, show_vis=True, verbose=False) -> dict :
    # extract name of the dataset and the vertices
    dataset = dataset_to_skeletonise[0]
    dataset_name = dataset_to_skeletonise[1]
    fixed = sk.pre.fix_mesh(dataset, remove_disconnected = 5, inplace = False)
    # ___!!___
    truth = np.asfortranarray(truth)    # so pcu can calculate NNs, whatever skeletor returns
    # is also F_CONIGOUS : True so making this match
    truth = np.asarray(truth, dtype=np.float64)
    skel = None
    n = 0
    res = dict()
    optimal_cycle = dict()
    for i in wave_vals_to_try :
        for j in step_size_vals_to_try :
            s = time.process_time()
            skel = sk.skeletonize.by_wavefront(fixed, waves = i, step_size  = j, progress=show_prog)
            algorithm_run_time = time.process_time() - s
            skel = sk.post.clean_up(skel)   # this one seems to lower the chamfer distance by a little bit
            skel = sk.post.smooth(skel) # this one not so much
            verts = skel.vertices
            verts = np.asarray(skel.vertices, dtype=np.float64)
            chamf = pcu.chamfer_distance(verts, truth)
            chamf = float(chamf)
            haus = pcu.hausdorff_distance(verts, truth)
            e_mover = pcu.earth_movers_distance(verts, truth)
            e_mover = float(e_mover[0])
            chamf = round(chamf, 5)
            haus = round(haus, 5)
            e_mover = round(e_mover, 5)
            algorithm_run_time = round(algorithm_run_time, 5)
            if verbose :
                print(f"Iteration {n + 1}, on dataset {dataset_name}: waves param = {i}, step size param = {j}")
                print("Chamfer Distance:", chamf)
                print("Hausdorf Distance:", haus)
                print("Earth Mover's Distace:", e_mover)
                print("------")
            res[str(n)] = {"on dataset" : dataset_name, "Chamfer" : chamf, "Hausdorff" : haus, "Earth Mover's" : e_mover, "no. waves" : i, "step size" : j, "skeleton" : skel, "runtime" : algorithm_run_time}
            n += 1

    # extract max and min chamfer, hausdorf and E. Mover values
    min_max_c_h_e = extract_max_min_haus_chamf_mover(res)
    skel_with_max_hausdorff = min_max_c_h_e["max_h"]
    skel_with_min_hausdorff = min_max_c_h_e["min_h"]
    skel_with_max_chamfer = min_max_c_h_e["max_c"]
    skel_with_min_chamfer = min_max_c_h_e["min_c"]
    skel_with_max_emover = min_max_c_h_e["max_e"]
    skel_with_min_emover = min_max_c_h_e["min_e"]
    print(res[skel_with_min_emover])
    optimal_cycle = copy.deepcopy(res[skel_with_min_emover])
    optimal_cycle["no. skeleton vertices"] = optimal_cycle["skeleton"].vertices.shape[0]
    optimal_cycle.pop("skeleton")

    # plot max and min hausdorf and max and min chamfer
    if show_vis :
        visualise_skel_dataset(dataset[0], skel=res[skel_with_min_emover]["skeleton"], g_truth_v=truth)
    print("------")
    save_output_as_visualisation(dataset[0], f"{dataset_name}_wavefront", skel=skel, g_truth_v=truth)

    return optimal_cycle

def tangent_ball_pipeline_one_dataset (dataset_to_skeletonise, truth, show_prog=True, show_vis=True, verbose=False) -> dict :
    # extract name of the dataset and the vertices
    dataset = dataset_to_skeletonise[0]
    dataset_name = dataset_to_skeletonise[1]
    fixed = sk.pre.fix_mesh(dataset, fix_normals=True, remove_disconnected = 5, inplace = False)
    # ___!!___
    truth = np.asfortranarray(truth)    # so pcu can calculate NNs, whatever skeletor returns
    # is also F_CONTIGOUS : True so making this match
    truth = np.asarray(truth, dtype=np.float64)
    s = time.process_time()
    skel = sk.skeletonize.by_tangent_ball(fixed)
    algorithm_run_time = time.process_time() - s
    skel = sk.post.clean_up(skel)   # this one seems to lower the chamfer distance by a little bit
    skel = sk.post.smooth(skel) # this one not so much
    verts = skel.vertices
    verts = np.asarray(skel.vertices, dtype=np.float64)
    chamf = pcu.chamfer_distance(verts, truth)
    chamf = float(chamf)
    haus = pcu.hausdorff_distance(verts, truth)
    e_mover = pcu.earth_movers_distance(verts, truth)
    e_mover = float(e_mover[0])
    chamf = round(chamf, 5)
    haus = round(haus, 5)
    e_mover = round(e_mover, 5)
    algorithm_run_time = round(algorithm_run_time, 5)
    if verbose:
        print("Chamfer", chamf, "Hausdorff", haus, "Earth Mover's", e_mover, "skeleton", skel)

    # plot max and min hausdorf and max and min chamfer
    if show_vis :
        visualise_skel_dataset(dataset[0], skel=skel, g_truth_v=truth)
    print("------")
    save_output_as_visualisation(dataset[0], f"{dataset_name}_tangent_ball", skel=skel, g_truth_v=truth)

    return {"on dataset" : dataset_name, "Chamfer" : chamf, "Hausdorff" : haus, "Earth Mover's" : e_mover, "runtime" : algorithm_run_time, "no. skeleton vertices" : skel.vertices.shape[0]}

def mean_curvature_pipeline_one_dataset (dataset_to_skeletonise, truth, epsilons, collapse_factors, init_attraction_weights, show_prog = True, show_vis = True, verbose = False) :
    # extract name of the dataset and the vertices
    dataset = dataset_to_skeletonise[0]
    dataset_name = dataset_to_skeletonise[1]
    fixed = sk.pre.fix_mesh(dataset, remove_disconnected = 5, inplace = False)
    # ___!!___
    truth = np.asfortranarray(truth)    # so pcu can calculate NNs, whatever skeletor returns
    # is also F_CONIGOUS : True so making this match
    truth = np.asarray(truth, dtype=np.float64)
    skel = None
    n = 0
    res = dict()
    optimal_cycle = dict()
    for i in epsilons :
        for j in collapse_factors :
            for k in init_attraction_weights :
                s = time.process_time()
                skel = sk.skeletonize.by_mean_curvature(fixed, i, collapse_factor=j, WH0=k, progress=show_prog)
                algorithm_run_time = time.process_time() - s
                skel = sk.post.clean_up(skel)   
                skel = sk.post.smooth(skel)
                verts = skel.vertices
                verts = np.asarray(skel.vertices, dtype=np.float64)
                chamf = pcu.chamfer_distance(verts, truth)
                chamf = float(chamf)
                haus = pcu.hausdorff_distance(verts, truth)
                e_mover = pcu.earth_movers_distance(verts, truth)
                e_mover = float(e_mover[0])
                chamf = round(chamf, 5)
                haus = round(haus, 5)
                e_mover = round(e_mover, 5)
                i = round(i, 5)
                j = round(j, 5)
                k = round(k, 5)
                algorithm_run_time = round(algorithm_run_time, 5)
                if verbose :
                            print(f"Iteration {n + 1}: waves param = {i}, step size param = {j}")
                            print("Chamfer Distance:", chamf)
                            print("Hausdorff Distance:", haus)
                            print("Earth Mover's Distace:", e_mover)
                            print("------")
                res[str(n)] = {"on dataset" : dataset_name, "Chamfer" : chamf, "Hausdorff" : haus, "Earth Mover's" : e_mover, "epsilon" : i, "collapse factor" : j, "initial attraction weight" : k, "skeleton" : skel, "runtime" : algorithm_run_time}
                n += 1

    # extract max and min chamfer, hausdorf and E. Mover values
    min_max_c_h_e = extract_max_min_haus_chamf_mover(res)
    skel_with_max_hausdorff = min_max_c_h_e["max_h"]
    skel_with_min_hausdorff = min_max_c_h_e["min_h"]
    skel_with_max_chamfer = min_max_c_h_e["max_c"]
    skel_with_min_chamfer = min_max_c_h_e["min_c"]
    skel_with_max_emover = min_max_c_h_e["max_e"]
    skel_with_min_emover = min_max_c_h_e["min_e"]
    print(res[skel_with_min_emover])
    optimal_cycle = copy.deepcopy(res[skel_with_min_emover])
    optimal_cycle["no. skeleton vertices"] = optimal_cycle["skeleton"].vertices.shape[0]
    optimal_cycle.pop("skeleton")

    # plot max and min hausdorf and max and min chamfer
    if show_vis :
        visualise_skel_dataset(dataset[0], skel=res[skel_with_min_emover]["skeleton"], g_truth_v=truth)
    print("------")
    save_output_as_visualisation(dataset[0], f"{dataset_name}_mean_curvature", skel=skel, g_truth_v=truth)

    return optimal_cycle

# this one is fast however by nature of the algorithm the skeleton is on the surface of the tube
def teasar_pipeline_one_dataset (dataset_to_skeletonise, truth, inv_distances, show_prog = True, show_vis = True, verbose = False) :
    # extract name of the dataset and the vertices
    dataset = dataset_to_skeletonise[0]
    dataset_name = dataset_to_skeletonise[1]
    fixed = sk.pre.fix_mesh(dataset, remove_disconnected = 5, inplace = False)
    # ___!!___
    truth = np.asfortranarray(truth)    # so pcu can calculate NNs, whatever skeletor returns
    # is also F_CONIGOUS : True so making this match
    truth = np.asarray(truth, dtype=np.float64)
    skel = None
    n = 0
    res = dict()
    optimal_cycle = dict()
    for i in inv_distances :
        s = time.process_time()
        skel = sk.skeletonize.by_teasar(fixed, i, progress=show_prog)
        algorithm_run_time = time.process_time() - s
        skel = sk.post.clean_up(skel)   
        skel = sk.post.smooth(skel)
        verts = skel.vertices
        verts = np.asarray(skel.vertices, dtype=np.float64)
        chamf = pcu.chamfer_distance(verts, truth)
        chamf = float(chamf)
        haus = pcu.hausdorff_distance(verts, truth)
        e_mover = pcu.earth_movers_distance(verts, truth)
        e_mover = float(e_mover[0])
        chamf = round(chamf, 5)
        haus = round(haus, 5)
        e_mover = round(e_mover, 5)
        i = round(i, 5)
        algorithm_run_time = round(algorithm_run_time, 5)
        if verbose :
            print(f"Iteration {n + 1}: waves param = {i}, step size param = {j}")
            print("Chamfer Distance:", chamf)
            print("Hausdorf Distance:", haus)
            print("Earth Mover's Distace:", e_mover)
            print("------")
        res[str(n)] = {"on dataset" : dataset_name, "Chamfer" : chamf, "Hausdorff" : haus, "Earth Mover's" : e_mover, "invalidation distance" : i, "skeleton" : skel, "runtime" : algorithm_run_time}
        n += 1

   # extract max and min chamfer, hausdorf and E. Mover values
    min_max_c_h_e = extract_max_min_haus_chamf_mover(res)
    skel_with_max_hausdorff = min_max_c_h_e["max_h"]
    skel_with_min_hausdorff = min_max_c_h_e["min_h"]
    skel_with_max_chamfer = min_max_c_h_e["max_c"]
    skel_with_min_chamfer = min_max_c_h_e["min_c"]
    skel_with_max_emover = min_max_c_h_e["max_e"]
    skel_with_min_emover = min_max_c_h_e["min_e"]
    print(res[skel_with_min_emover])
    optimal_cycle = copy.deepcopy(res[skel_with_min_emover])
    optimal_cycle["no. skeleton vertices"] = optimal_cycle["skeleton"].vertices.shape[0]
    optimal_cycle.pop("skeleton")

    # plot max and min hausdorf and max and min chamfer
    if show_vis :
        visualise_skel_dataset(dataset[0], skel=res[skel_with_min_emover]["skeleton"], g_truth_v=truth)
    print("------")
    save_output_as_visualisation(dataset[0], f"{dataset_name}_teasar", skel=skel, g_truth_v=truth)

    return optimal_cycle

# the pipline in this one is slighly different
def vertex_clusters_pipeline_one_dataset (dataset_to_skeletonise, truth, samp_dist, epsilons, show_prog = True, show_vis = True, verbose = False) :
    dataset = dataset_to_skeletonise[0]
    dataset_name = dataset_to_skeletonise[1]
    fixed = sk.pre.fix_mesh(dataset, remove_disconnected = 5, inplace = False)
    # ___!!___
    truth = np.asfortranarray(truth)    # so pcu can calculate NNs, whatever skeletor returns
    # is also F_CONIGOUS : True so making this match
    truth = np.asarray(truth, dtype=np.float64)
    skel = None
    n = 0
    res = dict()
    optimal_cycle = dict()
    for i in samp_dist :
        for j in epsilons :
            s = time.process_time()
            cont = sk.pre.contract(fixed, j)    # contraction is necessary for this step
            # sample distance = i, this should be tuned based on the resolution of the mesh
            skel = sk.skeletonize.by_vertex_clusters(cont, i, progress=show_prog)
            algorithm_run_time = time.process_time() - s
            skel = sk.post.clean_up(skel)   
            skel = sk.post.smooth(skel)
            sk.post.radii(skel, method="knn")   # this method does not add radii automatically
            verts = skel.vertices
            verts = np.asarray(skel.vertices, dtype=np.float64)
            chamf = pcu.chamfer_distance(verts, truth)
            chamf = float(chamf)
            haus = pcu.hausdorff_distance(verts, truth)
            e_mover = pcu.earth_movers_distance(verts, truth)
            e_mover = float(e_mover[0])
            chamf = round(chamf, 5)
            haus = round(haus, 5)
            e_mover = round(e_mover, 5)
            i = round(i, 5)
            j = round(j, 5)
            algorithm_run_time = round(algorithm_run_time, 5)
            if verbose :
                print(f"Iteration {n + 1}: waves param = {i}, step size param = {j}")
                print("Chamfer Distance:", chamf)
                print("Hausdorf Distance:", haus)
                print("Earth Mover's Distace:", e_mover)
                print("------")
            res[str(n)] = {"on dataset" : dataset_name, "Chamfer" : chamf, "Hausdorff" : haus, "Earth Mover's" : e_mover, "sampling distance" : i, "contraction %" : j * 100, "skeleton" : skel, "runtime" : algorithm_run_time}
            n += 1

    # extract max and min chamfer, hausdorf and E. Mover values
    min_max_c_h_e = extract_max_min_haus_chamf_mover(res)
    skel_with_max_hausdorff = min_max_c_h_e["max_h"]
    skel_with_min_hausdorff = min_max_c_h_e["min_h"]
    skel_with_max_chamfer = min_max_c_h_e["max_c"]
    skel_with_min_chamfer = min_max_c_h_e["min_c"]
    skel_with_max_emover = min_max_c_h_e["max_e"]
    skel_with_min_emover = min_max_c_h_e["min_e"]
    print(res[skel_with_min_emover])
    optimal_cycle = copy.deepcopy(res[skel_with_min_emover])
    optimal_cycle["no. skeleton vertices"] = optimal_cycle["skeleton"].vertices.shape[0]
    optimal_cycle.pop("skeleton")

    # plot max and min hausdorf and max and min chamfer
    # this was removed from the visualisation call , g_truth_v=truth
    if show_vis :
        visualise_skel_dataset(dataset[0], skel=res[skel_with_min_emover]["skeleton"])
    print("------")
    save_output_as_visualisation(dataset[0], f"{dataset_name}_vertex_clusters", skel=skel, g_truth_v=truth)

    return optimal_cycle

def main () :
    data_path = "C:/Users/vdwq25/data"
    data_path_out = "C:/Users/vdwq25/data/npy"

    # run this line below if you made changes to the /data folder
    #batch_center_objs(data_path)
    ply_datasets = batch_load_datasets(data_path)
    ply_truth_skels = batch_load_truth_skels(data_path)
    to_dup = ply_truth_skels[1]
    ply_truth_skels.insert(2, to_dup)
    # load names of the datasets
    ply_datasets_names = dataset_names(data_path)
    # create a list that containst tuples (dataset vertices and faces, dataset name)
    named_datasets = convert_to_list_of_tuples(ply_datasets, ply_datasets_names)
    # named_datasets and ply_truth_skels are now parallel - dataset with index i has corresponding skeleton in the 
    # other list at the same index

    waves = range(1, 11)
    steps = range(1, 7)
    optimal_res_wavefront = []
    for i in range(len(ply_datasets)) :
        optm = wavefront_pipeline_one_dataset(named_datasets[i], ply_truth_skels[i], [2, 5, 6, 8], [1, 2], show_vis=False, show_prog=True)
        optimal_res_wavefront.append(optm)

    # df = pd.DataFrame(optimal_res_wavefront)
    # df.to_csv("./out/wavefront_statistics.csv", index=False)

    optimal_res_tangent_ball = []
    for i in range(len(ply_datasets)) :
        optm = tangent_ball_pipeline_one_dataset(named_datasets[i], ply_truth_skels[i], show_vis=False, verbose=True)
        optimal_res_tangent_ball.append(optm)
    
    # df = pd.DataFrame(optimal_res_tangent_ball)
    # df.to_csv("./out/tangent_ball_statistics.csv", index=False)

    optimal_res_mean_curvature = []
    epsilons_ = np.arange(0.05, 0.35, 0.05) 
    collapse_factors_ = np.arange(0.2, 0.55, 0.05)
    initial_attraction_weights = np.arange(0.25, 2.25, 0.25)
    for i in range(len(ply_datasets)) :
        if i == 2 : continue    # the algorithm refuses to work on a dataset with holes
        optim = mean_curvature_pipeline_one_dataset(named_datasets[i],
                                            ply_truth_skels[i],
                                            epsilons=[0.1, ],
                                            collapse_factors=collapse_factors_,
                                            init_attraction_weights=initial_attraction_weights,
                                            show_vis=False, 
                                            show_prog=True)
        optimal_res_mean_curvature.append(optim)

    # df = pd.DataFrame(optimal_res_mean_curvature)
    # df.to_csv("./out/mean_curvature_statistics.csv", index=False)

    optimal_res_teasar = []
    inv_dists = np.arange(0.1, 1.05, 0.05)
    for i in range(len(ply_datasets)) :
        optim = teasar_pipeline_one_dataset(named_datasets[i], ply_truth_skels[i], [0.1, 0.3], show_prog=False, show_vis = False)
        optimal_res_teasar.append(optim)

    # df = pd.DataFrame(optimal_res_teasar)
    # df.to_csv("./out/teasar_statistics.csv", index = False)

    optimal_res_vclusts = []
    sampling_distance = [0.2, 0.3, 0.5, 0.6, 0.9]
    epsilons_vclusts = np.arange(0.1, 0.3, 0.1)
    for i in range(len(ply_datasets)) :
        optim = vertex_clusters_pipeline_one_dataset(named_datasets[i], ply_truth_skels[i], [0.2, 0.3, 0.5, 0.6, 0.9], [0.1, 0.2], show_vis=False)
        optimal_res_vclusts.append(optim)

    # df = pd.DataFrame(optimal_res_vclusts)
    # df.to_csv("./out/vclusts_statistics.csv", index = False)

if __name__ == "__main__" :
    main()
