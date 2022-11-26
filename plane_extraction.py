import numpy as np
import open3d as o3d
from tqdm import tqdm

def ransac(pts,pts_n, thresh_d=0.05,thresh_n=0.8,epoch=1000, tqdm_bool=False) :
    """
    points = np.array(N,3)
    """
    n_pts    = len(pts)
    idx_pts  = list(np.arange(0,n_pts,1))
    best_inlier_mask = None
    best_n_inliers = 0

    iterator = range(epoch)
    if tqdm_bool:
        iterator = tqdm(iterator)
    for _ in iterator :

        pts_sample = pts[np.random.choice(idx_pts,3)]
    
        vecA = pts_sample[1, :] - pts_sample[0, :]
        vecB = pts_sample[2, :] - pts_sample[0, :]

        normal      =  np.cross(vecA, vecB)
        d           = -np.dot(normal,pts_sample[0,:])
        norm_normal = np.linalg.norm(normal)

        plane = np.array([normal[0], normal[1], normal[2], d])
    
        dist_pt     = np.abs(np.dot(normal[:3],pts.T)+ d) / norm_normal

        normal = normal/norm_normal
        # TODO clustering ?
        inlier_mask = np.less_equal(dist_pt,thresh_d)*np.greater_equal(np.abs(np.dot(pts_n,normal)),thresh_n)
        n_inliers = np.sum(inlier_mask)

        if n_inliers> best_n_inliers:
            best_plane        = plane
            best_inlier_mask  = inlier_mask
            best_n_inliers    = n_inliers

    return best_plane,list(np.where(best_inlier_mask)[0])

def get_all_planes(pcd, voxel_size = 0.1, n_inliers = 100, n_plane=1, display = False):
    pcd_plane = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_plane.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    planes = []
    inliers_lst = []
    while(len(planes)<n_plane and len(pcd_plane.points) > 3) :
        pts_n = np.asarray(pcd_plane.normals)
        pts = np.asarray(pcd_plane.points)
        plane,inliers= ransac(pts,pts_n,thresh_n=0.9,thresh_d=0.1,epoch=1000)
        if len(inliers) < n_inliers :
            break
        inlier_cloud = pcd_plane.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([np.random.uniform(0,0.5),np.random.uniform(0,1),np.random.uniform(0,1)]) #vert
        outlier_cloud = pcd_plane.select_by_index(inliers, invert=True)
        outlier_cloud.paint_uniform_color([1, 0, 0]) #rouge
        pcd_plane = outlier_cloud
        inliers_lst.append(inlier_cloud)
        planes.append(plane)
    if display :
        o3d.visualization.draw_geometries(inliers_lst+[outlier_cloud], zoom=0.8,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])
    return planes,inliers_lst

                                


if __name__ =='__main__':
    pcd = o3d.io.read_point_cloud("TFE-SPOT\hough-plane-python-master\RES\map_go_5.pcd")    
    get_all_planes(pcd,n_inliers=1000, n_plane=99, display = True)
    # x = np.arange(0,10,1)
    # print(x)
    # print(np.greater_equal(x,3)*np.less_equal(x,5))

