import numpy as np
from scipy.linalg import norm
import open3d as o3d
import random 
from tqdm import tqdm



def ransac(pts,pts_n, thresh_d=0.05,thresh_n=0.8,epoch=1000) :
    """
    points = np.array(N,3)
    """
    n_pts    = len(pts)
    idx_pts  = list(np.arange(0,n_pts,1))
    inlier_idx  = np.arange(0,n_pts,1)
    best_inlier_mask = None
    n_inliers = 0


    for i in tqdm(range(epoch)):

        pts_sample = pts[random.sample(idx_pts,3)]
    
        vecA = pts_sample[1, :] - pts_sample[0, :]
        vecB = pts_sample[2, :] - pts_sample[0, :]

        normal      =  np.cross(vecA, vecB)
        d           = -np.dot(normal,pts_sample[0,:])
        norm_normal = norm(normal)

        plane = np.array([normal[0], normal[1], normal[2], d])
    
        dist_pt     = np.abs(np.dot(normal[:3],pts.T)) / norm_normal

        
        normal = normal/norm_normal
        # o = np.dot(pts_n,normal)
        inlier1 = np.less_equal(dist_pt,thresh_d)
        inlier_mask = np.less_equal(dist_pt,thresh_d)*np.greater_equal(np.abs(np.dot(pts_n,normal)),thresh_n)

        a = np.sum(inlier1)
        b = np.sum(inlier_mask)

        if np.sum(inlier_mask)> n_inliers:
            best_plane        = plane
            best_inlier_mask  = inlier_mask
            n_inliers         = np.sum(inlier_mask)

    return best_plane,best_inlier_mask

def get_all_planes(pts,pts_n,n_plane,i_plane=0):
    if i_plane==n_plane:
        return
    
    plane,inlier_mask= ransac(pts,pts_n,thresh_n=0.9,thresh_d=0.1,epoch=1000)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pts_idx = np.arange(0,len(pts),1)
    inlier_idx = pts_idx[inlier_mask]
    inlier_cloud = pcd.select_by_index(inlier_idx)
    inlier_cloud.paint_uniform_color([0, 1, 0]) #vert
    outlier_cloud = pcd.select_by_index(inlier_idx, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0]) #rouge
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom=0.8,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
    get_all_planes(pts[np.logical_not(inlier_mask)],pts_n[np.logical_not(inlier_mask)],n_plane=n_plane,i_plane=i_plane+1)
    return

# def get_all_planeso3d(pts,pts_n,n_plane,i_plane=0):
#     if i_plane==n_plane:
#         return
    
#     plane,inlier_mask= ransac(pts,pts_n,thresh_n=0.9,thresh_d=0.1,epoch=1000)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts)
#     pts_idx = np.arange(0,len(pts),1)
#     inlier_idx = pts_idx[inlier_mask]
#     inlier_cloud = pcd.select_by_index(inlier_idx)
#     inlier_cloud.paint_uniform_color([0, 1, 0]) #vert
#     outlier_cloud = pcd.select_by_index(inlier_idx, invert=True)
#     outlier_cloud.paint_uniform_color([1, 0, 0]) #rouge
#     o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom=0.8,
#                                   front=[-0.4999, -0.1659, -0.8499],
#                                   lookat=[2.1813, 2.0619, 2.0999],
#                                   up=[0.1204, -0.9852, 0.1215])
#     get_all_planes(pts[np.logical_not(inlier_mask)],pts_n[np.logical_not(inlier_mask)],n_plane=n_plane,i_plane=i_plane+1)
#     return


                                


if __name__ =='__main__':
    pts = o3d.io.read_point_cloud(r'hough-plane-python-master\RES\map_go_5.pcd')
    pts = pts.voxel_down_sample(voxel_size=0.1)
    pts.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pts_n = np.asarray(pts.normals)
    pts= np.asarray(pts.points)

    get_all_planes(pts,pts_n,n_plane=5)
    # x = np.arange(0,10,1)
    # print(x)
    # print(np.greater_equal(x,3)*np.less_equal(x,5))

