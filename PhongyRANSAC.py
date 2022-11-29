import numpy as np
import open3d as o3d
from tqdm import tqdm
import time
# def ransac(pts,pts_n, thresh_d=0.05,thresh_n=0.8,epoch=1000, tqdm_bool=False) :
#     """
#     points = np.array(N,3)
#     """
#     n_pts    = len(pts)
#     idx_pts  = np.arange(0,n_pts,1)
#     best_inlier_mask  = None
#     best_outlier_mask = None
#     best_n_inliers = 0

#     iterator = range(epoch)
#     if tqdm_bool:
#         iterator = tqdm(iterator)
#     for _ in iterator :

#         pts_sample = pts[np.random.choice(idx_pts,3)]
    
#         vecA = pts_sample[1, :] - pts_sample[0, :]
#         vecB = pts_sample[2, :] - pts_sample[0, :]

#         normal      =  np.cross(vecA, vecB)
#         d           = -np.dot(normal,pts_sample[0,:])
#         norm_normal = np.linalg.norm(normal)

#         plane = np.array([normal[0], normal[1], normal[2], d])
    
#         dist_pt     = np.abs(np.dot(normal[:3],pts.T)+ d) / norm_normal

#         normal = normal/norm_normal
#         # TODO clustering ?

#         inlier_mask = np.less_equal(dist_pt,thresh_d)*np.greater_equal(np.abs(np.dot(pts_n,normal)),thresh_n)
#         outlier_mask = np.logical_xor(np.less_equal(dist_pt,thresh_d)*np.greater_equal(np.abs(np.dot(pts_n,normal)),thresh_n))
#         n_inliers = np.sum(inlier_mask)

#         if n_inliers> best_n_inliers:
#             best_plane        = plane
#             best_inlier_mask  = inlier_mask
#             best_outlier_mask = outlier_mask
#             best_n_inliers    = n_inliers

#     return best_plane,idx_pts[best_inlier_mask],idx_pts[best_outlier_mask]


def ransac(pts,pts_n, thresh_d=0.05,thresh_n=0.8,confidence_interval=0.95,plane_proportion=0.1, tqdm_bool=False) :
    """
    points = np.array(N,3)
    """
    n_pts    = len(pts)
    idx_pts  = np.arange(0,n_pts,1)
    best_inlier_mask  = None
    best_outlier_mask = None
    best_d = None
    best_n_inliers = 0
    #https://halshs.archives-ouvertes.fr/halshs-00264843/document
    eps = 1- plane_proportion
    alpha = confidence_interval
    epoch = int(np.log(1-alpha)/np.log(1-np.power(1-eps,3)))
    print('epoch=',epoch)

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
        inlier_mask = np.less_equal(dist_pt,thresh_d)*np.greater_equal(np.abs(np.dot(pts_n,normal)),thresh_n)

        n_inliers = np.sum(inlier_mask)

        if n_inliers> best_n_inliers:
            best_plane        = plane
            best_inlier_mask  = inlier_mask
            best_outlier_mask = np.less_equal(dist_pt,thresh_d)*np.logical_xor(np.less_equal(dist_pt,thresh_d),np.greater_equal(np.abs(np.dot(pts_n,normal)),thresh_n))
            best_n_inliers    = n_inliers

    return best_plane,idx_pts[best_inlier_mask],idx_pts[best_outlier_mask]

def get_all_planes(pcd, voxel_size = 0.1, n_inliers = 100, n_plane=1, display = False):
    pcd_plane = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_plane.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    

    planes = []
    inliers_lst = []
    

    while(True) :
        pts_n = np.asarray(pcd_plane.normals)
        pts   = np.asarray(pcd_plane.points)
        idx_pts = np.arange(0,len(pts),1)
        plane,inliers,best_outliers= ransac(pts,pts_n,thresh_n=0.95,thresh_d=0.1,plane_proportion=0.1,confidence_interval=0.95)
        inliers = idx_pts[inliers]
        # clustering ? Filtering inliers too far away?

        # mask = np.full(len(best_outliers),False)
        # for i,j in tqdm(enumerate(best_outliers)):
        #     [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[j], 0.2)
        #     p = np.asarray(idx)
        #     q = np.in1d(np.asarray(idx),inliers)
        #     r = np.sum(np.in1d(np.asarray(idx),inliers))/len(idx)
        #     mask[i] = np.sum(np.in1d(np.asarray(idx),inliers))/len(idx)>0.6
        # mask = np.full(len(best_outliers),False)
        # inliers = np.append(inliers,best_outliers[mask])
        # print(np.sum(mask))
        if len(inliers) < n_inliers :
            break

        inlier_cloud          = pcd_plane.select_by_index(inliers)
        inlier_cloud, inlier2 = inlier_cloud.remove_statistical_outlier(nb_neighbors=5,std_ratio=2.0)
        inlier_cloud.paint_uniform_color([np.random.uniform(0,0.5),np.random.uniform(0,1),np.random.uniform(0,1)]) 
        inliers = inliers[inlier2]#inliers[inlier2+mask]

        #add outlier that are inliers:
        
        outlier_cloud = pcd_plane.select_by_index(inliers, invert=True)
        outlier_cloud, outlier2 = outlier_cloud.remove_statistical_outlier(nb_neighbors=5,std_ratio=2.0)
        outlier_cloud.paint_uniform_color([1, 0, 0]) #rouge

        pcd_plane = outlier_cloud
        inliers_lst.append(inlier_cloud)
        planes.append(plane)
    if display :
        print('number of planes:',len(planes))
        o3d.visualization.draw_geometries(inliers_lst+[outlier_cloud], zoom=0.8,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])#+[outlier_cloud]
    return planes,inliers_lst

                                


if __name__ =='__main__':
    pcd = o3d.io.read_point_cloud("hough-plane-python-master\RES\map_go_5.pcd")    
    get_all_planes(pcd,n_inliers=1000, n_plane=20, display = True)

