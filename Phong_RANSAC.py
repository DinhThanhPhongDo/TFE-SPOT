import numpy as np
from scipy.linalg import norm
import open3d as o3d
import random 



def ransac(pts, thresh=0.05,epoch=1000) :
    """
    points = np.array(N,3)
    """
    n_pts    = len(pts)
    idx_pts  = list(np.arange(0,n_pts,1))
    n_inliers = 0
    i =0
    while (i < epoch) :
        # TODO Générer un plan à partir de curr_pnt
        # TODO Inlier/outlier 

        pts_sample = pts[random.sample(idx_pts,3)]
    
        vecA = pts_sample[1, :] - pts_sample[0, :]
        vecB = pts_sample[2, :] - pts_sample[0, :]

        normal = np.cross(vecA, vecB)
        d      = -np.dot(normal,pts_sample[0,:])
        norm_normal = norm(normal)

        plane = np.array([normal[0], normal[1], normal[2], d])
    
        dist_pt     = (np.dot(normal[:3],pts_sample.T)) / norm_normal
        inlier_mask = np.less_equal(dist_pt,thresh)

        if sum(inlier_mask)> n_inliers:
            best_plane = plane
            n_inliers = sum(inlier_mask)
        i += 1

    return plane



if __name__ =='__main__':
    pts = o3d.io.read_point_cloud(r'hough-plane-python-master\RES\map_go_5.pcd')
    pts = pts.voxel_down_sample(voxel_size=0.1)
    points = np.asarray(pts.points)
    plane = ransac(points)
    print(plane)

