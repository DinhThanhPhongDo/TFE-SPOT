import numpy as np
import open3d as o3d
import math
import logging
import vizualization as viz
import matplotlib.pyplot as plt
logger = logging.getLogger('Hough Plane')

CAN_USE_TQDM = False
try:
    from tqdm import tqdm
    CAN_USE_TQDM = True
except:
    pass

def _calc_normals(fis_deg, thetas_deg):
    '''
    input: 
        fis_deg         : list of fi in degree
        thetas_deg      : list of theta in degree
    Compute 2d array of vectors with length 1,
    each representing direction at angles φ and θ in spherical coordinate system
    
    '''

    fis_len    = len(fis_deg)
    thetas_len = len(thetas_deg)

    normals = np.zeros((fis_len, thetas_len, 3), dtype=np.float64)

    fis_rad    = fis_deg    *(np.pi/180)
    thetas_rad = thetas_deg *(np.pi/180)

    for i in range(fis_len):
        fi = fis_rad[i]
        for j in range(thetas_len):
            theta = thetas_rad[j]
            normal = np.array([
                math.sin(theta) * math.cos(fi),
                math.sin(theta) * math.sin(fi),
                math.cos(theta)
            ])
            normals[i, j] = normal

    return normals

def _dot_prod(point,normals):
    '''For one point compute projections of this point to all vectors in array "normals (cartesian)"'''

    x,y,z = point
    xx,yy,zz = normals[:,:,0], normals[:,:,1], normals[:,:,2]
    dot = x*xx + y*yy + z*zz
    return dot

def _fi_theta_depth_to_point(fi, theta, depth):
    '''Reconstruct point back from parameter space to 3D Euclidian space'''

    normal = np.array([
        math.sin(theta) * math.cos(fi),
        math.sin(theta) * math.sin(fi),
        math.cos(theta)
    ])
    return normal * depth

def vectors_len(vectors):
    '''Get lengths for array of 3D vectors'''

    vectors_sqr = vectors**2
    vectors_sum = np.sum(vectors_sqr, axis=1)
    vectors_sqrt = vectors_sum**0.5
    return vectors_sqrt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def hough_planes(points, threshold, use_tqdm=True,
                 fi_step=1, fi_bounds=(0, 360),
                 theta_step=1, theta_bounds=(0, 180),
                 depth_steps=100, depth_bounds=(0, None), depth_start_step=3,
                 dbscan_eps=3, dbscan_min_points=5,
                 ):
    '''Detects planes in 3D point clouds using Hough Transform algorithm.

    Algorithm transforms 3D points (e. g. [1.0, -2.0, 0.0]) into parameter space
    with axes φ (fi), θ (theta), d.
    These 3 parameters represent planes in 3D space. Angles φ and θ define normal vector,
    and d defines distance from zero to plane, orthogonal to this vector.

    After filling accumulator, we clusterize it via dbscan algorithm.
    Then, for every cluster we find its center of mass to get plane representation
    and its size for comparsion with other clusters.

    ---
    General parameters:
    :param points: 3D points as numpy array with shape=(?, 3) and dtype=np.float
    :param threshold: Minimal value in accumulator cell.
    :param use_tqdm: This flag defines whether use tqdm or not for the slowest part of algorithm
    which is filling accumulator with values point by point.

    ---
    Parameters φ, θ and d make up a 3D tensor.
    You can specify bounds and accuracy along each axis:

    :param fi_step: Step in degrees along φ axis in parameter space
    :param fi_bounds: Pair of two values: lower and upper bound for φ axis in degrees

    :param theta_step: Step in degrees along θ axis in parameter space
    :param theta_bounds: Pair of two values: lower and upper bound for φ axis in degrees

    :param depth_steps: Number of values in accumulator along d axis
    :param depth_bounds: Pair of two values: lower and upper bound for d axis.
    By default lower bound is 0, and upper bound is computed from point cloud
    so that no one point would go out of accumulator bounds.
    However, this results in almost unused upper half of accumulator along axis d.

    :param depth_start_step: Hough space is not uniform: it's density proportional to 1/4π*d.
    That's why lower slices of accumulator along axis d usually contain a lot of mess.
    To get rid of it, set this parameter to some small value: 3..5 (default is 3).

    !!Attention!! This only needed if you set lower bound of depth_bounds to 0.
    Instead, set depth_start_step to 0 in order to not to lose meaningful data!

    ---
    Dbscan parameters. Parameters passed to open3d.geometry.PointCloud.cluster_dbscan():
    :param dbscan_eps: Minimal distance between points in cluster.
    :param dbscan_min_points: Minimal points in cluster.

    ---
    :return: Returns 2 objects:
    -   np.array shape=(?,4) of planes, each represented by 3D point (x,y,z) that belongs to plane,
        which is also a vector normal to that plane;
        and s - size of cluster in parameter space that was collapsed into that plane;
        resulting in [x,y,z,s] vector for each plane.
    -   points in accumulator which value (v) is above threshold.
        Points format is (?,4), where each point has format [φ, θ, d, v]
    '''

    assert(type(points) == np.ndarray)
    assert(len(points.shape) == 2 and points.shape[1] == 3)

    assert(threshold >= 0)

    assert(fi_step > 0)
    assert(theta_step > 0)

    assert(depth_steps > 0 and type(depth_steps) == int)
    assert(depth_bounds[0] >= 0)
    assert(depth_start_step >= 0 and type(depth_start_step) == int)


    fis        = np.arange(fi_bounds[0]   , fi_bounds[1]   , fi_step   )
    thetas     = np.arange(theta_bounds[0], theta_bounds[1], theta_step)
    fis_len    = len(fis)
    thetas_len = len(thetas)

    accum   = np.zeros([fis_len, thetas_len, depth_steps], dtype=np.int64)
    normals = _calc_normals(fis, thetas)

    depth_bounds = list(depth_bounds)

    if depth_bounds[0] > 0:
        #points = 3D points as numpy array with shape=(?, 3) and dtype=np.float
        mask = vectors_len(points) > depth_bounds[0] 
        points = points[mask]

    if depth_bounds[1] is None:
        # points = [[1,2,3],[4,5,6],[7,8,-9]]
        # print(np.max(points)) = 8 Erreur? il aurait peut être du mettre np.abs()?
        depth_bounds[1] = 2*np.max(np.abs(points))


    logger.debug(f'depth_bounds: {depth_bounds}')
    #depth_steps = nbre of value in accumulator

    #ex: total = [0,5], depth_bounds = [2,5]. I know i do depth_steps for the interval [2,5]. How many steps do I do for the interval [0,2]?
    depth_skipped_steps = depth_steps *((depth_bounds[0] - 0) / (depth_bounds[1] - depth_bounds[0])) 
    depth_total_steps   = depth_steps + depth_skipped_steps

    points_scaled       = points / depth_bounds[1] * depth_total_steps

    fi_idxes = np.zeros([fis_len, thetas_len], dtype=np.int64)
    for i in range(len(fis)): #for i in range fis_len 
        fi_idxes[i] = i #fi_idxes[i,:]
    fi_idxes = fi_idxes.flatten()
    # exemple: 
    # fis_len =3 , thetas_len=2  => fi_idxes   = [0 0 1 1 2 2]
    #                            => theta_idxes= [0 1 0 1 0 1]

    #il aurait pu utiliser un scatter entre autre. ça aurait été plus simple.

    theta_idxes = np.zeros([fis_len, thetas_len], dtype=np.int64)
    for i in range(len(thetas)):
        theta_idxes[:, i] = i
    theta_idxes = theta_idxes.flatten()


    iterator = range(0, len(points))
    print("len points",len(points))
    if CAN_USE_TQDM and use_tqdm:
        iterator = tqdm(iterator)
    for k in iterator:
        # point \in [0, depth_step]
        point = points_scaled[k]
        #For one point compute projections of this point to all planes 
        dists = _dot_prod(point, normals) - depth_skipped_steps 

        dists = dists.astype(np.int64)
        dists = dists.flatten()

        mask = (dists >= 0) * (dists < depth_steps) #select potential planes

        fi_idxes_    = fi_idxes[mask]
        theta_idxes_ = theta_idxes[mask]
        dists        = dists[mask]

        #tout les phis et thetas qui conviennent à ce point
        accum[fi_idxes_, theta_idxes_, dists] += 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #on essaie de trouver les meilleurs points de l'accumulateur
    points_best = []
    for i in range(len(fis)):
        for j in range(len(thetas)):
            for k in range(depth_start_step, depth_steps):
                v = accum[i, j, k]
                if v >= threshold:
                    points_best.append([i, j, k, v]) 

    points_best = np.array(points_best)
    if len(points_best) == 0:
        logger.warning('Failed to find hough planes: all points below threshold = # element in a point in acumulator')
        return None, None


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_best[:,:3])
    #eps = Mininal distance between point in cluster
    #min_points = minimal number of points in a cluster
    cluster_idxes = pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points)
    labels = np.array( pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.455,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])

    clusters = {}
    for i in range(len(cluster_idxes)):
        idx = cluster_idxes[i]

        if not idx in clusters:
            clusters[idx] = []

        clusters[idx].append(points_best[i])
    # print('clusters',clusters[0][0])

    if -1 in clusters:
        del clusters[-1]
    if len(clusters.keys()) == 0:
        logger.warning('Failed to clusterize points in parameter space!')
        return None, points_best

    logger.debug('Detect clusters in parameter space')
    planes_out = []
    for k, v in clusters.items():
        logger.debug(f'~~~~~~~~~~~~{k}~~~~~~~~~~~~')
        cluster = np.array(v, dtype=np.int64)

        coords = cluster[:, :3] #[x,y,z]
        weights = cluster[:, 3] # value in the accumulator[]

        # \sum_i (xi*wi + yi*wi + zi*wi) / \sum (wi) (coordonée sphérique)
        for i in range(3):
            coords[:, i] *= weights

        cluster_size = len(weights)

        coord = np.sum(coords, axis=0) / np.sum(weights) 
        logger.debug(f'coord={coord}, cluster_size={cluster_size}')

        #recuperer 

        fi = (fi_bounds[0] + coord[0]*fi_step) / 180 * np.pi
        theta = (theta_bounds[0] + coord[1]*theta_step) / 180 * np.pi
        depth = (coord[2] + depth_skipped_steps) / depth_total_steps * depth_bounds[1]

        point = _fi_theta_depth_to_point(fi, theta,depth)
        logger.debug(f'fi,theta,depth = ({fi},{theta},{depth})')
        logger.debug(f'plane point: {point}')

        plane = np.concatenate([point, [cluster_size]])
        planes_out.append(plane)
    planes_out = np.array(planes_out)
    logger.debug(f'~~~~~~~~~~~~~~~~~~~~~~~~')

    for i in len(points_best):
        points_best[i,:,:]*=(180/np.pi)
        points_best[:,i,:]*=(180/np.pi)

    return planes_out, points_best


if __name__ =='__main__':
    pcd = o3d.io.read_point_cloud(r'TFE-SPOT\hough-plane-python-master\RES\map_go_5.pcd')
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    points = np.asarray(pcd.points)
    print('Number of points:', len(points))

    planes, points_best = hough_planes(points, threshold=500, fi_step=2, theta_step=2, depth_bounds=(0, 4))
    print('planes \n',planes)
    print('point best \n', points_best)
    viz.visualize_plane(points, planes[:,:3])
    viz.show_points(points_best, is_hough_space=True)
