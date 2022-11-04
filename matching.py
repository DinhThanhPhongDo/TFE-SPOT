import copy
import open3d as o3d

pcd_before = o3d.io.read_point_cloud("TFE-SPOT/TestData/test_before/pcd/map_go_5.pcd")
pcd_after = o3d.io.read_point_cloud("TFE-SPOT/TestData/test_after/pcd/map_go_5.pcd")


def draw_registration_result(source, target, transformation = None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation : source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

draw_registration_result(pcd_before, pcd_after)