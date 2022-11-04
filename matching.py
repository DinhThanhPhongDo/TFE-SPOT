import open3d as o3d

pcd_before = o3d.io.read_point_cloud("TFE-SPOT/TestData/test_before.pcd")
pcd_after = o3d.io.read_point_cloud("TFE-SPOT/TestData/test_after.pcd")

o3d.visualization.draw_geometries([pcd_before])