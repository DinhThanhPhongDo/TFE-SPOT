import numpy as np
import time
import sys
import open3d as o3d

from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, ODOM_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client.robot_state import RobotStateClient

def main():

  print('Connecting...')

  sdk = create_standard_sdk('CaptureDataClient')
  robot = sdk.create_robot('192.168.80.3')
  robot.authenticate('user', 'upsa43jm7vnf')
  robot.sync_with_directory()
  robot.time_sync.wait_for_sync()

  print('Connected')

  point_cloud_client = robot.ensure_client('velodyne-point-cloud')

  i = 0

  while input("Start ?") != "y":
    pass

  while True:

    start = time.perf_counter()

    point_clouds = point_cloud_client.get_point_cloud_from_sources(["velodyne-point-cloud"])

    data = np.frombuffer(point_clouds[0].point_cloud.data, dtype=np.float32)
    xyz = np.reshape(data, (len(data)//3, 3))

    point_cloud_transform = point_clouds[0].point_cloud.source.transforms_snapshot
    frame_name_sensor = point_clouds[0].point_cloud.source.frame_name_sensor
    
    sensor_tform_odom = get_a_tform_b(
      point_cloud_transform,
      "sensor",
      ODOM_FRAME_NAME
    )
    
    odom_tform_sensor = sensor_tform_odom.inverse()

    xyz = sensor_tform_odom.transform_cloud(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud("TFE-SPOT/TestData/sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("TFE-SPOT/TestData/sync.ply")
    o3d.visualization.draw_geometries([pcd])

    #np.save(f"point-clouds/snapshot_{i}.npy", xyz)

    #np.save(f"positions/odom_tform_sensor_{i}.npy", odom_tform_sensor.to_matrix())

    print(f"Snapshot {i} captured (in {time.perf_counter() - start} s)")

    if 0.2 - (time.perf_counter() - start) > 0:
      time.sleep(0.2 - (time.perf_counter() - start))

    i += 1

  return True

if __name__ == "__main__":
  if not main():
    sys.exit(1)
  sys.exit(0)


