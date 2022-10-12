import numpy as np
import time
import sys
import open3d as o3d
import bosdyn

from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, ODOM_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2
import cv2

def main():
  

  print('Connecting...')

  sdk = create_standard_sdk('CaptureDataClient')
  robot = sdk.create_robot('192.168.80.3')
  robot.authenticate('user', 'upsa43jm7vnf')
  robot.sync_with_directory()
  robot.time_sync.wait_for_sync()

  print('Connected')

  point_cloud_client = robot.ensure_client('velodyne-point-cloud')
  
  image_client = robot.ensure_client(ImageClient.default_service_name)
  
  i = 0

  while input("Start ?") != "y":
    pass

  while True and i==0:

    start = time.perf_counter()

    point_clouds = point_cloud_client.get_point_cloud_from_sources(["velodyne-point-cloud"])
    
    data = np.frombuffer(point_clouds[0].point_cloud.data, dtype=np.float32)
    print(point_clouds[0].point_cloud.data)
    xyz = np.reshape(data, (len(data)//3, 3))
    point_cloud_transform = point_clouds[0].point_cloud.source.transforms_snapshot
    frame_name_sensor = point_clouds[0].point_cloud.source.frame_name_sensor
    
    sensor_tform_odom = get_a_tform_b(
      point_cloud_transform,
      "sensor",
      ODOM_FRAME_NAME
    )
    
    odom_tform_sensor = sensor_tform_odom.inverse()

    #xyz = sensor_tform_odom.transform_cloud(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud("TFE-SPOT/TestData/sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("TFE-SPOT/TestData/sync.ply")
    # o3d.visualization.draw_geometries([pcd_load])

    # np.save(f"point-clouds/snapshot_{i}.npy", xyz)
    # np.save(f"positions/odom_tform_sensor_{i}.npy", odom_tform_sensor.to_matrix())
    # print(f"Snapshot {i} captured (in {time.perf_counter() - start} s)")

    image_request = [
            build_image_request(source, pixel_format=image_pb2.Image.PixelFormat.keys()[0])
            for source in ["back_fisheye_image"]
        ]
    image_responses = image_client.get_image(image_request)
    for image in image_responses:
      img = np.frombuffer(image.shot.image.data, dtype=np.uint16)
      if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                try:
                    # Attempt to reshape array into a RGB rows X cols shape.
                    img = img.reshape((image.shot.image.rows, image.shot.image.cols, 1))
                except ValueError:
                    # Unable to reshape the image data, trying a regular decode.
                    img = cv2.imdecode(img, -1)
      else:
                img = cv2.imdecode(img, -1)
      image_saved_path = image.source.name
      image_saved_path = image_saved_path.replace(
                "/", '')  # Remove any slashes from the filename the image is saved at locally.
      
      cv2.imwrite(image_saved_path + ".png", img) 
              
    if 0.2 - (time.perf_counter() - start) > 0:
      time.sleep(0.2 - (time.perf_counter() - start))

    i += 1

  return True

if __name__ == "__main__":
  if not main():
    sys.exit(1)
  sys.exit(0)


