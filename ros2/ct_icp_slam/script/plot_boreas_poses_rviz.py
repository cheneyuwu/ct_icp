import os
import os.path as osp
import sys
import time
import numpy as np
import numpy.linalg as npla
import scipy.spatial.transform as sptf
import csv

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.time_source import CLOCK_TOPIC
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import rosgraph_msgs.msg as rosgraph_msgs

np.set_printoptions(precision=6, suppress=True)


def pose2tfstamped(pose, stamp, to_frame, from_frame):
  tran = pose[:3, 3]
  rot = sptf.Rotation.from_matrix(pose[:3, :3]).as_quat()

  tfs = geometry_msgs.TransformStamped()
  # The default (fixed) frame in RViz is called 'world'
  tfs.header.frame_id = to_frame
  tfs.header.stamp = stamp
  tfs.child_frame_id = from_frame
  tfs.transform.translation.x = tran[0]
  tfs.transform.translation.y = tran[1]
  tfs.transform.translation.z = tran[2]
  tfs.transform.rotation.x = rot[0]
  tfs.transform.rotation.y = rot[1]
  tfs.transform.rotation.z = rot[2]
  tfs.transform.rotation.w = rot[3]
  return tfs


def roll(r):
  return np.array([[1, 0, 0], [0, np.cos(r), np.sin(r)], [0, -np.sin(r), np.cos(r)]], dtype=np.float64)


def pitch(p):
  return np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]], dtype=np.float64)


def yaw(y):
  return np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]], dtype=np.float64)


def yawPitchRollToRot(y, p, r):
  return roll(r) @ pitch(p) @ yaw(y)


def get_inverse_tf(T):
  """Returns the inverse of a given 4x4 homogeneous transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: inv(T)
    """
  T2 = T.copy()
  T2[:3, :3] = T2[:3, :3].transpose()
  T2[:3, 3:] = -1 * T2[:3, :3] @ T2[:3, 3:]
  return T2


def convert_line_to_pose(line, dim=3):
  """Reads trajectory from list of strings (single row of the comma-separeted groundtruth file). See Boreas
    documentation for format
    Args:
        line (List[string]): list of strings
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (np.ndarray): 4x4 SE(3) pose
        (int): time in nanoseconds
    """
  # returns T_iv
  line = line.replace('\n', ',').split(',')
  line = [float(i) for i in line[:-1]]
  # x, y, z -> 1, 2, 3
  # roll, pitch, yaw -> 7, 8, 9
  T = np.eye(4, dtype=np.float64)
  T[0, 3] = line[1]  # x
  T[1, 3] = line[2]  # y
  if dim == 3:
    T[2, 3] = line[3]  # z
    T[:3, :3] = yawPitchRollToRot(line[9], line[8], line[7])
  elif dim == 2:
    T[2, 3] = 0
    T[:3, :3] = yawPitchRollToRot(line[9], np.round(line[8] / np.pi) * np.pi, np.round(line[7] / np.pi) * np.pi)
  else:
    raise ValueError('Invalid dim value in convert_line_to_pose. Use either 2 or 3.')
  time = int(line[0])
  return T, time


def enforce_orthog(T, dim=3):
  """Enforces orthogonality of a 3x3 rotation matrix within a 4x4 homogeneous transformation matrix.
    Args:
        T (np.ndarray): 4x4 transformation matrix
        dim (int): dimensionality of the transform 2==2D, 3==3D
    Returns:
        np.ndarray: 4x4 transformation matrix with orthogonality conditions on the rotation matrix enforced.
    """
  if dim == 2:
    if abs(np.linalg.det(T[0:2, 0:2]) - 1) < 1e-10:
      return T
    R = T[0:2, 0:2]
    epsilon = 0.001
    if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
      print("WARNING: this is not a proper rigid transformation:", R)
      return T
    a = (R[0, 0] + R[1, 1]) / 2
    b = (-R[1, 0] + R[0, 1]) / 2
    s = np.sqrt(a**2 + b**2)
    a /= s
    b /= s
    R[0, 0] = a
    R[0, 1] = b
    R[1, 0] = -b
    R[1, 1] = a
    T[0:2, 0:2] = R
  if dim == 3:
    if abs(np.linalg.det(T[0:3, 0:3]) - 1) < 1e-10:
      return T
    c1 = T[0:3, 1]
    c2 = T[0:3, 2]
    c1 /= np.linalg.norm(c1)
    c2 /= np.linalg.norm(c2)
    newcol0 = np.cross(c1, c2)
    newcol1 = np.cross(c2, newcol0)
    T[0:3, 0] = newcol0
    T[0:3, 1] = newcol1
    T[0:3, 2] = c2
  return T


def read_traj_file_gt(path, T_ab, dim):
  """Reads trajectory from a comma-separated file, see Boreas documentation for format
    Args:
        path (string): file path including file name
        T_ab (np.ndarray): 4x4 transformation matrix for calibration. Poses read are in frame 'b', output in frame 'a'
        dim (int): dimension for evaluation. Set to '3' for 3D or '2' for 2D
    Returns:
        (List[np.ndarray]): list of 4x4 poses
        (List[int]): list of times in microseconds
    """
  with open(path, 'r') as f:
    lines = f.readlines()
  poses = []
  times = []

  T_ab = enforce_orthog(T_ab)
  for line in lines[1:]:
    pose, time = convert_line_to_pose(line, dim)
    poses += [enforce_orthog(T_ab @ get_inverse_tf(pose))]  # convert T_iv to T_vi and apply calibration
    times += [int(time)]  # microseconds
  return poses, times


def poses2path(T_0t_list, stamp, frame):
  paths = nav_msgs.Path()
  paths.header.frame_id = frame
  paths.header.stamp = stamp
  for T_0t in T_0t_list:
    pose_msg = geometry_msgs.PoseStamped()
    tran = T_0t[:3, 3]
    rot = sptf.Rotation.from_matrix(T_0t[:3, :3]).as_quat()
    pose_msg.pose.position.x = tran[0]
    pose_msg.pose.position.y = tran[1]
    pose_msg.pose.position.z = tran[2]
    pose_msg.pose.orientation.x = rot[0]
    pose_msg.pose.orientation.y = rot[1]
    pose_msg.pose.orientation.z = rot[2]
    pose_msg.pose.orientation.w = rot[3]
    paths.poses.append(pose_msg)
  return paths


def main(args=None):
  rclpy.init(args=args)
  node = Node("boreas_plotter")

  # dataset_dir = '/home/yuchen/ASRL/data/BOREAS'
  seq = 'boreas-2022-05-13-10-30'
  dataset_dir = '/home/yuchen/ASRL/data/boreas/sequences'
  result_dirs = [
    "/home/yuchen/ASRL/temp/cticp/BOREAS/boreas_odometry_result",
    "/media/yuchen/T7/ASRL/temp/lidar/boreas/" + seq + ".icp/odometry_result",
  ]

  T_applanix_lidar = np.loadtxt(osp.join(dataset_dir, seq, 'calib', 'T_applanix_lidar.txt'))
  filepath = os.path.join(dataset_dir, seq, 'applanix/lidar_poses.csv')
  T_aw_list, timestamps = read_traj_file_gt(filepath, T_applanix_lidar, 3)
  T_a0_at_list = []
  for T_aw in T_aw_list:
    T_a0_at_list.append(T_aw_list[0] @ get_inverse_tf(T_aw))

  T_a0_at_list = T_a0_at_list[::10]  # downsample
  path = poses2path(T_a0_at_list, Time(seconds=0).to_msg(), 'world')
  ground_truth_path_publisher = node.create_publisher(nav_msgs.Path, '/ground_truth_path', 10)
  ground_truth_path_publisher.publish(path)


  for i, result_dir in enumerate(result_dirs):
    filename = osp.join(result_dir, seq + '.txt')
    T_a0_at_list = []
    with open(filename, 'r') as file:
      for row in csv.reader(file, delimiter=' '):
        T_aw = np.eye(4)
        T_aw[:3] = np.copy(np.array(row[1:]).reshape(3, 4))  # row[0] is timestamp
        T_wa = get_inverse_tf(T_aw)
        T_a0_at_list.append(T_wa)
    T_a0_w = get_inverse_tf(T_a0_at_list[0])
    for j in range(len(T_a0_at_list)):
      T_a0_at_list[j] = T_a0_w @ T_a0_at_list[j]

    T_a0_at_list = T_a0_at_list[::10]  # downsample
    path = poses2path(T_a0_at_list, Time(seconds=0).to_msg(), 'world')
    result_path_publisher = node.create_publisher(nav_msgs.Path, '/result_path' + str(i), 10)
    result_path_publisher.publish(path)

  rclpy.shutdown()


if __name__ == '__main__':
  main()
