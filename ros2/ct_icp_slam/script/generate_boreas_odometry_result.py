import os
import os.path as osp
import argparse
import numpy as np
import numpy.linalg as npla
import csv

from pyboreas import BoreasDataset

np.set_printoptions(suppress=True)

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

def get_T_applanix_sensor_func(sensor):
  def _func(sequence):
    if sensor == "aeva":
      T_applanix_aeva = sequence.calib.T_applanix_aeva
      print("T_applanix_aeva before:\n", T_applanix_aeva)
      # # this is a correction to the calibration
      # T_agt_apd = np.array([
      #     [0.995621, 0.002137, 0.09346, 0.002811],
      #     [-0.003235, 0.999928, 0.011597, -0.04655],
      #     [-0.093429, -0.011848, 0.995555, 0.128853],
      #     [0., 0., 0., 1.],
      # ])
      # T_applanix_aeva = T_agt_apd @ T_applanix_aeva
      # print("T_applanix_aeva after:\n", T_applanix_aeva)
      return T_applanix_aeva
    elif sensor == "velodyne":
      T_applanix_lidar = sequence.calib.T_applanix_lidar
      print("T_applanix_lidar before:\n", T_applanix_lidar)
      ## this is obtained from steam icp
      T_agt_apd = np.array([
          [0.999747, -0.019076, -0.011934, -0.007568],
          [0.019052,  0.999816, -0.002173, -0.01067 ],
          [0.011973,  0.001945,  0.999926, -0.183529],
          [0.      ,  0.,        0.,        1.      ],
      ])
      T_applanix_lidar = T_agt_apd @ T_applanix_lidar
      ## this is obtained from elastic icp
      # T_agt_apd = np.array([
      #     # [0.998827, -0.019172, -0.044469,  0.003457],
      #     # [0.019183,  0.999816, -0.00018 , -0.037566],
      #     # [0.044465, -0.000673,  0.999011, -0.107936],
      #     # [0.      ,  0.      ,  0.      ,  1.      ],
      #     # [0.999552, -0.000038, -0.029945,  0.005274],
      #     # [0.000097,  0.999998,  0.001969, -0.027022],
      #     # [0.029945, -0.001971,  0.99955 ,  0.07569 ],
      #     # [0.      ,  0.      ,  0.      ,  1.      ],
      #     [ 0.999727, -0.019749,  0.012491, -0.160597],
      #     [ 0.019656,  0.999779,  0.007515, -0.018761],
      #     [-0.012636, -0.007267,  0.999894,  1.354525],
      #     [ 0.      ,  0.      ,  0.      ,  1.      ],
      # ])
      # T_applanix_lidar = T_agt_apd @ T_applanix_lidar
      # T_applanix_lidar[:2, 3] = 0.
      # T_applanix_lidar[2, 3] = 0.31601375
      print("T_applanix_lidar after:\n", T_applanix_lidar)
      return T_applanix_lidar
    else:
      raise ValueError("Unknown sensor:", sensor)
  return _func

def get_frame_func(sensor):
  def _func(sequence, i):
    if sensor == "aeva":
      return sequence.aeva_frames[i]
    elif sensor == "velodyne":
      return sequence.lidar_frames[i]
    else:
      raise ValueError("Unknown sensor:", sensor)
  return _func


def main(dataset_dir, result_dir, sensor):
  result_dir = osp.normpath(result_dir)
  files = [file for file in os.listdir(result_dir) if not file.endswith("_dual_poses.txt") and file.endswith("_poses.txt")]
  sequences = [file.split("_")[0] for file in files]
  print("Result Directory:", result_dir)
  print("Odometry Sequences:", sequences)
  print("Dataset Directory:", dataset_dir)

  dataset = BoreasDataset(osp.normpath(dataset_dir), [[seq] for seq in sequences])

  # sensor specific stuff
  get_T_applanix_sensor = get_T_applanix_sensor_func(sensor)
  get_frame = get_frame_func(sensor)

  output_dir = osp.join(result_dir, "boreas_odometry_result")
  os.makedirs(output_dir, exist_ok=True)

  for sequence in dataset.sequences:
    print("Processing sequence:", sequence.ID)

    T_applanix_sensor = get_T_applanix_sensor(sequence)

    result = osp.join(result_dir, sequence.ID + "_poses.txt")
    converted_result = []

    with open(result, 'r') as file:
      reader = list(csv.reader(file, delimiter=' '))
      for i, row in enumerate(reader):

        timestamp = get_frame(sequence, i).timestamp_micro

        T_s0_st = np.eye(4)
        T_s0_st[:3] = np.array(row).reshape(3, 4)
        T_a0_at = T_applanix_sensor @ T_s0_st @ get_inverse_tf(T_applanix_sensor)
        T_at_a0_trunc = get_inverse_tf(T_a0_at).flatten().tolist()[:12]

        converted_result.append([timestamp] + T_at_a0_trunc)

    with open(osp.join(output_dir,  sequence.ID + ".txt"), "+w") as file:
      writer = csv.writer(file, delimiter=' ')
      writer.writerows(converted_result)
      print("Written to file:", osp.join(output_dir, sequence.ID + ".txt"))

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # Assuming following path structure:
  # <rosbag name>/metadata.yaml
  # <rosbag name>/<rosbag name>_0.db3
  parser.add_argument('--dataset', default=os.getcwd(), type=str, help='path to boreas dataset (contains boreas-*)')
  parser.add_argument('--path', default=os.getcwd(), type=str, help='path to vtr folder (default: os.getcwd())')
  parser.add_argument('--sensor', default="velodyne", type=str, help='aeva or velodyne')

  args = parser.parse_args()

  main(args.dataset, args.path, args.sensor)