import os
import os.path as osp
import numpy as np
from pyboreas.utils.odometry import read_traj_file, read_traj_file_gt

from pylgmath import Transformation
from pysteam.problem import OptimizationProblem, StaticNoiseModel, L2LossFunc, WeightedLeastSquareCostTerm
from pysteam.solver import GaussNewtonSolver
from pysteam.evaluable.se3 import SE3StateVar, compose_rinv, compose, tran2vec

np.set_printoptions(precision=6, suppress=True)


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


def trajectory_distances(poses):
  """Calculates path length along the trajectory.
    Args:
        poses (List[np.ndarray]): list of 4x4 poses (T_k_i, 'i' is a fixed reference frame)
    Returns:
        List[float]: distance along the trajectory, increasing as a function of time / list index
    """
  dist = [0]
  for i in range(1, len(poses)):
    P1 = get_inverse_tf(poses[i - 1])
    P2 = get_inverse_tf(poses[i])
    dx = P1[0, 3] - P2[0, 3]
    dy = P1[1, 3] - P2[1, 3]
    dz = P1[2, 3] - P2[2, 3]
    dist.append(dist[i - 1] + np.sqrt(dx**2 + dy**2 + dz**2))
  return dist


def last_frame_from_segment_length(dist, first_frame, length):
  """Retrieves the index of the last frame for our current analysis.
        last_frame should be 'dist' meters away from first_frame in terms of distance traveled along the trajectory.
    Args:
        dist (List[float]): distance along the trajectory, increasing as a function of time / list index
        first_frame (int): index of the starting frame for this sequence
        length (float): length of the current segment being evaluated
    Returns:
        last_frame (int): index of the last frame in this segment
    """
  for i in range(first_frame, len(dist)):
    if dist[i] > dist[first_frame] + length:
      return i
  return -1


def calc_sequence_errors(poses_gt, poses_pred, step_size):
  """Calculate the translation and rotation error for each subsequence across several different lengths.
    Args:
        T_gt (List[np.ndarray]): each entry in list is 4x4 transformation matrix, ground truth transforms
        T_pred (List[np.ndarray]): each entry in list is 4x4 transformation matrix, predicted transforms
        step_size (int): step size applied for computing distances travelled
    Returns:
        err (List[Tuple]): each entry in list is [first_frame, r_err, t_err, length, speed]
        lengths (List[int]): list of lengths that odometry is evaluated at
    """
  lengths = [100, 200, 300, 400, 500, 600, 700, 800]
  # Pre-compute distances from ground truth as reference
  dist = trajectory_distances(poses_gt)

  pred_T_v_vpn, gt_T_v_vpn = [], []

  for first_frame in range(0, len(poses_gt), step_size):
    for length in lengths:
      last_frame = last_frame_from_segment_length(dist, first_frame, length)
      if last_frame == -1:
        continue

      tmp = poses_pred[first_frame] @ get_inverse_tf(poses_pred[last_frame])
      tmp = SE3StateVar(Transformation(T_ba=tmp), locked=True)
      pred_T_v_vpn.append(tmp)

      tmp = poses_gt[first_frame] @ get_inverse_tf(poses_gt[last_frame])
      tmp = SE3StateVar(Transformation(T_ba=tmp), locked=True)
      gt_T_v_vpn.append(tmp)

  return gt_T_v_vpn, pred_T_v_vpn


seq = 'boreas-2022-05-13-11-47'
gt_dir = '/home/yuchen/ASRL/data/boreas/sequences'
pred_dir = '/home/yuchen/ASRL/temp/cticp/boreas/lidar/elastic/boreas_odometry_result'

# load predictions
pred_T_vi, _ = read_traj_file(osp.join(pred_dir, seq + ".txt"))  # T_applanix_world
print(len(pred_T_vi))

# load ground truth poses
filepath = os.path.join(gt_dir, seq, 'applanix/lidar_poses.csv')  # use 'lidar_poses.csv' for groundtruth
T_calib = np.loadtxt(os.path.join(gt_dir, seq, 'calib/T_applanix_lidar.txt'))
gt_T_vi, _ = read_traj_file_gt(filepath, T_calib, 3)
print(len(gt_T_vi))

gt_T_v_vpn, pred_T_v_vpn = calc_sequence_errors(gt_T_vi, pred_T_vi, 10)

###############################################################################################

T_ab = SE3StateVar(Transformation(T_ba=np.eye(4)))  #   T_ab = T gt pred

noise_model = StaticNoiseModel(np.eye(6))
loss_func = L2LossFunc()
cost_terms = []
for i in range(len(pred_T_v_vpn)):
  est_gt_T_v_vp1 = compose_rinv(compose(T_ab, pred_T_v_vpn[i]), T_ab)
  error_func = tran2vec(compose_rinv(est_gt_T_v_vp1, gt_T_v_vpn[i]))
  cost_terms.append(WeightedLeastSquareCostTerm(error_func, noise_model, loss_func))

opt_prob = OptimizationProblem()
opt_prob.add_state_var(T_ab)
opt_prob.add_cost_term(*cost_terms)

gauss_newton = GaussNewtonSolver(opt_prob, verbose=True, max_iterations=100)
gauss_newton.optimize()

print(T_ab.value)