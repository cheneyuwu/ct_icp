#include "steam_icp/datasets/utils.hpp"

namespace steam_icp {

namespace {

double translationError(const Eigen::Matrix4d &pose_error) { return pose_error.block<3, 1>(0, 3).norm(); }

double translationError2D(const Eigen::Matrix4d &pose_error) { return pose_error.block<2, 1>(0, 3).norm(); }

double rotationError(Eigen::Matrix4d &pose_error) {
  double a = pose_error(0, 0);
  double b = pose_error(1, 1);
  double c = pose_error(2, 2);
  double d = 0.5 * (a + b + c - 1.0);
  return std::acos(std::max(std::min(d, 1.0), -1.0));
}

std::vector<double> trajectoryDistances(const ArrayPoses &poses) {
  std::vector<double> dist(1, 0.0);
  for (size_t i = 1; i < poses.size(); i++) dist.push_back(dist[i - 1] + translationError(poses[i - 1] - poses[i]));
  return dist;
}

int lastFrameFromSegmentLength(const std::vector<double> &dist, int first_frame, double len) {
  for (int i = first_frame; i < (int)dist.size(); i++)
    if (dist[i] > dist[first_frame] + len) return i;
  return -1;
}

void computeMeanRPE(const ArrayPoses &poses_gt, const ArrayPoses &poses_result, Sequence::SeqError &seq_err) {
  // static parameter
  double lengths[] = {100, 200, 300, 400, 500, 600, 700, 800};
  size_t num_lengths = sizeof(lengths) / sizeof(double);

  // parameters
  int step_size = 10;  // every 10 frame (= every second for LiDAR at 10Hz)

  // pre-compute distances (from ground truth as reference)
  std::vector<double> dist = trajectoryDistances(poses_gt);

  int num_total = 0;
  double mean_rpe = 0;
  double mean_rpe_2d = 0;
  // for all start positions do
  for (int first_frame = 0; first_frame < (int)poses_gt.size(); first_frame += step_size) {
    // for all segment lengths do
    for (size_t i = 0; i < num_lengths; i++) {
      // current length
      double len = lengths[i];

      // compute last frame
      int last_frame = lastFrameFromSegmentLength(dist, first_frame, len);

      // next frame if sequence not long enough
      if (last_frame == -1) continue;

      // compute translational errors
      Eigen::Matrix4d pose_delta_gt = poses_gt[first_frame].inverse() * poses_gt[last_frame];
      Eigen::Matrix4d pose_delta_result = poses_result[first_frame].inverse() * poses_result[last_frame];
      Eigen::Matrix4d pose_error = pose_delta_result.inverse() * pose_delta_gt;
      double t_err = translationError(pose_error);
      double t_err_2d = translationError2D(pose_error);
      double r_err = rotationError(pose_error);
      seq_err.tab_errors.emplace_back(t_err / len, r_err / len);

      mean_rpe += t_err / len;
      mean_rpe_2d += t_err_2d / len;
      num_total++;
    }
  }

  seq_err.mean_rpe = ((mean_rpe / static_cast<double>(num_total)) * 100.0);
  seq_err.mean_rpe_2d = ((mean_rpe_2d / static_cast<double>(num_total)) * 100.0);
}

}  // namespace

Sequence::SeqError evaluateOdometry(const ArrayPoses &poses_gt, const ArrayPoses &poses_estimated) {
  Sequence::SeqError seq_err;

  // Compute Mean and Max APE (Mean and Max Absolute Pose Error)
  seq_err.mean_ape = 0.0;
  seq_err.max_ape = 0.0;
  for (size_t i = 0; i < poses_gt.size(); i++) {
    double t_ape_err = translationError(poses_estimated[i].inverse() * poses_gt[i]);
    seq_err.mean_ape += t_ape_err;
    if (seq_err.max_ape < t_ape_err) {
      seq_err.max_ape = t_ape_err;
    }
  }
  seq_err.mean_ape /= static_cast<double>(poses_gt.size());

  // Compute Mean and Max Local Error
  seq_err.mean_local_err = 0.0;
  seq_err.max_local_err = 0.0;
  seq_err.index_max_local_err = 0;
  for (int i = 1; i < (int)poses_gt.size(); i++) {
    double t_local_err = fabs((poses_gt[i].block<3, 1>(0, 3) - poses_gt[i - 1].block<3, 1>(0, 3)).norm() -
                              (poses_estimated[i].block<3, 1>(0, 3) - poses_estimated[i - 1].block<3, 1>(0, 3)).norm());
    seq_err.mean_local_err += t_local_err;
    if (seq_err.max_local_err < t_local_err) {
      seq_err.max_local_err = t_local_err;
      seq_err.index_max_local_err = i;
    }
  }
  seq_err.mean_local_err /= static_cast<double>(poses_gt.size() - 1);

  // Compute sequence mean RPE errors
  computeMeanRPE(poses_gt, poses_estimated, seq_err);

  return seq_err;
}

}  // namespace steam_icp